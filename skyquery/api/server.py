"""FastAPI server.

    uvicorn skyquery.api.server:app --port 8000
    # or: skyquery serve

Flow for a UI: upload once (POST /api/images) -> ask many questions
(POST /api/ask with the image_id). Detections are cached per image and
pipeline, so follow-up questions cost only the planner (~ms).
"""

from __future__ import annotations

import io
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..classes import CLASS_EMOJI, CLASSES
from ..config import ROOT
from ..detector import PIPELINES
from ..pipeline import SkyQuery
from ..regions import REGIONS

SAMPLES_DIR = ROOT / "samples"
FRONTEND_DIST = ROOT / "frontend" / "dist"


@lru_cache(maxsize=1)
def engine() -> SkyQuery:
    return SkyQuery.from_config()


app = FastAPI(title="VisionQuery", version=__version__, description="Ask questions about drone images. Answers are computed from detector boxes.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskBody(BaseModel):
    image_id: str
    query: str
    pipeline: str | None = None
    model: str | None = Field(default=None, description="model id from /api/meta; default = the server's default model")
    count_threshold: float | None = Field(default=None, ge=0.05, le=0.95, description="override all per-class count thresholds (UI slider)")


class DetectBody(BaseModel):
    image_id: str
    pipeline: str | None = None
    model: str | None = None
    min_conf: float | None = None


class PlanBody(BaseModel):
    query: str


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/api/meta")
def meta():
    sq = engine()
    return {
        "classes": [{"name": c, "emoji": CLASS_EMOJI[c]} for c in CLASSES],
        "regions": {k: v for k, v in REGIONS.items()},
        "pipelines": list(PIPELINES),
        "default_pipeline": sq.resolve(None, None)[1],
        "models": [s.public() for s in sq.models.values()],
        "default_model": sq.default_model,
        "planner_policy": sq.cfg.planner_policy,
        "llm_available": sq.planner.llm.available,
        "calibration": {
            p: {"count_thresholds": dict(zip(CLASSES, c.count_thr.round(3).tolist())), "exist_thresholds": dict(zip(CLASSES, c.exist_thr.round(3).tolist())), "exist_recall": c.exist_recall}
            for p, c in sq.calibs.items()
        },
        "examples": [
            "how many cars are in the left half?",
            "any buses?",
            "is anyone sitting in the top right corner?",
            "are there more motorbikes than bicycles?",
            "show me all the vans",
            "how many fire trucks are there?",
            "what's in the bottom half?",
        ],
    }


@app.post("/api/images")
async def upload(file: UploadFile = File(...)):
    sq = engine()
    data = await file.read()
    if len(data) > sq.cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "image too large")
    try:
        iid = sq.register(data)
    except Exception as e:  # PIL errors
        raise HTTPException(400, f"not an image: {e}")
    w, h = sq.image(iid).size
    return {"image_id": iid, "width": w, "height": h}


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    sq = engine()
    try:
        im = sq.image(image_id)
    except KeyError:
        raise HTTPException(404, "unknown image_id (upload again)")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/samples")
def samples():
    if not SAMPLES_DIR.exists():
        return []
    return [{"name": p.stem, "url": f"/api/samples/{p.name}"} for p in sorted(SAMPLES_DIR.glob("*.jpg"))]


@app.get("/api/samples/{name}")
def sample_file(name: str):
    p = (SAMPLES_DIR / name).resolve()
    if p.parent != SAMPLES_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/samples/{name}/load")
def load_sample(name: str):
    p = (SAMPLES_DIR / name).resolve()
    if p.parent != SAMPLES_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    sq = engine()
    iid = sq.register(p)
    w, h = sq.image(iid).size
    return {"image_id": iid, "width": w, "height": h}


def _check(model, pipeline):
    """400 for an unknown model or a pipeline the model was not evaluated with; 409 when its weights are not on disk."""
    sq = engine()
    if pipeline is not None and pipeline not in PIPELINES:
        raise HTTPException(400, f"pipeline must be one of {list(PIPELINES)}")
    if model is not None and model not in sq.models:
        raise HTTPException(400, f"model must be one of {list(sq.models)}")
    s = sq.spec(model)
    if pipeline is not None and pipeline not in s.pipelines:
        raise HTTPException(400, f"{s.name} runs with {s.pipelines}, not {pipeline}")
    if not s.available:
        raise HTTPException(409, f"{s.name} is not installed: put {', '.join(s.missing)} in skyquery/weights/")


@app.post("/api/plan")
def plan(body: PlanBody):
    return engine().plan(body.query).model_dump()


@app.post("/api/detect")
def detect(body: DetectBody):
    _check(body.model, body.pipeline)
    sq = engine()
    try:
        det = sq.detect(body.image_id, body.pipeline, body.model)
    except KeyError:
        raise HTTPException(404, "unknown image_id (upload again)")
    cal = sq.calib(*sq.resolve(body.model, body.pipeline))
    floor = body.min_conf if body.min_conf is not None else 0.0
    boxes = det.to_json(min_conf=floor)
    for b in boxes:
        k = CLASSES.index(b["label"])
        b["counted"] = bool(b["conf"] >= cal.count_thr[k])
    return {"width": det.width, "height": det.height, "meta": det.meta, "boxes": boxes}


@app.post("/api/ask")
def ask(body: AskBody):
    _check(body.model, body.pipeline)
    sq = engine()
    if not body.query.strip():
        raise HTTPException(400, "empty query")
    t0 = time.perf_counter()
    try:
        r = sq.ask_id(body.image_id, body.query, body.pipeline, body.count_threshold, body.model)
    except KeyError:
        raise HTTPException(404, "unknown image_id (upload again)")
    return {
        "image_id": r.image_id,
        "plan": r.plan.model_dump(),
        "answer": r.answer.model_dump(),
        "detector": r.detections.meta,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


REPORTS_DIR = ROOT / "reports"


@app.get("/api/metrics")
def metrics():
    """Evaluation reports for the metrics page (whatever has been produced so far)."""
    import json

    out = {}
    for p in sorted(REPORTS_DIR.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception:
            continue
    return out


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
