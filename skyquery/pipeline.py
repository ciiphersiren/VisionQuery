"""SkyQuery: detector + planner + answer engine, wired together.

    sq = SkyQuery.from_config()
    out = sq.ask("frame.jpg", "how many cars are in the left half?")
    print(out.answer.text)
"""

from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .answer import Answer, Answerer
from .calibration import load_calibration
from .config import Config, load_config
from .detector import Detections, Detector
from .models import ModelSpec, load_models
from .planner.policy import Planner
from .openvocab import OpenVocab
from .planner.schema import Plan



@dataclass
class Result:
    plan: Plan
    answer: Answer
    detections: Detections
    image_id: str


def image_id_of(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def to_pil(image) -> tuple[Image.Image, bytes]:
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    elif isinstance(image, bytes):
        data = image
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
    else:
        raise TypeError(type(image))
    return Image.open(io.BytesIO(data)).convert("RGB"), data


def _taught_ap() -> dict[str, float]:
    """Per-concept AP of the drone-tuned YOLOE against SAM 3 on val, if the E3 report is present."""
    import json

    from .config import ROOT

    f = ROOT / "reports" / "ovd_e3_val.json"
    try:
        return json.loads(f.read_text())["groups"]["seen"]["per_concept"]["tuned"]
    except Exception:
        return {}


class SkyQuery:
    def __init__(self, cfg: Config):
        from .config import ROOT

        self.cfg = cfg
        self.models, default = load_models(cfg.models, ROOT, cfg.weights, cfg.calibration)
        self.default_model = cfg.model if cfg.model in self.models else default
        if not self.models[self.default_model].available:   # fall back to any model whose weights are on disk
            self.default_model = next((m for m, s in self.models.items() if s.available), self.default_model)
        self._calibs: dict[str, dict] = {}
        self._detectors: dict[tuple[str, str], Detector] = {}
        self.planner = Planner(cfg.planner_policy)
        self._cache: OrderedDict[tuple[str, str], Detections] = OrderedDict()
        self._images: OrderedDict[str, Image.Image] = OrderedDict()
        self.ovd = OpenVocab(cfg.ovd_weights, device=cfg.device, threshold=cfg.ovd_threshold, taught_weights=cfg.ovd_taught_weights or None,
                             taught_threshold=cfg.ovd_taught_threshold, taught_ap=_taught_ap()) if cfg.ovd_enabled else None
        self._ovd_cache: OrderedDict[tuple, Detections] = OrderedDict()

    @classmethod
    def from_config(cls, path: str | Path | None = None, **overrides) -> "SkyQuery":
        return cls(load_config(path, **overrides))

    # ------------------------------------------------------------------ models
    def spec(self, model: str | None = None) -> ModelSpec:
        return self.models[model or self.default_model]

    def calib(self, model: str | None = None, pipeline: str | None = None):
        s = self.spec(model)
        if s.id not in self._calibs:
            self._calibs[s.id] = load_calibration(s.calibration)
        return self._calibs[s.id][pipeline or s.default_pipeline]

    @property
    def calibs(self) -> dict:
        """Calibrations of the default model, by pipeline (kept for the original API)."""
        self.calib()
        return self._calibs[self.default_model]

    def resolve(self, model: str | None, pipeline: str | None) -> tuple[str, str]:
        """Fill in defaults: the model's own default pipeline when none (or the old global default) does not apply."""
        s = self.spec(model)
        if pipeline is None:
            pipeline = self.cfg.pipeline if self.cfg.pipeline in s.pipelines else s.default_pipeline
        return s.id, pipeline

    def detector(self, pipeline: str, model: str | None = None) -> Detector:
        s = self.spec(model)
        key = (s.id, pipeline)
        if key not in self._detectors:
            self._detectors[key] = Detector(
                s.weights, pipeline=pipeline, device=self.cfg.device,
                conf_floor=self.cfg.conf_floor, max_det=min(self.cfg.max_det, s.max_det) if self.cfg.max_det else s.max_det,
                fusion=self.cfg.fusion, fusion_iou=self.cfg.fusion_iou, half=self.cfg.half,
            )
        return self._detectors[key]

    # ------------------------------------------------------------------ images
    def register(self, image) -> str:
        pil, data = to_pil(image)
        iid = image_id_of(data)
        self._images[iid] = pil
        self._images.move_to_end(iid)
        while len(self._images) > self.cfg.cache_images:
            self._images.popitem(last=False)
        return iid

    def image(self, image_id: str) -> Image.Image:
        return self._images[image_id]

    def detect(self, image_id: str, pipeline: str | None = None, model: str | None = None) -> Detections:
        model, pipeline = self.resolve(model, pipeline)
        key = (image_id, model, pipeline)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = np.asarray(self._images[image_id])[:, :, ::-1]  # RGB -> BGR for ultralytics; KeyError if unknown
        det = self.detector(pipeline, model)(np.ascontiguousarray(img))
        det.meta["model"] = model
        self._cache[key] = det
        while len(self._cache) > self.cfg.cache_images * 2:
            self._cache.popitem(last=False)
        return det

    def open_vocab(self, iid: str, plan: Plan) -> dict[str, Detections]:
        """Run YOLOE for each out-of-vocabulary term (cached per image and prompt set)."""
        if not self.ovd or not plan.oov:
            return {}
        out = {}
        img = np.asarray(self._images[iid])[:, :, ::-1]
        for o in plan.oov:
            # sub-types are prompted alone: with the parent as a contrast prompt every box goes to the parent
            prompts = [o.term]
            key = (iid, tuple(prompts))
            if key not in self._ovd_cache:
                try:
                    self._ovd_cache[key] = self.ovd.detect(img, prompts)
                except Exception as e:  # weights missing / offline: fall back to the closed-set answer
                    print(f"[skyquery] open-vocabulary branch unavailable: {e}")
                    self.ovd = None
                    return {}
                while len(self._ovd_cache) > self.cfg.cache_images * 4:
                    self._ovd_cache.popitem(last=False)
            out[o.term] = self._ovd_cache[key]
        return out

    # ------------------------------------------------------------------ Q&A
    def plan(self, query: str) -> Plan:
        return self.planner(query)

    def ask(self, image, query: str, pipeline: str | None = None, count_threshold: float | None = None, model: str | None = None) -> Result:
        """Local use: ``image`` is a path, bytes or PIL image."""
        return self.ask_id(self.register(image), query, pipeline, count_threshold, model)

    def ask_id(self, iid: str, query: str, pipeline: str | None = None, count_threshold: float | None = None, model: str | None = None) -> Result:
        """API use: only previously registered image ids (never a file path). KeyError if unknown."""
        model, pipeline = self.resolve(model, pipeline)
        if iid not in self._images:
            raise KeyError(iid)
        det = self.detect(iid, pipeline, model)
        plan = self.plan(query)
        ovd = self.open_vocab(iid, plan)
        s = self.spec(model)
        ans = Answerer(self.calib(model, pipeline), count_threshold, self.cfg.ovd_threshold, self.cfg.ovd_subtype_threshold,
                       supported=s.classes, model_name=s.name)(plan, det, ovd)
        return Result(plan=plan, answer=ans, detections=det, image_id=iid)


def render(image: Image.Image, answer: Answer, out: str | Path | None = None) -> Image.Image:
    """Draw the region and evidence boxes; handy for the CLI and debugging."""
    from PIL import ImageDraw

    from .classes import CLASS_TO_ID

    palette = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#38d9a9", "#4dabf7", "#748ffc", "#da77f2", "#f783ac", "#a9e34b"]
    im = image.copy()
    d = ImageDraw.Draw(im, "RGBA")
    if answer.region != "whole":
        d.rectangle(answer.region_box, outline=(255, 255, 255, 220), width=3, fill=(255, 255, 255, 25))
    for ev in answer.evidence:
        col = palette[CLASS_TO_ID[ev.label]]
        d.rectangle(ev.box, outline=col, width=2 if ev.strength == "counted" else 1)
    if out:
        im.save(out)
    return im
