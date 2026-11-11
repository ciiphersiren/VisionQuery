"""Run the detector once per image/scale and cache raw predictions.

Everything downstream (AP, counting, threshold tuning, bootstrap) reads the
cache, so the expensive forward passes happen exactly once.

    python -m skyquery.calibrate.cache_preds --split data/VisDrone2019-DET-val \
        --scale 1280 --out cache/val_1280.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="VisDrone split dir with images/ and annotations/")
    ap.add_argument("--weights", default="weights/yolo26s_visdrone_sahi.pt")
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(a.weights)
    from ..classes import CLASS_TO_ID

    # map the model's class ids back to VisDrone ids (identity for the 10-class model,
    # a re-index for the held-out-class model trained on a subset)
    lut = np.array([CLASS_TO_ID[model.names[i]] for i in range(len(model.names))], np.float32)
    imgs = sorted((Path(a.split) / "images").glob("*.jpg"))
    if a.limit:
        imgs = imgs[: a.limit]
    out = Path(a.out)
    done: dict[str, np.ndarray] = {}
    if out.exists():
        with np.load(out) as z:
            done = {k: z[k] for k in z.files}
    todo = [p for p in imgs if p.stem not in done and f"{p.stem}__shape" not in done]
    print(f"{len(imgs)} images, {len(done)//2} cached, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        r = model.predict(str(p), imgsz=a.scale, conf=a.conf, max_det=a.max_det, device=a.device, verbose=False)[0]
        b = r.boxes
        arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], lut[b.cls.cpu().numpy().astype(int)][:, None]], 1).astype(np.float32)
        done[p.stem] = arr
        done[f"{p.stem}__shape"] = np.array(r.orig_shape[::-1], np.int32)  # (w, h)
        if i % 25 == 0 or i == len(todo):
            np.savez_compressed(out, **done)
            el = time.time() - t0
            print(f"[{a.scale}] {i}/{len(todo)}  {el/i:.2f}s/img  eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)


def load_cache(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    """Return ({stem: (N,6) preds}, {stem: (w, h)})."""
    preds, shapes = {}, {}
    with np.load(path) as z:
        for k in z.files:
            if k.endswith("__shape"):
                w, h = z[k]
                shapes[k[: -len("__shape")]] = (int(w), int(h))
            else:
                preds[k] = z[k]
    return preds, shapes


if __name__ == "__main__":
    main()
