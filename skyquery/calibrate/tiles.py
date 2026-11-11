"""Build a slice-aided (SAHI-style) YOLO training set from VisDrone, optionally holding classes out.

For each training image: overlapping 640 px tiles (20% overlap) plus the full
frame. Ignored regions are painted grey so they act as neither positives nor
negatives. Held-out classes are removed from the labels *and* class ids are
re-indexed, so the detector never sees their names or boxes.

    python -m skyquery.calibrate.tiles --src data/VisDrone2019-DET-train --val data/VisDrone2019-DET-val \
        --out runs/vd_heldout --holdout bus bicycle awning-tricycle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..classes import CLASSES
from ..eval.visdrone import load_split


def tile_origins(size: int, tile: int, overlap: float) -> list[int]:
    if size <= tile:
        return [0]
    step = int(tile * (1 - overlap))
    xs = list(range(0, size - tile + 1, step))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def yolo_lines(boxes: np.ndarray, cls: np.ndarray, x0: int, y0: int, tw: int, th: int, min_vis: float = 0.5) -> list[str]:
    out = []
    for (x1, y1, x2, y2), c in zip(boxes, cls):
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        cx1, cy1 = max(x1, x0), max(y1, y0)
        cx2, cy2 = min(x2, x0 + tw), min(y2, y0 + th)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if (cx2 - cx1) * (cy2 - cy1) / area < min_vis:
            continue
        out.append(f"{int(c)} {((cx1 + cx2) / 2 - x0) / tw:.6f} {((cy1 + cy2) / 2 - y0) / th:.6f} {(cx2 - cx1) / tw:.6f} {(cy2 - cy1) / th:.6f}")
    return out


def build(src: Path, out: Path, split: str, keep: list[str], tile: int = 640, overlap: float = 0.2, tiles: bool = True, limit: int = 0):
    remap = {CLASSES.index(c): i for i, c in enumerate(keep)}
    gts = load_split(src)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for k, (stem, g) in enumerate(sorted(gts.items())):
        if limit and k >= limit:
            break
        im = Image.open(src / "images" / f"{stem}.jpg").convert("RGB")
        a = np.asarray(im).copy()
        for x1, y1, x2, y2 in g.ignore.astype(int):
            a[max(y1, 0):y2, max(x1, 0):x2] = 114
        sel = np.isin(g.cls, list(remap))
        boxes, cls = g.boxes[sel], np.array([remap[int(c)] for c in g.cls[sel]], int)
        H, W = a.shape[:2]
        views = [(0, 0, W, H, "full")]
        if tiles:
            views += [(x0, y0, min(tile, W), min(tile, H), f"t{x0}_{y0}") for y0 in tile_origins(H, tile, overlap) for x0 in tile_origins(W, tile, overlap)]
        for x0, y0, tw, th, tag in views:
            lines = yolo_lines(boxes, cls, x0, y0, tw, th)
            if tag != "full" and not lines:
                continue  # skip empty tiles
            name = f"{stem}_{tag}"
            Image.fromarray(a[y0:y0 + th, x0:x0 + tw]).save(out / "images" / split / f"{name}.jpg", quality=92)
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="VisDrone train split")
    ap.add_argument("--val", required=True, help="VisDrone val split (full frames only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    keep = [c for c in CLASSES if c not in set(a.holdout)]
    out = Path(a.out)
    n_tr = build(Path(a.src), out, "train", keep, limit=a.limit)
    n_va = build(Path(a.val), out, "val", keep, tiles=False, limit=a.limit)
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(keep)))
    (out / "holdout.json").write_text(json.dumps({"keep": keep, "holdout": a.holdout, "train_views": n_tr, "val_images": n_va}, indent=2))
    print(f"{n_tr} training views, {n_va} val images -> {yaml}")


if __name__ == "__main__":
    main()
