"""Training-set builder: the YOLO26m-Tile "D2" recipe for any dataset in the common format.

Per image: ignore regions painted gray (114) -> saved as a full image with YOLO labels; then 640 and 960 px
tiles with 25% overlap (step = T * 0.75); a cut box is kept if >= 60% of it lies inside the tile, otherwise
painted gray; empty tiles kept with probability 0.1, seeded per image (random.Random(image key)).
Same arithmetic as the visdrone-training notebook (Cell 3), including the 6-decimal YOLO label round-trip,
so VisDrone input reproduces the original 6,471 + 63,745 images.
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TILE_SIZES = (640, 960)
OVERLAP = 0.25
MIN_VIS = 0.6
EMPTY_KEEP = 0.1
GREY = 114


def starts(L, T, step):
    if L <= T:
        return [0]
    s = list(range(0, L - T + 1, step))
    if s[-1] + T < L:
        s.append(L - T)
    return s


def yolo_lines(boxes, W, H):
    """Clip to the image, drop empty boxes, format as YOLO lines (class cx cy w h, 6 decimals)."""
    out = []
    for c, x1, y1, x2, y2, *_ in boxes.tolist():      # Python floats: same rounding as the original notebook
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(f"{int(c)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return out


def _boxes_from_lines(lines, W, H):
    out = []
    for r in lines:
        c, cx, cy, w, h = r.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append([int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out, dtype=np.float32).reshape(-1, 5)


def _process(args):
    import cv2
    rec, out_full, out_tiles, tile_sizes, overlap, min_vis, empty_keep, grey = args
    rng = random.Random(rec.key)
    img = cv2.imread(str(rec.path))
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in rec.ignore.astype(int):
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        img[max(0, y):y + h, max(0, x):x + w] = grey
    lines = yolo_lines(rec.boxes, W, H)
    cv2.imwrite(str(out_full / "images" / f"{rec.key}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out_full / "labels" / f"{rec.key}.txt").write_text("\n".join(lines))
    boxes = _boxes_from_lines(lines, W, H)
    n = 0
    for T in tile_sizes:
        if T >= max(W, H):
            continue
        step = int(T * (1 - overlap))
        for y0 in starts(H, T, step):
            for x0 in starts(W, T, step):
                x1, y1, x2, y2 = x0, y0, min(x0 + T, W), min(y0 + T, H)
                tile = img[y1:y2, x1:x2].copy()
                tw, th = x2 - x1, y2 - y1
                tl = []
                if len(boxes):
                    bx1 = np.clip(boxes[:, 1], x1, x2); by1 = np.clip(boxes[:, 2], y1, y2)
                    bx2 = np.clip(boxes[:, 3], x1, x2); by2 = np.clip(boxes[:, 4], y1, y2)
                    inter = (bx2 - bx1) * (by2 - by1)
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
                    vis = np.where(area > 0, inter / area, 0)
                    for i in np.where(inter > 0)[0]:
                        if vis[i] >= min_vis:
                            tl.append(f"{int(boxes[i, 0])} {((bx1[i] + bx2[i]) / 2 - x1) / tw:.6f} "
                                      f"{((by1[i] + by2[i]) / 2 - y1) / th:.6f} {(bx2[i] - bx1[i]) / tw:.6f} "
                                      f"{(by2[i] - by1[i]) / th:.6f}")
                        else:   # gray out badly-cut objects so they aren't unlabeled positives
                            tile[int(by1[i] - y1):int(by2[i] - y1), int(bx1[i] - x1):int(bx2[i] - x1)] = grey
                if not tl and rng.random() > empty_keep:
                    continue
                name = f"{rec.key}_t{T}_{x0}_{y0}"
                cv2.imwrite(str(out_tiles / "images" / f"{name}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out_tiles / "labels" / f"{name}.txt").write_text("\n".join(tl))
                n += 1
    return n


def build_d2(split, out_dir, tile_sizes=TILE_SIZES, overlap=OVERLAP, min_vis=MIN_VIS, empty_keep=EMPTY_KEEP,
             grey=GREY, workers=None):
    """Write <out_dir>/full/{images,labels} and <out_dir>/tiles/{images,labels}. Returns (n_full, n_tiles)."""
    out_dir = Path(out_dir)
    full, tiles = out_dir / "full", out_dir / "tiles"
    for d in (full, tiles):
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    jobs = [(r, full, tiles, tuple(tile_sizes), overlap, min_vis, empty_keep, grey) for r in split.images]
    if workers == 0:
        counts = [_process(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
            counts = list(ex.map(_process, jobs, chunksize=16))
    return len(jobs), int(sum(counts))


# --------------------------------------------------------------------------- YOLO26s recipe (teammate's prompt1/prompt2 notebooks)
SF_TILE = 640
SF_OVERLAP = 0.20
SF_MIN_VIS = 0.30
SF_MIN_SIDE = 2
SF_EMPTY_KEEP = 0.10
SF_SEED = 42


def sf_tile_plan(rec, W, H):
    """Kept tiles of one image under the YOLO26s recipe: [(x1, y1, x2, y2, label lines)]. Labels only, so it can be counted without images.
    A box stays if >= 30% of it is inside the tile and both clipped sides are >= 2 px; other cut boxes are dropped, not painted.
    Empty tiles are kept with probability 0.1, seeded per image."""
    from sahi.slicing import get_slice_bboxes
    rng = random.Random(f"{SF_SEED}-{rec.key}")
    b = rec.boxes
    area = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
    out = []
    for x1, y1, x2, y2 in get_slice_bboxes(image_height=H, image_width=W, slice_height=SF_TILE, slice_width=SF_TILE, auto_slice_resolution=False,
                                           overlap_height_ratio=SF_OVERLAP, overlap_width_ratio=SF_OVERLAP):
        cx1, cy1 = np.clip(b[:, 1], x1, x2), np.clip(b[:, 2], y1, y2)
        cx2, cy2 = np.clip(b[:, 3], x1, x2), np.clip(b[:, 4], y1, y2)
        cw, ch = cx2 - cx1, cy2 - cy1
        keep = (cw >= SF_MIN_SIDE) & (ch >= SF_MIN_SIDE) & (cw * ch >= SF_MIN_VIS * area)
        if not keep.any() and rng.random() > SF_EMPTY_KEEP:
            continue
        tw, th = x2 - x1, y2 - y1
        lines = [f"{int(b[i, 0])} {(cx1[i] - x1 + cw[i] / 2) / tw:.6f} {(cy1[i] - y1 + ch[i] / 2) / th:.6f} {cw[i] / tw:.6f} {ch[i] / th:.6f}"
                 for i in np.where(keep)[0]]
        out.append((x1, y1, x2, y2, lines))
    return out


def _process_sf(args):
    import cv2
    rec, out_full, out_tiles, grey = args
    img = cv2.imread(str(rec.path))
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in rec.ignore.astype(int):
        img[max(0, y1):y2, max(0, x1):x2] = grey
    cv2.imwrite(str(out_full / "images" / f"{rec.key}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out_full / "labels" / f"{rec.key}.txt").write_text("\n".join(yolo_lines(rec.boxes, W, H)))
    plan = sf_tile_plan(rec, W, H)
    for x1, y1, x2, y2, lines in plan:
        name = f"{rec.key}_{x1}_{y1}"
        cv2.imwrite(str(out_tiles / "images" / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_tiles / "labels" / f"{name}.txt").write_text("\n".join(lines))
    return len(plan)


def build_sf(split, out_dir, grey=GREY, workers=None):
    """YOLO26s training set: <out_dir>/full and <out_dir>/tiles, each with images/ and labels/. Returns (n_full, n_tiles)."""
    out_dir = Path(out_dir)
    full, tiles = out_dir / "full", out_dir / "tiles"
    for d in (full, tiles):
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
    jobs = [(r, full, tiles, grey) for r in split.images]
    if workers == 0:
        counts = [_process_sf(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
            counts = list(ex.map(_process_sf, jobs, chunksize=16))
    return len(jobs), int(sum(counts))


def write_yaml(path, root, train_dirs, val_dir, names):
    """Ultralytics dataset yaml. train_dirs/val_dir are relative to root."""
    tr = "\n".join(f"  - {d}" for d in train_dirs)
    nb = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    Path(path).write_text(f"path: {root}\ntrain:\n{tr}\nval: {val_dir}\nnames:\n{nb}\n")
    return Path(path)
