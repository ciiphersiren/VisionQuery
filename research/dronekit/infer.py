"""Inference engine ("evaluation v2"), raw-detection cache, and merge setups.

Same logic as evaluation(1).ipynb: SAHI's slicer (`get_slice_bboxes`) and SAHI's own merge
functions (`batched_nms`, `batched_greedy_nmm`) called directly with every setting explicit,
so nothing can switch silently. Never call `get_sliced_prediction`.

Pass keys: "full@S" = whole image at imgsz S; "sT@S" = T x T slices, each resized to S.
Detections are (N, 6) float32 arrays: x1, y1, x2, y2, score, class_id (0-based).
Slice passes also return slice boxes (N, 4) so the boundary filter can run later on CPU.

Cache: one parquet per model x split holding every raw pass down to CONF (0.01), plus a
JSON sidecar with timings and settings. Thresholds and merges can change later without the GPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONF = 0.01           # detections kept down to 0.01
MAX_DET = 1000        # per forward pass
OVERLAP = 0.25        # slice overlap
SLICE_BATCH = 16
EDGE_MARGIN = 2.0     # boundary filter: px from an interior slice edge

# Named setups. "passes" are combined for one model, then merged.
SETUPS = {
    "whole1536":   dict(passes=["full@1536"], method=None),
    "best_sliced": dict(passes=["s512@1280", "full@1536"], method="NMS", metric="IOU", thr=0.6, edge=True),
    "fast":        dict(passes=["s512@1024", "full@1536"], method="NMS", metric="IOU", thr=0.6, edge=True),
}
ENSEMBLE_WBF_THR = 0.65


def parse_pass(key: str):
    kind, sz = key.split("@")
    return ("full", None, int(sz)) if kind == "full" else ("slice", int(kind[1:]), int(sz))


def slice_boxes(H: int, W: int, T: int, overlap: float = OVERLAP):
    from sahi.slicing import get_slice_bboxes
    return get_slice_bboxes(image_height=H, image_width=W, slice_height=T, slice_width=T,
                            auto_slice_resolution=False, overlap_height_ratio=overlap, overlap_width_ratio=overlap)


# --------------------------------------------------------------------------- running a model
def _arr(r, cls_map=None):
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6), np.float32)
    a = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()].astype(np.float32)
    if cls_map is not None:          # model class index -> dataset class id
        a[:, 5] = np.vectorize(cls_map.__getitem__)(a[:, 5].astype(int))
    return a


def run_pass_on_image(model, img, key, *, conf=CONF, max_det=MAX_DET, half=True, device=0,
                      slice_batch=SLICE_BATCH, overlap=OVERLAP, cls_map=None):
    """One pass on one BGR image. Returns (dets (N,6), slice_boxes (N,4) or None)."""
    kind, T, S = parse_pass(key)
    kw = dict(imgsz=S, conf=conf, max_det=max_det, half=half, device=device, verbose=False)
    if kind == "full":
        return _arr(model.predict(img, **kw)[0], cls_map), None
    H, W = img.shape[:2]
    boxes = slice_boxes(H, W, T, overlap)
    crops = [np.ascontiguousarray(img[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes]
    D, SB = [], []
    for i in range(0, len(crops), slice_batch):
        rs = model.predict(crops[i:i + slice_batch], **kw)
        for r, (x1, y1, x2, y2) in zip(rs, boxes[i:i + slice_batch]):
            a = _arr(r, cls_map)
            a[:, [0, 2]] += x1
            a[:, [1, 3]] += y1
            D.append(a)
            SB.append(np.tile(np.float32([x1, y1, x2, y2]), (len(a), 1)))
    return np.concatenate(D), np.concatenate(SB).reshape(-1, 4)


COLS = ["image", "pass", "x1", "y1", "x2", "y2", "score", "cls", "sx1", "sy1", "sx2", "sy2"]


def run_and_cache(model, split, passes, out_path, *, meta=None, cuda=True, **kw):
    """Run every pass on every image of `split`; write <out_path>.parquet + <out_path>.json.

    Skips work if the parquet already exists. Returns the cache path.
    Latency = GPU time per image per pass (torch.cuda.synchronize around each image).
    """
    import cv2
    out_path = Path(out_path)
    pq, js = out_path.with_suffix(".parquet"), out_path.with_suffix(".json")
    if pq.exists() and js.exists():
        return pq
    sync = (lambda: __import__("torch").cuda.synchronize()) if cuda else (lambda: None)
    frames, ms = [], {}
    for key in passes:
        t = 0.0
        for rec in split.images:
            img = cv2.imread(str(rec.path))
            sync(); t0 = time.perf_counter()
            d, sb = run_pass_on_image(model, img, key, **kw)
            sync(); t += time.perf_counter() - t0
            if sb is None:
                sb = np.full((len(d), 4), np.nan, np.float32)
            f = pd.DataFrame(np.c_[d, sb].astype(np.float32), columns=COLS[2:])
            f.insert(0, "pass", key); f.insert(0, "image", rec.key)
            frames.append(f)
        ms[key] = 1000 * t / max(len(split), 1)
        print(f"  {out_path.name} {key}: {ms[key]:.0f} ms/img")
    df = pd.concat(frames, ignore_index=True)
    df["image"] = df["image"].astype("category"); df["pass"] = df["pass"].astype("category")
    df["cls"] = df["cls"].astype(np.int16)
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq, compression="zstd", index=False)
    js.write_text(json.dumps(dict(meta or {}, split=split.name, dataset=split.dataset, n_images=len(split),
                                  passes=list(passes), ms_per_img=ms, conf=kw.get("conf", CONF),
                                  max_det=kw.get("max_det", MAX_DET), overlap=kw.get("overlap", OVERLAP),
                                  rows=len(df)), indent=2))
    return pq


def load_cache(path):
    """-> (raw, meta). raw[pass][image_key] = (dets (N,6), slice_boxes (N,4) or None)."""
    path = Path(path)
    df = pd.read_parquet(path.with_suffix(".parquet"))
    meta = json.loads(path.with_suffix(".json").read_text())
    raw = {p: {} for p in meta.get("passes", [])}   # a pass with zero detections still exists
    for (p, im), g in df.groupby(["pass", "image"], observed=True, sort=False):
        a = g[COLS[2:]].to_numpy(np.float32)
        raw.setdefault(str(p), {})[str(im)] = (a[:, :6], None if parse_pass(str(p))[0] == "full" else a[:, 6:])
    return raw, meta


# --------------------------------------------------------------------------- merging
def edge_filter(d, sb, W, H, margin=EDGE_MARGIN):
    """Drop slice detections within `margin` px of an *interior* slice edge (a cut object).
    Edges lying on the image border are not interior and never filter anything."""
    if sb is None or len(d) == 0:
        return d
    x1, y1, x2, y2 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    sx1, sy1, sx2, sy2 = sb[:, 0], sb[:, 1], sb[:, 2], sb[:, 3]
    cut = (((sx1 > 0) & (x1 <= sx1 + margin)) | ((sy1 > 0) & (y1 <= sy1 + margin)) |
           ((sx2 < W) & (x2 >= sx2 - margin)) | ((sy2 < H) & (y2 >= sy2 - margin)))
    return d[~cut]


def _overlap(a, b, metric):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    den = min(aa, ab) if metric == "IOS" else (aa + ab - inter)
    return inter / den if den > 0 else 0.0


def merge(parts, method="NMS", metric="IOU", thr=0.5):
    """Class-wise merge with SAHI's own functions."""
    from sahi.postprocess.combine import batched_greedy_nmm, batched_nms
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros((0, 6), np.float32)
    P = np.concatenate(parts)
    if method == "NMS":
        return P[batched_nms(P, match_metric=metric, match_threshold=thr)]
    if method == "GREEDYNMM":                     # same logic as SAHI's _apply_merge: union box, max score
        out = []
        for keep, mlist in batched_greedy_nmm(P, match_metric=metric, match_threshold=thr).items():
            b = P[keep].copy()
            for j in mlist:
                if _overlap(b, P[j], metric) >= thr:
                    b[0], b[1] = min(b[0], P[j, 0]), min(b[1], P[j, 1])
                    b[2], b[3] = max(b[2], P[j, 2]), max(b[3], P[j, 3]); b[4] = max(b[4], P[j, 4])
            out.append(b)
        return np.array(out, np.float32).reshape(-1, 6)
    raise ValueError(method)


def wbf(dets_per_model, W, H, thr=ENSEMBLE_WBF_THR, skip=CONF):
    """Weighted box fusion across models (each model already merged)."""
    from ensemble_boxes import weighted_boxes_fusion
    bl, sl, ll = [], [], []
    for d in dets_per_model:
        bl.append(np.clip(d[:, :4] / [W, H, W, H], 0, 1).tolist()); sl.append(d[:, 4].tolist()); ll.append(d[:, 5].tolist())
    b, s, l = weighted_boxes_fusion(bl, sl, ll, iou_thr=thr, skip_box_thr=skip)
    return np.c_[np.asarray(b).reshape(-1, 4) * [W, H, W, H], s, l].astype(np.float32).reshape(-1, 6)


def apply_setup(raw, split, setup):
    """Merge cached raw passes into final detections. Returns (dets {image_key: (N,6)}, merge ms/img)."""
    if isinstance(setup, str):
        setup = SETUPS[setup]
    passes = setup["passes"]
    missing = [p for p in passes if p not in raw]
    if missing:
        raise KeyError(f"passes not in cache: {missing}")
    out, t0 = {}, time.perf_counter()
    single_full = len(passes) == 1 and parse_pass(passes[0])[0] == "full"
    for rec in split.images:
        parts = []
        for k in passes:
            d, sb = raw[k].get(rec.key, (np.zeros((0, 6), np.float32), None))
            parts.append(edge_filter(d, sb, rec.width, rec.height) if setup.get("edge") else d)
        out[rec.key] = parts[0] if single_full else merge(parts, setup["method"], setup["metric"], setup["thr"])
    return out, 1000 * (time.perf_counter() - t0) / max(len(split), 1)


def ensemble(dets_list, split, thr=ENSEMBLE_WBF_THR):
    """WBF of several models' merged detections, per image."""
    return {r.key: wbf([d[r.key] for d in dets_list], r.width, r.height, thr) for r in split.images}


def setup_ms(meta, setup, merge_ms=0.0):
    """Latency of a setup = sum of its passes' GPU ms/img + merge time (as in eval v2)."""
    if isinstance(setup, str):
        setup = SETUPS[setup]
    return sum(meta["ms_per_img"][p] for p in setup["passes"]) + merge_ms
