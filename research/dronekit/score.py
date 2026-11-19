"""Scorer ("evaluation v2"): COCO AP via pycocotools with the VisDrone settings.

- maxDets = [1, 100, 500]; every number reads the 500 index (`stats` hard-codes 100)
- ignore regions inserted as crowd boxes for every class -> detections inside are neither TP nor FP
- "others" already dropped by the loader
- size buckets: tiny < 16^2, small 16^2-32^2, medium+ >= 32^2 (plus COCO small < 32^2)
Also: per-class AP50, precision/recall at a fixed confidence, PR curves, confusion matrix.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

AREAS = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [16 ** 2, 32 ** 2],
         "coco_small": [0, 32 ** 2], "medium+": [32 ** 2, 1e10]}
MAX_DETS = [1, 100, 500]


def coco_gt(split) -> dict:
    """COCO-format GT dict. Image id = position in split.images; category id = class_id + 1."""
    images, anns, aid = [], [], 1
    nc = len(split.class_names)
    for i, r in enumerate(split.images):
        images.append(dict(id=i, file_name=r.path.name, width=r.width, height=r.height))
        for x1, y1, x2, y2 in r.ignore:
            w, h = float(x2 - x1), float(y2 - y1)
            for c in range(1, nc + 1):
                anns.append(dict(id=aid, image_id=i, category_id=c, bbox=[float(x1), float(y1), w, h],
                                 area=w * h, iscrowd=1)); aid += 1
        for c, x1, y1, x2, y2, *_ in r.boxes:
            w, h = float(x2 - x1), float(y2 - y1)
            anns.append(dict(id=aid, image_id=i, category_id=int(c) + 1, bbox=[float(x1), float(y1), w, h],
                             area=w * h, iscrowd=0)); aid += 1
    return dict(images=images, annotations=anns,
                categories=[dict(id=k + 1, name=n) for k, n in enumerate(split.class_names)])


def _coco(split):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(); G.dataset = coco_gt(split); G.createIndex()
    return G


def _ap(p):
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def run_cocoeval(split, dets, gt=None):
    """dets: {image_key: (N,6) x1,y1,x2,y2,score,cls}. Returns the evaluated+accumulated COCOeval, or None."""
    from pycocotools.cocoeval import COCOeval
    G = gt or _coco(split)
    rows = []
    for i, r in enumerate(split.images):
        d = dets.get(r.key)
        if d is not None and len(d):
            rows.append(np.c_[np.full(len(d), i), d[:, 0], d[:, 1], d[:, 2] - d[:, 0], d[:, 3] - d[:, 1],
                              d[:, 4], d[:, 5] + 1])
    if not rows:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = G.loadRes(np.concatenate(rows).astype(np.float64))
        E = COCOeval(G, dt, "bbox")
        E.params.imgIds = list(range(len(split.images)))
        E.params.maxDets = list(MAX_DETS)
        E.params.areaRng = list(AREAS.values()); E.params.areaRngLbl = list(AREAS.keys())
        E.evaluate(); E.accumulate()
    return E


def metrics_from_eval(E, class_names):
    """-> (metrics dict, per-class AP50 dict). Same keys as evaluation v2."""
    if E is None:
        return {"AP": 0.0, "AP50": 0.0}, {c: 0.0 for c in class_names}
    P = E.eval["precision"]                       # [IoU, recall, class, area, maxDets]
    m = {"AP": _ap(P[:, :, :, 0, -1]), "AP50": _ap(P[0, :, :, 0, -1]), "AP75": _ap(P[5, :, :, 0, -1])}
    for a, lbl in enumerate(AREAS):
        if lbl != "all":
            m[f"AP_{lbl}"] = _ap(P[:, :, :, a, -1]); m[f"AP50_{lbl}"] = _ap(P[0, :, :, a, -1])
    per_cls = {c: _ap(P[0, :, k, 0, -1]) for k, c in enumerate(class_names)}
    return m, per_cls


def evaluate(split, dets, gt=None):
    """-> (metrics, per-class AP50, COCOeval)."""
    E = run_cocoeval(split, dets, gt)
    m, pc = metrics_from_eval(E, split.class_names)
    return m, pc, E


def per_class_ap(E, class_names):
    """Per-class AP (IoU 0.50:0.95, area all, maxDets 500). NaN for a class with no GT."""
    if E is None:
        return {c: 0.0 for c in class_names}
    P = E.eval["precision"]
    return {c: _ap(P[:, :, k, 0, -1]) for k, c in enumerate(class_names)}


def ignore_share(split, dets, conf=0.25):
    """Share of detections with score >= conf lying >= 50% inside an ignore region (same 50% rule as
    confusion_matrix). -> (share, number of detections with score >= conf)."""
    inside = total = 0
    for r in split.images:
        d = dets.get(r.key)
        if d is None or not len(d):
            continue
        d = d[d[:, 4] >= conf]
        total += len(d)
        if len(d) and len(r.ignore):
            inter, _, area = _iou(d[:, :4], r.ignore)
            inside += int(((inter / np.clip(area[:, None], 1e-9, None)).max(1) >= 0.5).sum())
    return inside / total if total else 0.0, total


def matches_at_iou50(E):
    """Per-detection outcome at IoU 0.5, area 'all', maxDets 500, using COCOeval's own matching.

    Returns (scores, is_tp, n_gt) per class: scores/is_tp are arrays of non-ignored detections
    (detections matched to a crowd/ignore box are dropped: neither TP nor FP).
    """
    p = E.params
    nA, nI = len(p.areaRng), len(p.imgIds)
    out = {}
    for k, cat in enumerate(p.catIds):
        sc, tp, ngt = [], [], 0
        for i in range(nI):
            e = E.evalImgs[k * nA * nI + 0 * nI + i]           # area index 0 = 'all'
            if e is None:
                continue
            ngt += int((~np.asarray(e["gtIgnore"], bool)).sum())
            dm, di = e["dtMatches"][0], e["dtIgnore"][0]
            keep = ~np.asarray(di, bool)
            sc.append(np.asarray(e["dtScores"])[keep]); tp.append((np.asarray(dm) > 0)[keep])
        out[cat] = (np.concatenate(sc) if sc else np.zeros(0), np.concatenate(tp) if tp else np.zeros(0, bool), ngt)
    return out


def pr_at_conf(E, conf):
    """Pooled (all classes) precision/recall/F1 at IoU 0.5 for detections with score >= conf."""
    if E is None:
        return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, n_gt=0)
    tp = fp = ngt = 0
    for sc, t, n in matches_at_iou50(E).values():
        s = sc >= conf
        tp += int(t[s].sum()); fp += int((~t[s]).sum()); ngt += n
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / ngt if ngt else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, tp=tp, fp=fp, n_gt=ngt)


def best_f1_conf(E, grid=None):
    """Confidence with the highest pooled F1 (pick on val, reuse unchanged on test)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    M = matches_at_iou50(E)
    sc = np.concatenate([v[0] for v in M.values()]); t = np.concatenate([v[1] for v in M.values()])
    ngt = sum(v[2] for v in M.values())
    best = (0.0, float(grid[0]))
    for c in grid:
        s = sc >= c; tp = t[s].sum(); fp = (~t[s]).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / ngt if ngt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best[0]:
            best = (f1, float(c))
    return best[1]


def pr_curves(E):
    """-> (recall thresholds (101,), precision (101, n_classes)) at IoU 0.5, area all, maxDets 500.
    COCO's interpolated precision; -1 (no GT) becomes NaN."""
    P = E.eval["precision"][0, :, :, 0, -1].astype(float)
    P[P < 0] = np.nan
    return np.asarray(E.params.recThrs), P


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter, inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None), aa


def confusion_matrix(split, dets, conf, iou_thr=0.5):
    """(nc+1, nc+1) counts; rows = true class (+ last row 'background' = false positives),
    columns = predicted class (+ last column 'background' = missed objects).

    Class-agnostic greedy matching by score at IoU >= iou_thr. Unmatched detections whose area
    is >= 50% inside an ignore region are dropped, mirroring the crowd rule (COCO uses
    intersection / detection area for crowd boxes).
    """
    nc = len(split.class_names)
    cm = np.zeros((nc + 1, nc + 1), np.int64)
    for r in split.images:
        d = dets.get(r.key, np.zeros((0, 6), np.float32))
        d = d[d[:, 4] >= conf]
        d = d[np.argsort(-d[:, 4], kind="stable")]
        g = r.boxes
        used = np.zeros(len(g), bool)
        unmatched = []
        if len(d) and len(g):
            _, iou, _ = _iou(d[:, :4], g[:, 1:5])
        for j in range(len(d)):
            if len(g):
                cand = np.where(~used & (iou[j] >= iou_thr))[0]
                if len(cand):
                    gi = cand[np.argmax(iou[j, cand])]
                    used[gi] = True
                    cm[int(g[gi, 0]), int(d[j, 5])] += 1
                    continue
            unmatched.append(j)
        if unmatched:
            u = d[unmatched]
            if len(r.ignore):
                inter, _, area = _iou(u[:, :4], r.ignore)
                u = u[(inter / np.clip(area[:, None], 1e-9, None)).max(1) < 0.5]
            for c in u[:, 5].astype(int):
                cm[nc, c] += 1
        for gi in np.where(~used)[0]:
            cm[int(g[gi, 0]), nc] += 1
    return cm


def normalize_rows(cm):
    s = cm.sum(1, keepdims=True).astype(float)
    return np.divide(cm, s, out=np.zeros_like(cm, float), where=s > 0)
