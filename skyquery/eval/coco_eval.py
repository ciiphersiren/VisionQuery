"""COCO-style AP on VisDrone, following the report's protocol.

* ignore regions become iscrowd=1 boxes, replicated for every class, so a
  detection of any class landing in an ignore region is neither TP nor FP
* maxDets = 500
* an extra "tiny" bucket for objects under 16 px (area < 16^2)
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from ..classes import CLASSES, NUM_CLASSES
from .visdrone import ImageGT

AREA_RANGES = {
    "all": [0, 1e10],
    "tiny": [0, 16**2],
    "small": [0, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e10],
}


def _coco_objects(gts: dict[str, ImageGT], preds: dict[str, np.ndarray], shapes: dict[str, tuple[int, int]], names=None):
    """``names`` defaults to the 10 VisDrone classes; pass another list to score any vocabulary (cls indexes it)."""
    names = list(names or CLASSES)
    images, anns, dets = [], [], []
    aid = 1
    for iid, (stem, g) in enumerate(sorted(gts.items()), 1):
        w, h = shapes.get(stem, (0, 0))
        images.append({"id": iid, "file_name": stem, "width": w, "height": h})
        for (x1, y1, x2, y2), c in zip(g.boxes, g.cls):
            bw, bh = float(x2 - x1), float(y2 - y1)
            anns.append({"id": aid, "image_id": iid, "category_id": int(c) + 1, "bbox": [float(x1), float(y1), bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
        for x1, y1, x2, y2 in g.ignore:
            bw, bh = float(x2 - x1), float(y2 - y1)
            for c in range(len(names)):
                anns.append({"id": aid, "image_id": iid, "category_id": c + 1, "bbox": [float(x1), float(y1), bw, bh], "area": bw * bh, "iscrowd": 1})
                aid += 1
        p = preds.get(stem)
        if p is not None:
            for x1, y1, x2, y2, s, c in p:
                dets.append({"image_id": iid, "category_id": int(c) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    return {"images": images, "annotations": anns, "categories": cats}, dets


def coco_ap(gts, preds, shapes, max_det: int = 500, small_def: str = "coco", names=None) -> dict[str, float]:
    """Return AP, AP50, AP75, AP_tiny, AP_s, AP_m, AP_l (percent).

    small_def="coco" uses the standard [0, 32^2] small bucket; "excl_tiny"
    uses [16^2, 32^2] instead.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_json, dets = _coco_objects(gts, preds, shapes, names)
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt_json
        cg.createIndex()
        cd = cg.loadRes(dets) if dets else COCO()
        E = COCOeval(cg, cd, "bbox")
        ranges = dict(AREA_RANGES)
        if small_def == "excl_tiny":
            ranges["small"] = [16**2, 32**2]
        E.params.areaRng = list(ranges.values())
        E.params.areaRngLbl = list(ranges.keys())
        E.params.maxDets = [max_det]
        E.evaluate()
        E.accumulate()
    P = E.eval["precision"]  # [T, R, K, A, M]

    def ap(t=None, area="all"):
        a = E.params.areaRngLbl.index(area)
        p = P[:, :, :, a, 0] if t is None else P[[t], :, :, a, 0]
        p = p[p > -1]
        return float(p.mean() * 100) if p.size else float("nan")

    return {
        "AP": ap(), "AP50": ap(0), "AP75": ap(5),
        "AP_tiny": ap(area="tiny"), "AP_s": ap(area="small"),
        "AP_m": ap(area="medium"), "AP_l": ap(area="large"),
    }


def per_class_ap(gts, preds, shapes, max_det: int = 500, names=None) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = list(names or CLASSES)
    gt_json, dets = _coco_objects(gts, preds, shapes, names)
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt_json; cg.createIndex()
        E = COCOeval(cg, cg.loadRes(dets) if dets else COCO(), "bbox")
        E.params.maxDets = [max_det]
        E.evaluate(); E.accumulate()
    P = E.eval["precision"][:, :, :, 0, 0]
    out = {}
    for k, name in enumerate(names):
        p = P[:, :, k]
        p = p[p > -1]
        out[name] = float(p.mean() * 100) if p.size else float("nan")
    return out


# --------------------------------------------------------------------------- sequence bootstrap for AP
def eval_imgs(gts, preds, shapes, max_det: int = 500, names=None):
    """Run COCOeval.evaluate() once; returns (E, stems in image-id order)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_json, dets = _coco_objects(gts, preds, shapes, names)
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt_json; cg.createIndex()
        E = COCOeval(cg, cg.loadRes(dets) if dets else COCO(), "bbox")
        E.params.areaRng = [AREA_RANGES["all"]]
        E.params.areaRngLbl = ["all"]
        E.params.maxDets = [max_det]
        E.evaluate()
    return E, [im["file_name"] for im in gt_json["images"]]


def ap_from_eval(E, img_idx) -> float:
    """COCO AP (percent) from cached per-image matches over a (possibly repeated) list of image positions.

    Re-implements COCOeval.accumulate() for area 'all' so resampled image lists with duplicates work.
    """
    p = E.params
    K, I = len(p.catIds), len(p.imgIds)
    R = p.recThrs
    md = p.maxDets[0]
    aps = []
    for k in range(K):
        es = [E.evalImgs[k * I + i] for i in img_idx]  # one area range -> index k*A*I + 0*I + i
        es = [e for e in es if e is not None]
        if not es:
            continue
        gtIg = np.concatenate([e["gtIgnore"] for e in es])
        npig = int(np.count_nonzero(gtIg == 0))
        if npig == 0:
            continue
        scores = np.concatenate([e["dtScores"][:md] for e in es])
        order = np.argsort(-scores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :md] for e in es], axis=1)[:, order]
        dtig = np.concatenate([e["dtIgnore"][:, :md] for e in es], axis=1)[:, order]
        tps = np.cumsum(np.logical_and(dtm, ~dtig.astype(bool)), axis=1).astype(float)
        fps = np.cumsum(np.logical_and(~dtm.astype(bool), ~dtig.astype(bool)), axis=1).astype(float)
        prec_t = []
        for t in range(len(p.iouThrs)):
            tp, fp = tps[t], fps[t]
            if len(tp) == 0:
                prec_t.append(np.zeros(len(R)))
                continue
            rc = tp / npig
            pr = tp / (tp + fp + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, R, side="left")
            q = np.zeros(len(R))
            ok = idx < len(pr)
            q[ok] = pr[idx[ok]]
            prec_t.append(q)
        aps.append(np.mean(prec_t))
    return float(np.mean(aps) * 100) if aps else float("nan")


def bootstrap_ap(E, stems, n: int = 200, seed: int = 0, others=None):
    """95% CI of AP resampling whole sequences. With ``others`` (list of E on the same images),
    also returns CIs of AP(other) - AP(E) from the same resamples."""
    from .visdrone import sequence_of

    rng = np.random.default_rng(seed)
    seqs = {}
    for i, s in enumerate(stems):
        seqs.setdefault(sequence_of(s), []).append(i)
    units = list(seqs.values())
    base, diffs = [], [[] for _ in (others or [])]
    for _ in range(n):
        pick = rng.integers(0, len(units), len(units))
        idx = [i for u in pick for i in units[u]]
        a = ap_from_eval(E, idx)
        base.append(a)
        for j, o in enumerate(others or []):
            diffs[j].append(ap_from_eval(o, idx) - a)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
    return {"ap": round(ap_from_eval(E, list(range(len(stems)))), 2), "ci": ci(base), "diff_ci": [ci(d) for d in diffs]}
