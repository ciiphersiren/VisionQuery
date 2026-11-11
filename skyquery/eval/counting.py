"""Counting / existence evaluation on image-by-class cells.

Protocols
  * ignore-aware: detections whose centre lies in an ignore region are dropped
    before counting (default; matches the report)
  * raw: all detections count, which inflates over-counting

Metrics (per the report)
  MAE            mean |pred - gt| over all image x class cells
  MAE_nz         same, only cells with gt > 0
  exact_nz       fraction of gt > 0 cells counted exactly
  rel_err        mean |pred - gt| / gt over gt > 0 cells
  bias           sum(pred) / sum(gt)
  exist_*        accuracy / precision / recall of (pred > 0) vs (gt > 0)
"""

from __future__ import annotations

import numpy as np

from ..classes import NUM_CLASSES
from .visdrone import ImageGT, in_ignore


def prepare(preds: dict[str, np.ndarray], gts: dict[str, ImageGT], protocol: str = "ignore-aware"):
    """Return (stems, gt_counts[I,K], per-image filtered preds list)."""
    stems = sorted(gts)
    G = np.stack([gts[s].counts() for s in stems])
    P = []
    for s in stems:
        p = preds.get(s, np.zeros((0, 6), np.float32))
        if protocol == "ignore-aware" and len(p):
            p = p[~in_ignore(p[:, :4], gts[s].ignore)]
        P.append(p)
    return stems, G, P


def counts_at(P: list[np.ndarray], thr: np.ndarray) -> np.ndarray:
    """Threshold counts: [I, K]."""
    out = np.zeros((len(P), NUM_CLASSES), np.int64)
    for i, p in enumerate(P):
        if len(p):
            c = p[:, 5].astype(int)
            keep = p[:, 4] >= thr[c]
            out[i] = np.bincount(c[keep], minlength=NUM_CLASSES)
    return out


def count_metrics(C: np.ndarray, G: np.ndarray, E: np.ndarray | None = None) -> dict[str, float]:
    """C = predicted counts, G = gt counts, E = optional separate existence predictions (bool)."""
    C = np.asarray(C, float)
    G = np.asarray(G, float)
    nz = G > 0
    err = np.abs(C - G)
    ex_pred = (C > 0) if E is None else E.astype(bool)
    ex_gt = nz
    tp = (ex_pred & ex_gt).sum()
    return {
        "MAE": float(err.mean()),
        "MAE_nz": float(err[nz].mean()),
        "exact_nz": float((err[nz] == 0).mean()),
        "rel_err": float((err[nz] / G[nz]).mean()),
        "bias": float(C.sum() / G.sum()),
        "exist_acc": float((ex_pred == ex_gt).mean()),
        "exist_prec": float(tp / max(ex_pred.sum(), 1)),
        "exist_rec": float(tp / max(ex_gt.sum(), 1)),
    }


def per_class_table(C: np.ndarray, G: np.ndarray, E: np.ndarray | None = None) -> list[dict]:
    from ..classes import CLASSES

    rows = []
    for k, name in enumerate(CLASSES):
        m = count_metrics(C[:, [k]], G[:, [k]], None if E is None else E[:, [k]])
        rows.append({"class": name, "gt_boxes": int(G[:, k].sum()), "pred_boxes": int(round(C[:, k].sum())), **m})
    return rows
