"""Threshold and count-estimator calibration (detector weights stay frozen).

Everything works on cached predictions (see cache_preds.py) and VisDrone GT.
"""

from __future__ import annotations

import numpy as np

from ..classes import NUM_CLASSES
from ..eval.counting import count_metrics, counts_at

GRID = np.round(np.arange(0.05, 0.7001, 0.025), 3)


# --------------------------------------------------------------------------- folds
def group_folds(groups: list[str], k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Deterministic sequence-grouped folds -> list of test-index arrays."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    # greedy balance by number of frames
    sizes = {g: groups.count(g) for g in uniq}
    bins: list[list[str]] = [[] for _ in range(k)]
    load = [0] * k
    for g in sorted(uniq, key=lambda g: -sizes[g]):
        j = int(np.argmin(load))
        bins[j].append(g)
        load[j] += sizes[g]
    garr = np.array(groups)
    return [np.where(np.isin(garr, b))[0] for b in bins]


# --------------------------------------------------------------------------- counts on grid
def count_grid(P: list[np.ndarray], grid: np.ndarray = GRID) -> np.ndarray:
    """[I, K, T] counts of class-k detections with conf >= grid[t]."""
    out = np.zeros((len(P), NUM_CLASSES, len(grid)), np.int64)
    for i, p in enumerate(P):
        for k in range(NUM_CLASSES):
            s = np.sort(p[p[:, 5] == k, 4]) if len(p) else np.zeros(0)
            out[i, k] = len(s) - np.searchsorted(s, grid, side="left")
    return out


def tune_count_thresholds(CG: np.ndarray, G: np.ndarray, grid=GRID) -> np.ndarray:
    """Per-class threshold minimising MAE (cells are independent per class)."""
    mae = np.abs(CG - G[:, :, None]).mean(0)  # [K, T]
    return grid[mae.argmin(1)]


def tune_global_threshold(CG, G, grid=GRID) -> float:
    mae = np.abs(CG - G[:, :, None]).mean((0, 1))
    return float(grid[mae.argmin()])


def tune_existence_thresholds(CG, G, count_thr, min_prec=0.80, grid=GRID) -> np.ndarray:
    out = np.zeros(NUM_CLASSES)
    gt = G > 0
    for k in range(NUM_CLASSES):
        pred = CG[:, k, :] > 0  # [I, T]
        tp = (pred & gt[:, [k]]).sum(0)
        prec = tp / np.maximum(pred.sum(0), 1)
        rec = tp / max(gt[:, k].sum(), 1)
        ok = np.where((prec >= min_prec) & (grid <= count_thr[k] + 1e-9))[0]
        if len(ok):
            out[k] = grid[ok.min()]
        else:
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            f1[grid > count_thr[k] + 1e-9] = -1
            out[k] = grid[f1.argmax()]
    return out


def existence_at(P, thr) -> np.ndarray:
    return counts_at(P, thr) > 0


# --------------------------------------------------------------------------- matching
def _iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    bb = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1e-9)


def tp_labels(p: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray, iou_thr=0.5) -> np.ndarray:
    """Greedy (by conf) same-class matching; returns 1 for TP, 0 for FP."""
    lab = np.zeros(len(p), np.int8)
    for k in np.unique(p[:, 5]).astype(int) if len(p) else []:
        di = np.where(p[:, 5] == k)[0]
        di = di[np.argsort(-p[di, 4])]
        g = gt_boxes[gt_cls == k]
        if len(g) == 0:
            continue
        ious = _iou(p[di, :4], g)
        used = np.zeros(len(g), bool)
        for r, d in enumerate(di):
            row = np.where(used, -1, ious[r])
            j = row.argmax()
            if row[j] >= iou_thr:
                used[j] = True
                lab[d] = 1
    return lab


# --------------------------------------------------------------------------- estimators
class ExpectedCount:
    """Candidate B: count = round(s_c * sum_i iso_c(conf_i))."""

    def __init__(self, floor: float = 0.05):
        self.floor = floor
        self.iso = [None] * NUM_CLASSES
        self.scale = np.ones(NUM_CLASSES)

    def fit(self, P, labels, G):
        from sklearn.isotonic import IsotonicRegression

        allp = np.concatenate([p for p in P]) if P else np.zeros((0, 6))
        alll = np.concatenate(labels) if labels else np.zeros(0)
        m = allp[:, 4] >= self.floor
        allp, alll = allp[m], alll[m]
        for k in range(NUM_CLASSES):
            mk = allp[:, 5] == k
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip", increasing=True)
            if mk.sum() >= 10:
                ir.fit(allp[mk, 4], alll[mk])
            else:
                ir.fit([0, 1], [0, 1])
            self.iso[k] = ir
        E = self.expected(P)
        self.scale = G.sum(0) / np.maximum(E.sum(0), 1e-9)
        return self

    def expected(self, P) -> np.ndarray:
        out = np.zeros((len(P), NUM_CLASSES))
        for i, p in enumerate(P):
            if not len(p):
                continue
            p = p[p[:, 4] >= self.floor]
            for k in range(NUM_CLASSES):
                c = p[p[:, 5] == k, 4]
                if len(c):
                    out[i, k] = self.iso[k].predict(c).sum()
        return out

    def predict(self, P) -> np.ndarray:
        return np.rint(self.expected(P) * self.scale).astype(np.int64)

    def to_json(self) -> dict:
        return {
            "floor": self.floor,
            "scale": self.scale.round(4).tolist(),
            "isotonic": [
                {"x": ir.X_thresholds_.round(4).tolist(), "y": ir.y_thresholds_.round(4).tolist()} for ir in self.iso
            ],
        }


def scaled_threshold_counts(C_train, G_train, C_test) -> tuple[np.ndarray, np.ndarray]:
    """Candidate C: s_c * n_c, s_c fitted on train."""
    s = G_train.sum(0) / np.maximum(C_train.sum(0), 1)
    return np.rint(C_test * s).astype(np.int64), s


# --------------------------------------------------------------------------- bootstrap
def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, groups: list[str], n=2000, seed=0, unit="sequence"):
    """Mean of (err_b - err_a) per cell with a 95% CI; resample images or whole sequences.

    err_* are [I, K] absolute errors. Returns (mean, lo, hi).
    """
    rng = np.random.default_rng(seed)
    d = (err_b - err_a).sum(1)  # per-image summed difference
    n_cells = err_a.shape[1]
    if unit == "image":
        units = [np.array([i]) for i in range(len(d))]
    else:
        garr = np.array(groups)
        units = [np.where(garr == g)[0] for g in sorted(set(groups))]
    sums = np.array([d[u].sum() for u in units])
    cnts = np.array([len(u) for u in units])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(units), len(units))
        stats.append(sums[idx].sum() / (cnts[idx].sum() * n_cells))
    stats = np.array(stats)
    return float(d.sum() / (len(d) * n_cells)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


__all__ = [
    "GRID", "group_folds", "count_grid", "tune_count_thresholds", "tune_global_threshold",
    "tune_existence_thresholds", "existence_at", "tp_labels", "ExpectedCount",
    "scaled_threshold_counts", "paired_bootstrap", "count_metrics",
]


# --------------------------------------------------------------------------- round 2: density-aware
DENSITY_EDGES = (40, 100)


def density_bucket(P, ref_conf: float = 0.25) -> np.ndarray:
    n = np.array([int((p[:, 4] >= ref_conf).sum()) if len(p) else 0 for p in P])
    return np.digitize(n, DENSITY_EDGES)  # 0, 1, 2


def tune_density_thresholds(CG, G, buckets, fallback, min_images=30, grid=GRID) -> np.ndarray:
    """[B, K] thresholds."""
    out = np.tile(fallback, (len(DENSITY_EDGES) + 1, 1)).astype(float)
    for b in range(len(DENSITY_EDGES) + 1):
        m = buckets == b
        if m.sum() >= min_images:
            out[b] = tune_count_thresholds(CG[m], G[m], grid)
    return out


def counts_density(P, buckets, thr_bk) -> np.ndarray:
    out = np.zeros((len(P), NUM_CLASSES), np.int64)
    for i, (p, b) in enumerate(zip(P, buckets)):
        out[i] = counts_at([p], thr_bk[b])[0]
    return out
