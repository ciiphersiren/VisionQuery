"""Count set: one record per (image, class or group, region), with the exact ground-truth count and stratification variables.

Rules (Phase 2 plan): an object counts toward a region if its box center lies inside; objects centered in an ignore region are excluded;
records where ignore regions cover more than 20% of the region are dropped. Size and density tertiles are computed on the dev split and
frozen. Records hold an image's boxes through counts only, so the same functions count detections (`count_dets`).
"""
from __future__ import annotations

import numpy as np

REGIONS = {"whole": (0, 0, 1, 1), "left": (0, 0, .5, 1), "right": (.5, 0, 1, 1),
           "q1": (0, 0, .5, .5), "q2": (.5, 0, 1, .5), "q3": (0, .5, .5, 1), "q4": (.5, .5, 1, 1)}
GROUPS = {"people": ["pedestrian", "people"],
          "vehicles": ["car", "van", "truck", "bus", "tricycle", "awning-tricycle", "motor"]}
MAX_IGNORE = 0.20
COUNT_BINS = [(0, 0, "0"), (1, 8, "1-8"), (9, 20, "9-20"), (21, 50, "21-50"), (51, 10 ** 9, ">50")]
SEED = 0


def count_bin(n: int) -> str:
    return next(lbl for a, b, lbl in COUNT_BINS if a <= n <= b)


def targets(class_names):
    """{target name: class ids}. Every class, plus a group when at least two of its classes exist in this dataset."""
    out = {c: [i] for i, c in enumerate(class_names)}
    for g, members in GROUPS.items():
        ids = [class_names.index(m) for m in members if m in class_names]
        if len(ids) >= 2:
            out[g] = ids
    return out


def region_box(name, W, H):
    a, b, c, d = REGIONS[name]
    return a * W, b * H, c * W, d * H


def centers(b):
    return (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2


def in_region(b, box):
    """Half-open, so that a center on a shared border belongs to exactly one region."""
    cx, cy = centers(b)
    return (cx >= box[0]) & (cx < box[2]) & (cy >= box[1]) & (cy < box[3])


def in_ignore(b, ignore):
    if not len(b) or not len(ignore):
        return np.zeros(len(b), bool)
    cx, cy = centers(b)
    return ((cx[:, None] >= ignore[None, :, 0]) & (cx[:, None] < ignore[None, :, 2]) &
            (cy[:, None] >= ignore[None, :, 1]) & (cy[:, None] < ignore[None, :, 3])).any(1)


def ignore_masks(rec, step=4):
    """{region: share of the region inside ignore regions (union)}, on a coarse grid."""
    W, H = rec.width, rec.height
    m = np.zeros((int(np.ceil(H / step)), int(np.ceil(W / step))), bool)
    for x1, y1, x2, y2 in rec.ignore:
        m[max(0, int(y1 // step)):max(0, int(np.ceil(y2 / step))), max(0, int(x1 // step)):max(0, int(np.ceil(x2 / step)))] = True
    out = {}
    for k in REGIONS:
        x1, y1, x2, y2 = region_box(k, m.shape[1], m.shape[0])
        out[k] = float(m[int(y1):int(y2), int(x1):int(x2)].mean())
    return out


def density(b):
    """Mean nearest-neighbor center distance between counted objects, in median box sides. NaN below 2 objects."""
    if len(b) < 2:
        return float("nan")
    cx, cy = centers(b)
    d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
    np.fill_diagonal(d2, np.inf)
    side = np.median(np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    return float(np.sqrt(d2.min(1)).mean() / max(side, 1e-6))


def records(split):
    """Ground-truth records for a Split, as a list of dicts."""
    tg = targets(split.class_names)
    rows = []
    for r in split.images:
        keep = ~in_ignore(r.boxes[:, 1:5], r.ignore)
        B = r.boxes[keep]
        ign = ignore_masks(r)
        for reg in REGIONS:
            if ign[reg] > MAX_IGNORE:
                continue
            box = region_box(reg, r.width, r.height)
            inside = B[in_region(B[:, 1:5], box)]
            for t, ids in tg.items():
                b = inside[np.isin(inside[:, 0], ids)]
                n = len(b)
                sides = np.sqrt((b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])) if n else np.zeros(0)
                rows.append(dict(dataset=split.dataset, split=split.name, image=r.key, sequence=r.sequence, target=t, region=reg,
                                 count=n, count_bin=count_bin(n), median_side=float(np.median(sides)) if n else float("nan"),
                                 median_area_frac=float(np.median(sides ** 2) / (r.width * r.height)) if n else float("nan"),
                                 density=density(b[:, 1:5]),
                                 frac_occluded=float((b[:, 5] > 0).mean()) if n and (b[:, 5] >= 0).all() else float("nan"),
                                 frac_truncated=float((b[:, 6] > 0).mean()) if n and (b[:, 6] >= 0).all() else float("nan"),
                                 altitude=r.attrs.get("altitude"), weather=r.attrs.get("weather")))
    return rows


def tertile_edges(values):
    """Two cut points from dev data; frozen and reused on every other split."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.quantile(v, [1 / 3, 2 / 3])]


def tertile(values, edges, labels=("low", "mid", "high")):
    v = np.asarray(values, float)
    out = np.full(len(v), "none", dtype=object)
    ok = np.isfinite(v)
    out[ok] = np.array(labels, dtype=object)[np.searchsorted(edges, v[ok], side="right")]
    return out


def count_dets(d, image_size, ignore, class_ids, region="whole", thresholds=None):
    """Predicted count for one image from detections (N,6), by the same rules as the ground truth.
    thresholds: {class id: minimum score}; classes not listed are not counted."""
    if not len(d):
        return 0
    box = region_box(region, *image_size)
    m = np.isin(d[:, 5].astype(int), class_ids)
    if thresholds is not None:
        m &= d[:, 4] >= np.array([thresholds.get(int(c), np.inf) for c in d[:, 5]])
    b = d[m][:, :4]
    return int((in_region(b, box) & ~in_ignore(b, ignore)).sum())


def stratified_subset(df, n=300, seed=SEED, by=("dataset", "count_bin", "size_bin")):
    """Fixed image subset balanced across dataset x count bin x size bin, using each image's whole-image record of its most frequent
    target. Cells are filled in turn until n images are held. Returns the sorted (dataset, image) list."""
    rng = np.random.default_rng(seed)
    w = df[(df["region"] == "whole") & (df["count"] > 0)]
    w = w.sort_values("count", ascending=False, kind="stable").drop_duplicates(["dataset", "image"])
    cells = {k: list(g["image"]) for k, g in w.groupby(list(by), sort=True)}
    for k in cells:
        rng.shuffle(cells[k])
    seen, out, active = set(), [], sorted(cells)
    while len(out) < n and active:
        for k in list(active):
            while cells[k] and (k[0], cells[k][-1]) in seen:
                cells[k].pop()
            if not cells[k]:
                active.remove(k)
                continue
            im = cells[k].pop()
            seen.add((k[0], im)); out.append((k[0], im))
            if len(out) == n:
                break
    return sorted(out)


THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _kept_scores(rec, d, cid):
    d = d[d[:, 5].astype(int) == cid]
    return d[~in_ignore(d[:, :4], rec.ignore), 4] if len(d) else np.zeros(0)


def tune_thresholds(split, dets, class_ids):
    """{class id: threshold with the lowest whole-image count MAE on this (dev) split}. GT and detections follow the record rules:
    centered in an ignore region = not counted. Ties go to the lower threshold, and images with heavily ignored regions stay out."""
    ign = [ignore_masks(r)["whole"] <= MAX_IGNORE for r in split.images]
    out = {}
    for c in class_ids:
        y, sc = [], []
        for r, ok in zip(split.images, ign):
            if ok:
                b = r.boxes[r.boxes[:, 0] == c]
                y.append(int((~in_ignore(b[:, 1:5], r.ignore)).sum()))
                sc.append(np.sort(_kept_scores(r, dets.get(r.key, np.zeros((0, 6))), c)))
        y = np.array(y)
        n = np.array([[len(s) - np.searchsorted(s, t, side="left") for t in THRESHOLD_GRID] for s in sc]) if sc else np.zeros((0, len(THRESHOLD_GRID)))
        out[int(c)] = float(THRESHOLD_GRID[int(np.argmin(np.abs(n - y[:, None]).mean(0)))]) if len(y) else 0.5
    return out


def _side_density(b):
    if not len(b):
        return float("nan"), float("nan")
    return float(np.median(np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))), density(b)


def paired_records(split, dets, thresholds, pred_spec, gt_spec, regions=("whole",)):
    """One row per (image, target, region): true count y, predicted count yhat, ground-truth strata variables and the same variables
    from the detections (usable at test time). pred_spec / gt_spec: {target: class ids of the model / of the split's labels}."""
    rows = []
    for r in split.images:
        ign = ignore_masks(r)
        B = r.boxes[~in_ignore(r.boxes[:, 1:5], r.ignore)]
        d = dets.get(r.key, np.zeros((0, 6), np.float32))
        keep = np.array([d_[4] >= thresholds.get(int(d_[5]), np.inf) for d_ in d], bool) if len(d) else np.zeros(0, bool)
        D = d[keep]
        D = D[~in_ignore(D[:, :4], r.ignore)] if len(D) else D
        for reg in regions:
            if ign[reg] > MAX_IGNORE:
                continue
            box = region_box(reg, r.width, r.height)
            Bi, Di = B[in_region(B[:, 1:5], box)], D[in_region(D[:, :4], box)]
            for t in pred_spec:
                g = Bi[np.isin(Bi[:, 0], gt_spec[t])]
                p = Di[np.isin(Di[:, 5].astype(int), pred_spec[t])]
                gs, gd = _side_density(g[:, 1:5])
                ps, pd_ = _side_density(p[:, :4])
                rows.append(dict(dataset=split.dataset, split=split.name, image=r.key, sequence=r.sequence, target=t, region=reg, y=len(g), yhat=len(p), count_bin=count_bin(len(g)),
                                 gt_side=gs, gt_density=gd, pred_side=ps, pred_density=pd_, altitude=r.attrs.get("altitude")))
    return rows
