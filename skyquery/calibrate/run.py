"""Calibration experiments (thresholds and count intervals tuned on validation data).

    python -m skyquery.calibrate.run repro  --split data/VisDrone2019-DET-val
    python -m skyquery.calibrate.run cv     --split data/VisDrone2019-DET-val --pipeline fi1280
    python -m skyquery.calibrate.run freeze --split data/VisDrone2019-DET-val --pipeline fi1280
    python -m skyquery.calibrate.run testdev --split data/VisDrone2019-DET-test-dev --pipeline fi1280
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..classes import CLASSES, NUM_CLASSES
from ..config import ROOT
from ..detector import fuse
from ..eval.coco_eval import coco_ap
from ..eval.counting import count_metrics, counts_at, per_class_table, prepare
from ..eval.visdrone import load_split, sequence_of
from .cache_preds import load_cache
from .tune import (
    GRID, ExpectedCount, count_grid, existence_at, group_folds, paired_bootstrap, scaled_threshold_counts,
    tp_labels, tune_count_thresholds, density_bucket, tune_density_thresholds, counts_density, tune_existence_thresholds, tune_global_threshold,
)

CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
CALIB = ROOT / "configs" / "calibration.json"
WEAK = ["people", "bicycle", "tricycle", "awning-tricycle"]
WEAK_IDX = [CLASSES.index(c) for c in WEAK]
FLOOR = 0.05  # runtime detector floor; calibration only uses detections above it


def split_tag(split: str) -> str:
    return "testdev" if "test" in Path(split).name else "val"


def load_preds(split: str, pipeline: str, fusion_iou: float = 0.6, floor: float = 0.0):
    tag = split_tag(split)
    scales = [1280] if pipeline == "fi1280" else [1280, 1920]
    caches = [load_cache(CACHE / f"{tag}_{s}.npz") for s in scales]
    shapes = caches[0][1]
    if len(caches) == 1:
        preds = caches[0][0]
    else:
        preds = {k: fuse([c[0][k] for c in caches], "nms", fusion_iou, 500) for k in caches[0][0]}
    if floor > 0:
        preds = {k: v[v[:, 4] >= floor] for k, v in preds.items()}
    return preds, shapes


def eval_cached(split, caches, pipeline="fi1280", fusion_iou=0.6):
    gts = load_split(split)
    parts = [load_cache(c) for c in caches]
    shapes = parts[0][1]
    preds = parts[0][0] if len(parts) == 1 else {k: fuse([p[0][k] for p in parts], "nms", fusion_iou, 500) for k in parts[0][0]}
    return coco_ap(gts, preds, shapes)


def _r(d, n=4):
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- 0. repro
def cmd_repro(a):
    gts = load_split(a.split)
    out = {}
    preds, shapes = load_preds(a.split, "fi1280")
    for sd in ("coco", "excl_tiny"):
        out[f"fi1280/small={sd}"] = _r(coco_ap(gts, preds, shapes, small_def=sd), 2)
    c2 = CACHE / f"{split_tag(a.split)}_1920.npz"
    if c2.exists() and set(load_cache(c2)[0]) >= set(preds):
        for iou in (0.5, 0.6, 0.7):
            p2, _ = load_preds(a.split, "fi1280+fi1920", iou)
            out[f"fi1280+fi1920/nms{iou}"] = _r(coco_ap(gts, p2, shapes), 2)
    print(json.dumps(out, indent=2))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"repro_{split_tag(a.split)}.json").write_text(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- 1+2. CV
def cmd_cv(a):
    gts = load_split(a.split)
    preds, _ = load_preds(a.split, a.pipeline, a.fusion_iou, FLOOR)
    stems, G, P = prepare(preds, gts, "ignore-aware")
    groups = [sequence_of(s) for s in stems]
    folds = group_folds(groups, 5)
    CG = count_grid(P)
    labels = [tp_labels(p, gts[s].boxes, gts[s].cls) for s, p in zip(stems, P)]

    I = len(stems)
    C_A = np.zeros((I, NUM_CLASSES), int)
    C_G = np.zeros((I, NUM_CLASSES), int)
    C_B = np.zeros((I, NUM_CLASSES), int)
    C_C = np.zeros((I, NUM_CLASSES), int)
    C_D = np.zeros((I, NUM_CLASSES), int)
    DB = density_bucket(P)
    E_sep = np.zeros((I, NUM_CLASSES), bool)
    fold_params = []
    for te in folds:
        tr = np.setdiff1d(np.arange(I), te)
        Ptr, Pte = [P[i] for i in tr], [P[i] for i in te]
        t_cnt = tune_count_thresholds(CG[tr], G[tr])
        t_glb = tune_global_threshold(CG[tr], G[tr])
        t_ex = tune_existence_thresholds(CG[tr], G[tr], t_cnt)
        C_A[te] = counts_at(Pte, t_cnt)
        C_G[te] = counts_at(Pte, np.full(NUM_CLASSES, t_glb))
        E_sep[te] = existence_at(Pte, t_ex)
        est = ExpectedCount(FLOOR).fit(Ptr, [labels[i] for i in tr], G[tr])
        C_B[te] = est.predict(Pte)
        C_C[te], _ = scaled_threshold_counts(counts_at(Ptr, t_cnt), G[tr], counts_at(Pte, t_cnt))
        t_den = tune_density_thresholds(CG[tr], G[tr], DB[tr], t_cnt)
        C_D[te] = counts_density(Pte, DB[te], t_den)
        fold_params.append({"count": t_cnt.tolist(), "global": t_glb, "exist": t_ex.tolist(), "density": t_den.tolist()})

    res = {
        "pipeline": a.pipeline, "images": I, "sequences": len(set(groups)),
        "A_per_class_thr": _r(count_metrics(C_A, G)),
        "global_thr": _r(count_metrics(C_G, G)),
        "B_expected_count": _r(count_metrics(C_B, G)),
        "C_scaled_thr": _r(count_metrics(C_C, G)),
        "D_density_thr": _r(count_metrics(C_D, G)),
        "A_with_separate_existence": _r(count_metrics(C_A, G, E_sep)),
    }
    # existence decision (weak classes)
    def weak_rec(E):
        gt = G > 0
        return float(np.mean([(E[:, k] & gt[:, k]).sum() / max(gt[:, k].sum(), 1) for k in WEAK_IDX]))

    res["existence_decision"] = {
        "weak_recall_count_thr": round(weak_rec(C_A > 0), 4),
        "weak_recall_separate": round(weak_rec(E_sep), 4),
        "pooled_prec_count_thr": res["A_per_class_thr"]["exist_prec"],
        "pooled_prec_separate": res["A_with_separate_existence"]["exist_prec"],
    }
    d = res["existence_decision"]
    d["adopt"] = bool(d["weak_recall_separate"] - d["weak_recall_count_thr"] >= 0.05 and d["pooled_prec_count_thr"] - d["pooled_prec_separate"] <= 0.05)
    d["per_class"] = {
        CLASSES[k]: {
            "rec_count_thr": round(float(((C_A[:, k] > 0) & (G[:, k] > 0)).sum() / max((G[:, k] > 0).sum(), 1)), 3),
            "rec_separate": round(float((E_sep[:, k] & (G[:, k] > 0)).sum() / max((G[:, k] > 0).sum(), 1)), 3),
            "prec_count_thr": round(float(((C_A[:, k] > 0) & (G[:, k] > 0)).sum() / max((C_A[:, k] > 0).sum(), 1)), 3),
            "prec_separate": round(float((E_sep[:, k] & (G[:, k] > 0)).sum() / max(E_sep[:, k].sum(), 1)), 3),
        }
        for k in range(NUM_CLASSES)
    }

    # count-estimator decision
    errA = np.abs(C_A - G)
    cands = {"B": C_B, "C": C_C}
    best = min(cands, key=lambda k: np.abs(cands[k] - G).mean())
    errBest = np.abs(cands[best] - G)
    mean, lo, hi = paired_bootstrap(errA, errBest, groups, unit="sequence")
    rel = (errBest.mean() - errA.mean()) / errA.mean()
    res["count_decision"] = {
        "best_candidate": best, "rel_change_vs_A": round(float(rel), 4),
        "bootstrap_seq_diff": [round(mean, 4), round(lo, 4), round(hi, 4)],
        "adopt": bool(rel <= -0.03 and hi < 0),
    }
    m, l, h = paired_bootstrap(np.abs(C_G - G), errA, groups, unit="sequence")
    errD = np.abs(C_D - G)
    mD, lD, hD = paired_bootstrap(errA, errD, groups, unit="sequence")
    relD = (errD.mean() - errA.mean()) / errA.mean()
    res["round2_density_decision"] = {"rel_change_vs_A": round(float(relD), 4), "bootstrap_seq_diff": [round(mD, 4), round(lD, 4), round(hD, 4)],
                                      "adopt": bool(relD <= -0.03 and hD < 0), "label": "exploratory",
                                      "bucket_sizes": np.bincount(DB, minlength=3).tolist()}
    res["per_class_vs_global_val_seq_bootstrap"] = [round(m, 4), round(l, 4), round(h, 4)]
    res["per_class_table_A"] = [_r(r, 3) for r in per_class_table(C_A, G, E_sep)]
    res["fold_params"] = fold_params
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"cv_{a.pipeline}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_class_table_A", "fold_params")}, indent=2))


# --------------------------------------------------------------------------- freeze
def _intervals(C: np.ndarray, G: np.ndarray) -> dict:
    """Per class, from out-of-fold counts: 10/90% quantiles of the residual (gt - pred).

    Bucket 0 stores absolute residuals; buckets with pred >= 1 store residuals
    relative to pred ("rel": true), which the runtime multiplies by n.
    Buckets with < 15 cells are merged into the next lower bucket's data.
    """
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10**6)]
    out = {}
    for k, c in enumerate(CLASSES):
        rows = []
        for lo, hi in buckets:
            m = (C[:, k] >= lo) & (C[:, k] <= hi)
            if m.sum() < 15 and lo > 0:
                m = C[:, k] >= max(1, lo // 2)
            if m.sum() < 15:
                continue
            r = G[m, k] - C[m, k]
            if lo == 0:
                q10, q90, rel = np.percentile(r, 10), np.percentile(r, 90), False
            else:
                rr = r / C[m, k]
                q10, q90, rel = np.percentile(rr, 10), np.percentile(rr, 90), True
            rows.append({"lo_pred": lo, "hi_pred": hi, "q10": round(float(q10), 3), "q90": round(float(q90), 3), "rel": rel, "n": int(m.sum())})
        out[c] = rows
    return out


def cmd_freeze(a):
    """Fit on full val and write configs/calibration.json for this pipeline."""
    cv = json.loads((REPORTS / f"cv_{a.pipeline}.json").read_text())
    gts = load_split(a.split)
    preds, _ = load_preds(a.split, a.pipeline, a.fusion_iou, FLOOR)
    stems, G, P = prepare(preds, gts, "ignore-aware")
    CG = count_grid(P)
    cal = json.loads(CALIB.read_text())
    entry = cal["pipelines"][a.pipeline]
    t_cnt = np.array([entry["count_thresholds"][c] for c in CLASSES])  # report's frozen values stay
    t_ex = tune_existence_thresholds(CG, G, t_cnt)
    entry["exist_thresholds"] = dict(zip(CLASSES, [float(x) for x in t_ex])) if cv["existence_decision"]["adopt"] else entry["count_thresholds"]
    entry["exist_threshold_status"] = "adopted (pre-registered rule passed)" if cv["existence_decision"]["adopt"] else "rejected; existence uses count thresholds"
    # out-of-fold counts with fold-tuned thresholds for honest intervals
    groups = [sequence_of(s) for s in stems]
    C_oof = np.zeros_like(G)
    for te, fp in zip(group_folds(groups, 5), cv["fold_params"]):
        C_oof[te] = counts_at([P[i] for i in te], np.array(fp["count"]))
    entry["intervals"] = _intervals(C_oof, G)
    entry["count_method"] = "threshold" if not cv["count_decision"]["adopt"] else f"candidate_{cv['count_decision']['best_candidate']}"
    E = existence_at(P, np.array([entry["exist_thresholds"][c] for c in CLASSES]))
    tab = per_class_table(C_oof, G, E)
    entry.setdefault("val_cv", {})
    entry["val_cv"]["exist_recall"] = {r["class"]: round(r["exist_rec"], 3) for r in tab}
    entry["exist_recall"] = entry["val_cv"]["exist_recall"]  # matches the thresholds actually deployed
    cal["version"] = "calibrated-2026-10-19"
    cal["notes"] = "Count thresholds: frozen. Existence thresholds + intervals: skyquery.calibrate.run (val, 5-fold sequence-grouped CV)."
    CALIB.write_text(json.dumps(cal, indent=2))
    print(json.dumps(entry["exist_thresholds"], indent=2), entry["exist_threshold_status"])


# --------------------------------------------------------------------------- test-dev, scored once
def cmd_testdev(a):
    gts = load_split(a.split)
    preds_full, shapes = load_preds(a.split, a.pipeline, a.fusion_iou)
    out = {"pipeline": a.pipeline, "AP": _r(coco_ap(gts, preds_full, shapes), 2)}
    preds = {k: v[v[:, 4] >= FLOOR] for k, v in preds_full.items()}
    stems, G, P = prepare(preds, gts, "ignore-aware")
    cal = json.loads(CALIB.read_text())["pipelines"][a.pipeline]
    t_cnt = np.array([cal["count_thresholds"][c] for c in CLASSES])
    t_ex = np.array([cal.get("exist_thresholds", cal["count_thresholds"])[c] for c in CLASSES])
    t_glb = np.full(NUM_CLASSES, cal["global_threshold"])
    C = counts_at(P, t_cnt)
    Cg = counts_at(P, t_glb)
    E = existence_at(P, t_ex)
    out["per_class_thr"] = _r(count_metrics(C, G))
    out["global_thr"] = _r(count_metrics(Cg, G))
    out["per_class_thr_with_exist_thr"] = _r(count_metrics(C, G, E))
    groups = [sequence_of(s) for s in stems]
    errA, errG = np.abs(C - G), np.abs(Cg - G)
    out["bootstrap_per_class_minus_global"] = {
        "image_unit": [round(x, 4) for x in paired_bootstrap(errG, errA, groups, unit="image")],
        "sequence_unit": [round(x, 4) for x in paired_bootstrap(errG, errA, groups, unit="sequence")],
        "n_images": len(stems), "n_sequences": len(set(groups)),
    }
    out["per_class"] = [_r(r, 3) for r in per_class_table(C, G, E)]
    (REPORTS / f"testdev_{a.pipeline}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repro", "cv", "freeze", "testdev"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--pipeline", default="fi1280")
    ap.add_argument("--fusion-iou", type=float, default=0.5)
    a = ap.parse_args(argv)
    {"repro": cmd_repro, "cv": cmd_cv, "freeze": cmd_freeze, "testdev": cmd_testdev}[a.cmd](a)


if __name__ == "__main__":
    main()
