"""Write reports/capstone.json (repository root) for the app's "How it works" page, from the capstone's result files only.

Run:  python tools/build_app_metrics.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
r3 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)  # noqa: E731

b = pd.concat([pd.read_csv(E / "phase0/results/basic_metrics.csv"), pd.read_csv(E / "phase0/results/basic_metrics_yolo26s.csv")])
det = []
for m in ["Ensemble", "YOLO26m-Tile", "YOLOE-26m-FT", "YOLO26s", "YOLOE-26m-LP", "YOLOE-26m-ZS"]:
    v = b[(b.model == m) & (b.setup == "best_sliced")].set_index("split")
    t = v.loc["test-dev"]
    det.append(dict(model=m, val_ap50=r3(v.loc["val", "mAP50"]), ap50=r3(t.mAP50), ap=r3(t["mAP50-95"]), tiny=r3(t.AP50_tiny), small=r3(t.AP50_small),
                    medium=r3(t["AP50_medium+"]), precision=r3(t.precision), recall=r3(t.recall), ms=round(float(t.ms_per_img))))
pc = pd.read_csv(E / "phase0/results/per_class_val_vs_testdev.csv")
per_class = [dict(cls=r["class"], objects=int(r["n_test-dev"]), ap50=r3(r["YOLO26m-Tile test-dev"]), val=r3(r["YOLO26m-Tile val"])) for _, r in pc.iterrows()]

M = pd.read_csv(E / "phase1/eval/results/transfer_matrix.csv")
M = M[M.van_rule == "van=car"]
transfer = []
for m in ["YOLO26m-Tile (VisDrone)", "YOLO26s (VisDrone)", "YOLO26m-Tile (UAVDT)", "YOLO26s (UAVDT)"]:
    row = {"model": m}
    for c, k in (("visdrone/test-dev", "visdrone"), ("uavdt/test", "uavdt"), ("carpk/test", "carpk")):
        v = M[(M.model == m) & (M.target == c)]
        row[k] = r3(v.AP50.iloc[0]) if len(v) else None
    transfer.append(row)
bu = pd.read_csv(E / "phase1/eval/results/basic_metrics_uavdt.csv")
uavdt = [dict(model=r.model, split=r.split, precision=r3(r.precision), recall=r3(r.recall), ap50=r3(r.mAP50), ap=r3(bu.loc[i, "mAP50-95"])) for i, r in bu.iterrows()]

count = []
for k, lab in (("in-domain_VisDrone", "VisDrone test-dev, VisDrone model"), ("in-domain_UAVDT", "UAVDT test, UAVDT model"),
               ("shift_to_UAVDT", "UAVDT test, VisDrone model"), ("shift_to_CARPK", "CARPK, VisDrone model")):
    d = pd.read_csv(E / f"phase4/results/paired__{k}.csv.gz")
    d = d[d.target == "car"]
    e = d.yhat - d.y
    count.append(dict(label=lab, images=len(d), mean=round(float(d.y.mean()), 1), mae=round(float(e.abs().mean()), 2), bias=round(float(e.mean()), 2)))

C = pd.read_csv(E / "phase4/results/coverage.csv")
S = pd.read_csv(E / "phase4/results/coverage_by_gt_stratum.csv")


def cov(chk, m, tc=0.9, mond="none", st="all"):
    v = C[(C.check == chk) & (C.method == m) & (C.score == "sqrt") & (C.mondrian_by == mond) & (C.stratum == st) & np.isclose(C.target_coverage, tc)]
    if v.empty:
        return None
    v = v.iloc[0]
    return dict(coverage=None if v.share_infinite == 1 else r3(v.coverage), width=None if v.share_infinite == 1 else round(float(v.mean_width), 1),
                videos=r3(v.share_sequences_at_target), infinite=bool(v.share_infinite == 1))


coverage = [dict(label=lab, sequence=cov(chk, "sequence"), image=cov(chk, "image")) for chk, lab in
            (("in-domain VisDrone", "VisDrone (home data)"), ("in-domain UAVDT", "UAVDT, its own model"), ("shift to UAVDT", "VisDrone model on UAVDT"), ("shift to CARPK", "VisDrone model on CARPK"))]
classes = [dict(cls=c, sequence=cov("in-domain VisDrone", "sequence", .9, "target", c), image=cov("in-domain VisDrone", "image", .9, "target", c))
           for c in ("car", "pedestrian", "people", "motor", "van", "truck", "bus", "bicycle")]
d = S[(S.check == "in-domain VisDrone") & (S.method == "sequence") & np.isclose(S.target_coverage, .9)].copy()
d["mondrian"] = d["mondrian"].fillna("none")
strata = []
for by, st, lab in (("count_bin", "1-8", "1 to 8 objects"), ("count_bin", "9-20", "9 to 20"), ("count_bin", "21-50", "21 to 50"), ("count_bin", ">50", "more than 50"),
                    ("gt_size_bin", "small", "small objects"), ("gt_size_bin", "large", "large objects")):
    g = d[(d.mondrian_by == by) & (d.stratum == st)].set_index("mondrian")
    strata.append(dict(label=lab, plain=r3(g.loc["none", "coverage"]), adjusted=r3(g.loc["pred_density_bin" if by == "count_bin" else "pred_size_bin", "coverage"])))
pv = C[(C.check == "in-domain VisDrone") & (C.method == "poisson") & (C.stratum == "car") & np.isclose(C.target_coverage, .9)].iloc[0]
rv = C[(C.check == "in-domain VisDrone") & (C.method == "residual 10-90") & (C.stratum == "car")].iloc[0]
FS = pd.read_csv(E / "phase4/results/few_shot_summary.csv")
few = [dict(k=int(r.k), uavdt=r3(r.coverage)) for r in FS[(FS.check == "shift to UAVDT") & (FS.method == "R1 target only") & np.isclose(FS.alpha, .1)].itertuples() if r.share_infinite < 1]

out = dict(version="capstone-2026-11-10", source="experiments/phase0, phase1, phase4 result files", detection=det, per_class=per_class, transfer=transfer, uavdt=uavdt,
           counting=count, coverage=coverage, classes=classes, strata=strata,
           baselines=dict(poisson=r3(pv.coverage), poisson_width=round(float(pv.mean_width), 1), residual=r3(rv.coverage),
                          conformal_car=cov("in-domain VisDrone", "sequence", .9, "target", "car")), few_shot=few)
(ROOT.parent / "reports/capstone.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("written", len(det), "detection rows;", out["baselines"])
