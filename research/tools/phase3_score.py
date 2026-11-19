"""Phase 3: score the counting systems on the fixed VLM queries with one script (CPU). The decision rule was fixed in the capstone report first.

Run:  python tools/phase3_score.py --visdrone DIR --uavdt-gt DIR --uavdt-attr DIR --carpk DIR --merged .../merged --answers .../vlm_answers.csv --for-d .../detector_counts_for_d.csv --out phase3
 or:  python tools/phase3_score.py --paired experiments/phase4/results --answers run1/vlm_answers.csv rerun/vlm_answers.csv --for-d run1/detector_counts_for_d.csv
      (detector counts from the Phase 4 paired records, which hold the same counts at the same dev-tuned thresholds; no labels needed)
Several --answers files: for each query and system the last answer without a GPU error is used. A query still without an answer from every
system after that is left out and reported (an out-of-memory error is not a model answer).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cpu_common import ROOT, UV, VD, add_paths, load_merged, load_splits

from dronekit import countset as K, plots, vlm

N_BOOT = 2000
SEED = 0
ap = argparse.ArgumentParser()
for a_ in ("visdrone", "uavdt-gt", "uavdt-attr", "carpk", "merged"):
    ap.add_argument("--" + a_, type=Path)
ap.add_argument("--paired", type=Path, help="experiments/phase4/results: detector counts from the paired records instead of labels + caches")
ap.add_argument("--answers", type=Path, nargs="+", required=True)
ap.add_argument("--for-d", type=Path, required=True)
ap.add_argument("--thresholds", type=Path, default=ROOT / "experiments/phase4/results/thresholds.json")
ap.add_argument("--out", type=Path, default=Path("experiments/phase3"))
A = ap.parse_args()
(A.out / "results").mkdir(parents=True, exist_ok=True)
(A.out / "figures").mkdir(parents=True, exist_ok=True)
Q = pd.read_csv(ROOT / "countset/vlm_queries.csv")
ans = pd.concat([pd.read_csv(f).assign(run=i) for i, f in enumerate(A.answers)], ignore_index=True)
ans["ok"] = ans.error.isna()
ans = ans.sort_values(["qid", "system", "ok", "run"]).groupby(["qid", "system"]).tail(1)     # last error-free answer, else the last one
failed = sorted(ans[~ans.ok].qid.unique())
ans = ans[~ans.qid.isin(failed)]
print(f"{ans.qid.nunique()} queries with an answer from every system; left out after GPU errors: {len(failed)}")
pd.DataFrame(dict(qid=failed)).to_csv(A.out / "results/left_out_gpu_errors.csv", index=False)
if "max_pixels_used" in ans:
    pd.DataFrame([dict(fallback_size_answers=int((ans.max_pixels_used.notna() & (ans.max_pixels_used < ans.max_pixels_used.max())).sum()))]).to_csv(A.out / "results/fallback_sizes.csv", index=False)
ford = pd.read_csv(A.for_d)
PAIRED = {}
if A.paired:
    for name, key in (("in-domain_VisDrone", "visdrone"), ("in-domain_UAVDT", "uavdt"), ("shift_to_UAVDT", "uavdt_shift"), ("shift_to_CARPK", "carpk")):
        d = pd.read_csv(A.paired / f"paired__{name}.csv.gz")
        d = d[d.region == "whole"]
        PAIRED[key] = {(str(r.image), r.target): (int(r.yhat), int(r.y)) for r in d.itertuples()}
else:
    SP = load_splits(A)
    REC = {(k[0], r.key): r for k, sp in SP.items() for r in sp.images if k != ("visdrone", "val") and k != ("uavdt", "val")}

TH = json.loads(A.thresholds.read_text())                       # written by phase4_conformal.py from the dev splits


def detector(tag, classes, dataset, split_of, target, image, y=None):
    if PAIRED:
        # CARPK has no model of its own: its paired records are the VisDrone model counting car + van, as detector() does below
        key = {"visdrone": "visdrone", "carpk": "carpk"}.get(dataset) or ("uavdt" if tag == "tile_uv" else "uavdt_shift")
        yhat, y_rec = PAIRED[key][(str(image), target)]
        assert y is None or y == y_rec, f"true count differs for {dataset} {image} {target}: {y} vs {y_rec}"
        return yhat
    d = load_cached(tag, (dataset, split_of))
    r = REC[(dataset, image)]
    names = {"car": ["car", "van"]}.get(target, [target]) if tag == "tile_vd" and dataset != "visdrone" else [target]
    th = {classes.index(c): v for c, v in TH[tag].items()}
    return K.count_dets(d.get(image, np.zeros((0, 6))), (r.width, r.height), r.ignore, [classes.index(n) for n in names], "whole", th)


_cache = {}
def load_cached(tag, key):
    if (tag, key) not in _cache:
        _cache[(tag, key)] = load_merged(A.merged, tag, key) or {}
    return _cache[(tag, key)]


# Every system is scored only on queries with a real answer in the same run, so comparisons are paired
done = ans.qid.unique()
ford = ford[ford.qid.isin(done)]
base = ford.merge(Q.drop(columns=["count"]), on=["dataset", "image", "target"], how="left")
rows = []
for b in base.itertuples():
    common = dict(qid=b.qid, dataset=b.dataset, image=b.image, sequence=b.sequence, target=b.target, y=b.count, count_bin=b.count_bin, size_bin=b.size_bin, density_bin=b.density_bin, median_side=b.median_side, altitude=b.altitude)
    rows.append(dict(common, system="Detector at D's threshold", pred=b.detector_count, parsed=True))
    rows.append(dict(common, system="A: detector, dev-tuned", pred=detector("tile_uv" if b.dataset == "uavdt" else "tile_vd", UV if b.dataset == "uavdt" else VD, b.dataset, b.split, b.target, b.image, b.count), parsed=True))
    if b.dataset != "visdrone":
        rows.append(dict(common, system="A: VisDrone detector under shift", pred=detector("tile_vd", VD, b.dataset, b.split, b.target, b.image, b.count), parsed=True))
    for s, name in (("B", "B: VLM direct"), ("C", "C: VLM localize"), ("D", "D: VLM grounded")):
        a = ans[(ans.qid == b.qid) & (ans.system == s)]
        if len(a):
            p = a.parsed.iloc[0]
            rows.append(dict(common, system=name, pred=0 if pd.isna(p) else int(p), parsed=not pd.isna(p)))     # unparsable = wrong; enters MAE as a prediction of 0
            if s == "C":       # exploratory, added after seeing the answers: also accept the box lines Qwen2.5-VL wrote instead of points
                p = vlm.parse_points_lenient(a.raw.iloc[0] if isinstance(a.raw.iloc[0], str) else "")
                rows.append(dict(common, system="C*: VLM localize, lenient parse (exploratory)", pred=0 if p is None else p, parsed=p is not None))
R = pd.DataFrame(rows)
R["err"] = (R.pred - R.y).abs()
R["within10"] = R.parsed & (R.err <= np.where(R.y > 0, 0.1 * R.y, 0))
R["exact"] = R.parsed & (R.err == 0)
R.to_csv(A.out / "results/per_query.csv", index=False)


def boot(df, fn, n=N_BOOT, seed=SEED):
    """Percentile interval from resampling whole sequences; fn takes a DataFrame."""
    rng = np.random.default_rng(seed)
    g = dict(tuple(df.groupby("sequence")))
    keys = list(g)
    out = []
    for _ in range(n):
        s = pd.concat([g[keys[i]] for i in rng.integers(0, len(keys), len(keys))], ignore_index=True)
        out.append(fn(s))
    return np.nanquantile(out, [.025, .975])


def metrics(d):
    nz = d[d.y > 0]
    a, b = boot(d, lambda s: s.err.mean(), 500) if d.sequence.nunique() > 1 else (np.nan, np.nan)
    return dict(n=len(d), n_sequences=d.sequence.nunique(), MAE=d.err.mean(), MAE_lo=a, MAE_hi=b, MAE_norm=(nz.err / nz.y).mean() if len(nz) else np.nan, exact=d.exact.mean(), within10=d.within10.mean(), parsed=d.parsed.mean())


rows = []
for by in (None, "dataset", "count_bin", "size_bin", "density_bin", "altitude"):
    for (sysn, *v), d in R.groupby(["system"] + ([by] if by else []), sort=False):
        rows.append(dict(system=sysn, by=by or "overall", value=v[0] if v else "all", **metrics(d)))
M = pd.DataFrame(rows); M.to_csv(A.out / "results/metrics.csv", index=False)
print(M[M.by.isin(["overall", "dataset"])].round(3).to_string())


# ---- decision rule -------------------------------------------------------------------------------------------------------
def slope(d, x):
    d = d[d[x] > 0]
    return np.polyfit(np.log(d[x]), d.err, 1)[0] if len(d) > 2 and d[x].nunique() > 1 else np.nan


DET = "A: detector, dev-tuned"
rows = []
for s in ("B: VLM direct", "C: VLM localize", "D: VLM grounded", "C*: VLM localize, lenient parse (exploratory)"):
    a = R[R.system == s].set_index("qid"); b = R[R.system == DET].set_index("qid")
    j = a.join(b[["err"]], rsuffix="_det").reset_index()
    if not len(j):
        continue
    j["gap"] = j.err - j.err_det
    gi = boot(j, lambda t: t.gap.mean())
    for x, lab in (("y", "log true count"), ("median_side", "log median box side")):
        est = slope(j.assign(err=j.gap), x)
        lo, hi = boot(j, lambda t: slope(t.assign(err=t.gap), x))
        rows.append(dict(system=s, n=len(j), n_sequences=j.sequence.nunique(), mean_gap=j.gap.mean(), gap_lo=gi[0], gap_hi=gi[1], trend_of_gap_with=lab, slope=est, slope_lo=lo, slope_hi=hi, flat=bool(lo <= 0 <= hi)))
T = pd.DataFrame(rows); T.to_csv(A.out / "results/decision_rule.csv", index=False)
print(T.round(3).to_string())
d = R[R.system == "D: VLM grounded"].set_index("qid").join(R[R.system == "Detector at D's threshold"].set_index("qid")[["err"]], rsuffix="_det").reset_index()
if len(d):
    d["gap"] = d.err - d.err_det
    lo, hi = boot(d, lambda t: t.gap.mean())
    pd.DataFrame([dict(comparison="D minus the detector at D's threshold", n=len(d), mean=d.gap.mean(), lo=lo, hi=hi, grounded_matches=bool(lo <= 0))]).to_csv(A.out / "results/grounded_vs_its_detector.csv", index=False)

# ---- figures -------------------------------------------------------------------------------------------------------------
plt = plots._plt()
COL = {"A: detector, dev-tuned": "#1f6fb4", "A: VisDrone detector under shift": "#6aa7d8", "B: VLM direct": "#d9822b", "C: VLM localize": "#7a4fa3", "D: VLM grounded": "#2e8b57"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, by, order, lab in ((ax[0], "count_bin", ["0", "1-8", "9-20", "21-50", ">50"], "True count"), (ax[1], "size_bin", ["small", "mid", "large"], "Median object size (tertiles from VisDrone val)")):
    for s, c in COL.items():
        m = M[(M.system == s) & (M.by == by)].set_index("value").reindex(order)
        a.errorbar(range(len(order)), m.MAE, yerr=[(m.MAE - m.MAE_lo).clip(lower=0), (m.MAE_hi - m.MAE).clip(lower=0)], marker="o", ms=5, lw=1.5, color=c, label=s, capsize=2)
    a.set_xticks(range(len(order)), order); a.set_xlabel(lab); a.set_yscale("symlog", linthresh=1)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Mean absolute count error"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(A.out / "figures/error_vs_count_and_size.png", dpi=300); fig.savefig(A.out / "figures/error_vs_count_and_size.pdf")
