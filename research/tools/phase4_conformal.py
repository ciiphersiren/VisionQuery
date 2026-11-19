"""Phase 4: conformal count intervals from merged detections and labels (CPU). The plan in the capstone report was fixed before any result.

Run:  python tools/phase4_conformal.py --visdrone DIR --uavdt-gt DIR --uavdt-attr DIR --carpk DIR --merged .../merged --out phase4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cpu_common import CLASSES, UV, VD, add_paths, dev_specs_for_shift, load_merged, load_splits, specs, table

from dronekit import conformal as C, countset as K, plots

ALPHAS = (0.1, 0.2)
TARGET_K = (5, 10, 25, 50)
N_BOOT = 1000

ap = argparse.ArgumentParser()
add_paths(ap)
ap.add_argument("--out", type=Path, default=Path("experiments/phase4"))
A = ap.parse_args()
(A.out / "results").mkdir(parents=True, exist_ok=True)
(A.out / "figures").mkdir(parents=True, exist_ok=True)
SP = load_splits(A)
E = json.loads((Path(__file__).resolve().parents[1] / "countset/tertiles.json").read_text())


def add_strata(d):
    d["pred_size_bin"] = K.tertile(d.pred_side, E["side"], ("small", "mid", "large"))
    d["pred_density_bin"] = K.tertile(d.pred_density, E["density"], ("dense", "mid", "sparse"))
    d["gt_size_bin"] = K.tertile(d.gt_side, E["side"], ("small", "mid", "large"))
    d["gt_density_bin"] = K.tertile(d.gt_density, E["density"], ("dense", "mid", "sparse"))
    return d


def thresholds(tag, classes, key):
    dets = load_merged(A.merged, tag, key)
    sp = SP[key]
    # model class ids are the dev split's ids here, because dev is the model's own dataset
    return K.tune_thresholds(sp, dets, list(range(len(classes)))), dets


TH = {}
TH["tile_vd"], DV = thresholds("tile_vd", VD, ("visdrone", "val"))
TH["tile_uv"], DU = thresholds("tile_uv", UV, ("uavdt", "val"))
TH["tile_uv_last"], DUL = thresholds("tile_uv_last", UV, ("uavdt", "val"))
(A.out / "results/thresholds.json").write_text(json.dumps({t: {(VD if t == "tile_vd" else UV)[k]: v for k, v in x.items()} for t, x in TH.items()}, indent=1))

# ---- the paired tables: (name, model, calibration table, evaluation table) ---------------------------------------------
pv, gv = specs(VD, "visdrone", False)
CAL_VD = add_strata(table(SP[("visdrone", "val")], DV, TH["tile_vd"], pv, gv))
TEST_VD = add_strata(table(SP[("visdrone", "test-dev")], load_merged(A.merged, "tile_vd", ("visdrone", "test-dev")), TH["tile_vd"], pv, gv))
pu, gu = specs(UV, "uavdt", False)
CAL_UV = add_strata(table(SP[("uavdt", "val")], DU, TH["tile_uv"], pu, gu))
TEST_UV = add_strata(table(SP[("uavdt", "test")], load_merged(A.merged, "tile_uv", ("uavdt", "test")), TH["tile_uv"], pu, gu))
CAL_UVL = add_strata(table(SP[("uavdt", "val")], DUL, TH["tile_uv_last"], pu, gu))
TEST_UVL = add_strata(table(SP[("uavdt", "test")], load_merged(A.merged, "tile_uv_last", ("uavdt", "test")), TH["tile_uv_last"], pu, gu))


def shift_tables(carvan):
    ps, gs = dev_specs_for_shift(VD, carvan)
    cal = add_strata(table(SP[("visdrone", "val")], DV, TH["tile_vd"], ps, gs))
    pu_, gu_ = specs(VD, "uavdt", True, carvan)
    uv_ = add_strata(table(SP[("uavdt", "test")], load_merged(A.merged, "tile_vd", ("uavdt", "test")), TH["tile_vd"], pu_, gu_))
    pc, gc = specs(VD, "carpk", True, carvan)
    cp = pd.concat([table(SP[("carpk", s)], load_merged(A.merged, "tile_vd", ("carpk", s)), TH["tile_vd"], pc, gc) for s in ("test", "train")], ignore_index=True)
    return cal, uv_, add_strata(cp)


CAL_S, UV_S, CP_S = shift_tables(True)
CAL_S0, UV_S0, CP_S0 = shift_tables(False)
CHECKS = [("in-domain VisDrone", "YOLO26m-Tile (VisDrone)", CAL_VD, TEST_VD),
          ("in-domain UAVDT", "YOLO26m-Tile (UAVDT)", CAL_UV, TEST_UV),
          ("in-domain UAVDT, last epoch", "YOLO26m-Tile (UAVDT), last epoch", CAL_UVL, TEST_UVL),
          ("shift to UAVDT", "YOLO26m-Tile (VisDrone)", CAL_S, UV_S),
          ("shift to CARPK", "YOLO26m-Tile (VisDrone)", CAL_S[CAL_S.target == "car"], CP_S),
          ("shift to UAVDT, car only", "YOLO26m-Tile (VisDrone)", CAL_S0, UV_S0),
          ("shift to CARPK, car only", "YOLO26m-Tile (VisDrone)", CAL_S0[CAL_S0.target == "car"], CP_S0)]
for name, _, cal, tst in CHECKS:
    tst.assign(check=name).to_csv(A.out / "results" / ("paired__" + name.replace(" ", "_").replace(",", "") + ".csv.gz"), index=False)


def report(check, model, method, score, alpha, strat, stratum, ev, lo, hi, q=np.nan, n_cal_seq=np.nan):
    y = ev.y.to_numpy()
    cov = ((y >= lo) & (y <= hi)).astype(float)
    a, b = C.bootstrap_ci(cov, ev.sequence.to_numpy(), n=N_BOOT) if len(ev) and ev.sequence.nunique() > 1 else (np.nan, np.nan)
    fin = np.isfinite(hi)
    sc = C.sequence_coverage(y, lo, hi, ev.sequence.to_numpy()) if len(ev) else {}
    return dict(share_sequences_at_target=float(np.mean([v >= (1 - alpha) for v in sc.values()])) if sc else np.nan, check=check, model=model, method=method, score=score, target_coverage=1 - alpha, mondrian_by=strat, stratum=stratum, n_records=len(ev), n_sequences=ev.sequence.nunique(), n_cal_sequences=n_cal_seq, q=q,
                coverage=cov.mean() if len(ev) else np.nan, cov_lo=a, cov_hi=b, mean_width=float((hi - lo)[fin].mean()) if fin.any() else np.inf, share_infinite=float((~fin).mean()) if len(ev) else np.nan,
                median_width=float(np.median(hi - lo)) if len(ev) else np.nan, seq_cov_min=min(C.sequence_coverage(y, lo, hi, ev.sequence.to_numpy()).values()) if len(ev) else np.nan)


def by_gt_strata(row_base, ev, lo, hi):
    """Coverage per ground-truth stratum, for one set of intervals."""
    out = []
    for col in ("target", "count_bin", "gt_size_bin", "gt_density_bin", "altitude"):
        if col == "altitude" and ev.altitude.isna().all():
            continue
        for v, ix in ev.groupby(col, dropna=True).indices.items():
            r = report(*row_base, col, v, ev.iloc[ix], lo[ix], hi[ix])
            out.append(r)
    return out


rows, strata_rows = [], []
for check, model, cal, tst in CHECKS:
    for alpha in ALPHAS:
        # Baselines without conformal calibration
        for t, ev in tst.groupby("target"):
            cc = cal[cal.target == t]
            lo, hi = C.poisson_interval(ev.yhat.to_numpy(), alpha)
            rows.append(report(check, model, "poisson", "-", alpha, "target", t, ev, lo, hi))
            if alpha == 0.2 and len(cc):                                    # the 10th-90th percentile range is an 80% range
                lo, hi = C.residual_range(ev.yhat.to_numpy(), cc.y.to_numpy(), cc.yhat.to_numpy())
                rows.append(report(check, model, "residual 10-90", "-", alpha, "target", t, ev, lo, hi))
        for kind in C.SCORES:
            sc = C.score(cal.y, cal.yhat, kind)
            for method in C.METHODS:
                # overall: one threshold across all targets, then Mondrian by class, predicted size and predicted density
                q = C.calibrate(sc, cal.sequence, alpha, method)
                lo, hi = C.interval(tst.yhat.to_numpy(), q, kind)
                rows.append(report(check, model, method, kind, alpha, "none", "all", tst, lo, hi, q, cal.sequence.nunique()))
                if kind == "sqrt":
                    strata_rows += by_gt_strata((check, model, method, kind, alpha), tst.reset_index(drop=True), lo, hi)
                for strat, col_cal in (("target", "target"), ("pred_size_bin", "pred_size_bin"), ("pred_density_bin", "pred_density_bin")):
                    Q = C.mondrian(sc, cal[col_cal], cal.sequence, alpha, method)
                    qv = tst[col_cal].map(Q).astype(float).fillna(np.inf).to_numpy()
                    lo = np.zeros(len(tst)); hi = np.zeros(len(tst))
                    for v in np.unique(qv):
                        m = qv == v
                        lo[m], hi[m] = C.interval(tst.yhat.to_numpy()[m], v, kind)
                    for s, ix in tst.reset_index(drop=True).groupby(col_cal).indices.items():
                        rows.append(report(check, model, method, kind, alpha, strat, s, tst.iloc[ix], lo[ix], hi[ix], Q.get(s, np.inf), cal[cal[col_cal] == s].sequence.nunique()))
                    if kind == "sqrt":
                        strata_rows += [dict(r, mondrian=strat) for r in by_gt_strata((check, model, method, kind, alpha), tst.reset_index(drop=True), lo, hi)]
COV = pd.DataFrame(rows); COV.to_csv(A.out / "results/coverage.csv", index=False)
STR = pd.DataFrame(strata_rows); STR.to_csv(A.out / "results/coverage_by_gt_stratum.csv", index=False)

# ---- few-shot recalibration under shift: car, whole image ---------------------------------------------------------------
few = []
for check, cal, tst in (("shift to UAVDT", CAL_S, UV_S), ("shift to CARPK", CAL_S, CP_S)):
    cc, tt = cal[cal.target == "car"], tst[tst.target == "car"].sort_values(["sequence", "image"]).reset_index(drop=True)
    scc = C.score(cc.y, cc.yhat)
    by = {s: g.index.to_numpy() for s, g in tt.groupby("sequence")}
    for alpha in ALPHAS:
        q0 = C.calibrate(scc, cc.sequence, alpha, "sequence")
        lo, hi = C.interval(tt.yhat.to_numpy(), q0)
        few.append(dict(check=check, alpha=alpha, method="source only", k=0, draw=-1, coverage=C.coverage(tt.y, lo, hi), mean_width=C.mean_width(lo, hi), n_eval=len(tt), n_eval_seq=len(by), share_infinite=float(np.isinf(hi).mean())))
        for k in TARGET_K:
            for i, draw in enumerate(C.few_shot_draws({s: len(v) for s, v in by.items()}, k, 20, seed=k)):
                idx = np.concatenate([by[s][:n] for s, n in draw])
                used = {s for s, _ in draw}
                ev = tt[~tt.sequence.isin(used)]
                if not len(ev):
                    continue
                cy, cyh = tt.y.to_numpy()[idx], tt.yhat.to_numpy()[idx]
                base = dict(check=check, alpha=alpha, k=k, draw=i, n_eval=len(ev), n_eval_seq=ev.sequence.nunique(), n_cal_seq=len(used))
                q1 = C.conformal_quantile(C.score(cy, cyh), alpha)
                lo, hi = C.interval(ev.yhat.to_numpy(), q1)
                few.append(dict(base, method="R1 target only", coverage=C.coverage(ev.y, lo, hi), mean_width=C.mean_width(lo, hi) if np.isfinite(q1) else np.inf, share_infinite=float(np.isinf(q1))))
                c = cy.sum() / max(cyh.sum(), 1)
                lo, hi = C.interval(c * ev.yhat.to_numpy(), q0)
                few.append(dict(base, method="R2 ratio + source q", coverage=C.coverage(ev.y, lo, hi), mean_width=C.mean_width(lo, hi) if np.isfinite(q0) else np.inf, share_infinite=float(np.isinf(q0))))
FEW = pd.DataFrame(few); FEW.to_csv(A.out / "results/few_shot_draws.csv", index=False)
FS = FEW.replace(np.inf, np.nan).groupby(["check", "alpha", "method", "k"]).agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"), mean_width=("mean_width", "mean"), share_infinite=("share_infinite", "mean"), n_eval_seq=("n_eval_seq", "mean"), draws=("coverage", "size")).reset_index()
FS.to_csv(A.out / "results/few_shot_summary.csv", index=False)

# ---- exploratory, added after seeing the few-shot result: whole labeled target sequences instead of frames --------------
exp = []
for check, tst, Ks in (("shift to UAVDT", UV_S, (1, 2, 3, 5, 9, 12, 15)), ("shift to CARPK", CP_S, (1, 2, 3, 5))):
    tt = tst[tst.target == "car"].reset_index(drop=True)
    seqs = sorted(tt.sequence.unique())
    sc_t = C.score(tt.y, tt.yhat)
    rng = np.random.default_rng(0)
    for K in Ks:
        for i in range(20):
            pick = set(rng.permutation(seqs)[:K])
            m = tt.sequence.isin(pick).to_numpy()
            ev = tt[~m]
            for alpha in ALPHAS:
                for method in ("sequence", "pooled"):
                    q = C.calibrate(sc_t[m], tt.sequence.to_numpy()[m], alpha, method)
                    lo, hi = C.interval(ev.yhat.to_numpy(), q)
                    cov = C.sequence_coverage(ev.y.to_numpy(), lo, hi, ev.sequence.to_numpy())
                    exp.append(dict(check=check, K=K, draw=i, alpha=alpha, method=method, frames=int(m.sum()), coverage=C.coverage(ev.y, lo, hi), share_sequences_at_target=float(np.mean([v >= 1 - alpha for v in cov.values()])),
                                    mean_width=C.mean_width(lo, hi) if np.isfinite(q) else np.nan, infinite=float(np.isinf(q)), n_eval_seq=ev.sequence.nunique()))
EXP = pd.DataFrame(exp); EXP.to_csv(A.out / "results/exploratory_whole_sequences_draws.csv", index=False)
ES = EXP.groupby(["check", "alpha", "method", "K"]).agg(frames=("frames", "mean"), coverage=("coverage", "mean"), cov_sd=("coverage", "std"), share_sequences_at_target=("share_sequences_at_target", "mean"), mean_width=("mean_width", "mean"), share_infinite=("infinite", "mean"), n_eval_seq=("n_eval_seq", "mean")).reset_index()
ES.to_csv(A.out / "results/exploratory_whole_sequences.csv", index=False)

# ---- regions (secondary scope in the plan): left, right and quadrants, in-domain VisDrone and shift to UAVDT -------------------
REG = ("left", "right", "q1", "q2", "q3", "q4")
reg_rows = []
for check, cal, tst in (("in-domain VisDrone", table(SP[("visdrone", "val")], DV, TH["tile_vd"], pv, gv, REG), table(SP[("visdrone", "test-dev")], load_merged(A.merged, "tile_vd", ("visdrone", "test-dev")), TH["tile_vd"], pv, gv, REG)),
                        ("shift to UAVDT", table(SP[("visdrone", "val")], DV, TH["tile_vd"], *dev_specs_for_shift(VD), REG), table(SP[("uavdt", "test")], load_merged(A.merged, "tile_vd", ("uavdt", "test")), TH["tile_vd"], *specs(VD, "uavdt", True), REG))):
    for alpha in ALPHAS:
        for method in ("sequence", "image"):
            q = C.calibrate(C.score(cal.y, cal.yhat), cal.sequence, alpha, method)
            lo, hi = C.interval(tst.yhat.to_numpy(), q)
            reg_rows.append(report(check, "YOLO26m-Tile (VisDrone)", method, "sqrt", alpha, "region_kind", "all regions", tst, lo, hi, q, cal.sequence.nunique()))
            m = (tst.target == "car").to_numpy()
            ccar = cal[cal.target == "car"]
            qc = C.calibrate(C.score(ccar.y, ccar.yhat), ccar.sequence, alpha, method)
            lo, hi = C.interval(tst.yhat.to_numpy()[m], qc)
            reg_rows.append(report(check, "YOLO26m-Tile (VisDrone)", method, "sqrt", alpha, "region_kind", "car, all regions", tst[m], lo, hi, qc, ccar.sequence.nunique()))
pd.DataFrame(reg_rows).to_csv(A.out / "results/coverage_regions.csv", index=False)

# ---- figures ------------------------------------------------------------------------------------------------------------
plt = plots._plt()
METHOD_COL = {"sequence": "#1f6fb4", "pooled": "#d9822b", "image": "#8a8a8a"}
main = COV[(COV.mondrian_by == "none") & (COV.score == "sqrt") & COV.method.isin(C.METHODS) & ~COV.check.str.contains("car only|last")]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, tc in zip(ax, (0.9, 0.8)):
    d = main[np.isclose(main.target_coverage, tc)]
    ck = list(dict.fromkeys(d.check))
    for j, m in enumerate(C.METHODS):
        dd = d[d.method == m].set_index("check").reindex(ck)
        x = np.arange(len(ck)) + (j - 1) * .22
        for xi, inf in zip(x, dd.share_infinite == 1):                 # no finite interval: coverage of 1.0 would mislead
            if inf:
                a.text(xi, .5, "infinite", rotation=90, ha="center", va="center", fontsize=8, color=METHOD_COL[m])
        dd = dd.where(dd.share_infinite < 1)
        a.errorbar(x, dd.coverage, yerr=[(dd.coverage - dd.cov_lo).clip(lower=0), (dd.cov_hi - dd.coverage).clip(lower=0)], fmt="o", ms=5, lw=1.5, color=METHOD_COL[m], label=m)
    a.axhline(tc, color="#444", lw=1, ls="--")
    a.set_xticks(range(len(ck)), [c.replace(" ", "\n", 1) for c in ck], fontsize=8)
    a.set_title(f"Target {tc:.0%}", fontsize=10)
    a.set_ylim(0, 1.02)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Coverage, whole-image records")
ax[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(A.out / "figures/coverage_by_check.png", dpi=300); fig.savefig(A.out / "figures/coverage_by_check.pdf")

fig, ax = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True)
for c, check in enumerate(("shift to UAVDT", "shift to CARPK")):
    for m, col in (("R1 target only", "#1f6fb4"), ("R2 ratio + source q", "#d9822b")):
        d = FS[(FS.check == check) & (FS.method == m) & np.isclose(FS.alpha, 0.1)]
        ax[0, c].errorbar(d.k, d.coverage, yerr=d.cov_sd, marker="o", ms=5, lw=1.5, color=col, label=m)
        ax[1, c].plot(d.k, d.mean_width, marker="o", ms=5, lw=1.5, color=col, label=m)
    s = FS[(FS.check == check) & (FS.method == "source only") & np.isclose(FS.alpha, 0.1)]
    if len(s):
        ax[0, c].axhline(s.coverage.iloc[0], color="#8a8a8a", lw=1.2, label="source threshold, no target frames")
        ax[1, c].axhline(s.mean_width.iloc[0], color="#8a8a8a", lw=1.2, label="source threshold, no target frames")
    ax[0, c].axhline(.9, color="#444", lw=1, ls="--")
    ax[0, c].set_title(check, fontsize=10); ax[0, c].set_ylim(0, 1.02)
    ax[1, c].set_xlabel("Labeled target frames, k")
    for a in ax[:, c]:
        a.spines[["top", "right"]].set_visible(False)
ax[0, 0].set_ylabel("Coverage (target 90%)"); ax[1, 0].set_ylabel("Mean interval width")
ax[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(A.out / "figures/few_shot_recalibration.png", dpi=300); fig.savefig(A.out / "figures/few_shot_recalibration.pdf")
print(main[["check", "method", "target_coverage", "n_cal_sequences", "q", "coverage", "cov_lo", "cov_hi", "mean_width", "share_infinite", "seq_cov_min"]].round(3).to_string())
print(FS[np.isclose(FS.alpha, .1)].round(3).to_string())
print(ES.round(3).to_string())
print(pd.DataFrame(reg_rows)[["check", "method", "target_coverage", "stratum", "n_records", "coverage", "cov_lo", "cov_hi", "share_sequences_at_target", "mean_width"]].round(3).to_string())
# CARPK has at most 6 calibration lots, so every 90% interval there is infinite; only UAVDT is drawn (stated in the caption)
fig, a = plt.subplots(1, 1, figsize=(4.8, 3.2))
for m, col in (("sequence", "#1f6fb4"), ("pooled", "#d9822b")):
    d = ES[(ES.check == "shift to UAVDT") & (ES.method == m) & np.isclose(ES.alpha, 0.1)]
    d = d[d.share_infinite < 1]
    a.errorbar(d.K, d.coverage, yerr=d.cov_sd, marker="o", ms=5, lw=1.5, color=col, label=m, capsize=2)
a.axhline(.9, color="#444", lw=1, ls="--")
a.set_title("shift to UAVDT (exploratory)", fontsize=10); a.set_xlabel("Labeled whole target sequences"); a.set_ylim(0, 1.02)
a.spines[["top", "right"]].set_visible(False)
a.set_ylabel("Coverage (target 90%)"); a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(A.out / "figures/exploratory_whole_sequences.png", dpi=300); fig.savefig(A.out / "figures/exploratory_whole_sequences.pdf")
