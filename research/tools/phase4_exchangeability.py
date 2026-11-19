"""Phase 4, exploratory (added 10 Nov 2026 after the coverage results): are VisDrone val and test-dev exchangeable at the sequence level?

The paper explained the unmet per-sequence coverage (82% of test sequences at target, expected >= 90%) by a difference between val and test-dev
in density and class mix, without testing it. This script tests it from the labels only (count set), with sequences as the unit:
  1. Sequence-level permutation tests of val vs test-dev on scene features (objects per image, object size, density, occlusion) and class mix.
  2. Whether the test sequences that are least like val are the ones whose records are covered less often, using the same 90% sequence-method
     threshold as the paper (q from coverage.csv) on the paired test-dev records.

Run:  python tools/phase4_exchangeability.py   ->  experiments/phase4/results/exchangeability*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase4/results"
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
N_PERM, SEED = 10000, 0
rng = np.random.default_rng(SEED)


def load(split):
    d = pd.read_json(ROOT / f"countset/visdrone__{split}.jsonl.gz", lines=True)
    return d[(d.region == "whole") & d.target.isin(CLASSES)].assign(split=split)


R = pd.concat([load("val"), load("test-dev")], ignore_index=True)
R["sequence"] = R.sequence.astype(str)

# per image: objects, and count-weighted size / density / occlusion over the classes present
pos = R[R["count"] > 0].copy()
pos["w_logside"] = np.log(pos.median_side) * pos["count"]
pos["w_dens"] = pos.density.fillna(pos.density.median()) * pos["count"]
pos["w_occ"] = pos.frac_occluded * pos["count"]
img = R.groupby(["split", "sequence", "image"], as_index=False)["count"].sum().rename(columns={"count": "objects"})
wi = pos.groupby(["split", "sequence", "image"], as_index=False)[["w_logside", "w_dens", "w_occ", "count"]].sum()
img = img.merge(wi, on=["split", "sequence", "image"], how="left")
img["median_side_px"] = np.exp(img.w_logside / img["count"])
img["density"] = img.w_dens / img["count"]
img["occluded_share"] = img.w_occ / img["count"]
FEAT = {"objects": "objects per image", "median_side_px": "object size (px, geometric mean of median side)",
        "density": "local spacing (nearest-neighbour distance in box sides)", "occluded_share": "share of occluded objects"}
seq = img.groupby(["split", "sequence"], as_index=False)[list(FEAT)].mean()
shares = R.pivot_table(index=["split", "sequence"], columns="target", values="count", aggfunc="sum", fill_value=0)[CLASSES]
shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reset_index()
seq = seq.merge(shares, on=["split", "sequence"])
lab = (seq.split == "test-dev").to_numpy()


def perm_p(x, stat):
    obs = stat(x, lab)
    null = np.array([stat(x, rng.permutation(lab)) for _ in range(N_PERM)])
    return obs, (1 + (np.abs(null) >= abs(obs)).sum()) / (N_PERM + 1)


rows = []
for f, name in FEAT.items():
    x = seq[f].to_numpy(float)
    obs, p = perm_p(x, lambda v, l: v[l].mean() - v[~l].mean())
    rows.append(dict(feature=name, val_mean=x[~lab].mean(), testdev_mean=x[lab].mean(), difference=obs, p_perm=p))
S = seq[CLASSES].to_numpy(float)
tv = lambda v, l: 0.5 * np.abs(v[l].mean(axis=0) - v[~l].mean(axis=0)).sum()                    # total-variation distance of mean class mix
obs, p = perm_p(S, tv)
rows.append(dict(feature="class mix (total-variation distance)", val_mean=np.nan, testdev_mean=np.nan, difference=obs, p_perm=p))
T = pd.DataFrame(rows)
order = np.argsort(T.p_perm.to_numpy())                                                    # Holm step-down over the five tests
adj = np.minimum(1, np.maximum.accumulate(T.p_perm.to_numpy()[order] * (len(T) - np.arange(len(T)))))
T["p_holm"] = np.empty(len(T)); T.loc[T.index[order], "p_holm"] = adj
T.insert(0, "n_sequences", f"val {(~lab).sum()}, test-dev {lab.sum()}")
T.to_csv(OUT / "exchangeability_tests.csv", index=False)
shares_tab = pd.DataFrame({"class": CLASSES, "val_share": S[~lab].mean(axis=0), "testdev_share": S[lab].mean(axis=0)})
shares_tab.to_csv(OUT / "exchangeability_class_mix.csv", index=False)

# 2) link to coverage: the paper's 90% sequence-method threshold on the paired test-dev records
C = pd.read_csv(OUT / "coverage.csv")
row = C[(C.check == "in-domain VisDrone") & (C.method == "sequence") & np.isclose(C.target_coverage, 0.9) & (C.score == "sqrt")
        & (C.mondrian_by == "none")].iloc[0]
q = float(row.q)
P = pd.read_csv(OUT / "paired__in-domain_VisDrone.csv.gz")
P = P[P.region == "whole"]
r = q * np.sqrt(P.yhat + 1)
P["covered"] = (P.y >= np.maximum(np.ceil(P.yhat - r - 1e-9), 0)) & (P.y <= np.floor(P.yhat + r + 1e-9))
P["sequence"] = P.sequence.astype(str)
cov = P.groupby("sequence").covered.mean().rename("coverage").reset_index()
check = dict(pooled_coverage=P.covered.mean(), share_at_target=(cov.coverage >= 0.9).mean(), paper_coverage=row.coverage,
             paper_share_at_target=row.share_sequences_at_target)
te = seq[lab].merge(cov, on="sequence")
# how unlike val is each test sequence: mean |z| over the scene features, z relative to the val sequences
va = seq[~lab]
z = np.column_stack([(te[f] - va[f].mean()) / va[f].std(ddof=1) for f in FEAT])
te["unlikeness"] = np.abs(z).mean(axis=1)
rho, p_rho = stats.spearmanr(te.unlikeness, te.coverage)
link = []
for f in list(FEAT) + ["unlikeness"]:
    rr, pp = stats.spearmanr(te[f], te.coverage)
    link.append(dict(feature=FEAT.get(f, "unlikeness to val (mean |z|)"), spearman_with_coverage=rr, p=pp,
                     mean_below_target=te.loc[te.coverage < 0.9, f].mean(), mean_at_target=te.loc[te.coverage >= 0.9, f].mean()))
L = pd.DataFrame(link)
L.insert(0, "n_test_sequences", len(te))
L.to_csv(OUT / "exchangeability_coverage_link.csv", index=False)
# 3) is 50 of 61 sequences at target unusual under exchangeability? With K calibration sequences and rank k = ceil((K+1)(1-alpha)), the chance c
# that a new sequence is at target, given the calibration set, is Beta(k, K+1-k) under exchangeability (no ties); the number of test sequences at
# target is then Binomial(n, c). Probability of seeing as few as observed:
K = int(row.n_cal_sequences); k = int(np.ceil((K + 1) * 0.9)); n = len(cov); x = int((cov.coverage >= 0.9).sum())
p_low = stats.betabinom(n, k, K + 1 - k).cdf(x)
check.update(n_cal_sequences=K, rank=k, n_test_sequences=n, at_target=x, expected_share=k / (K + 1), p_as_few_under_exchangeability=p_low)
pd.DataFrame([dict(q=q, **check)]).to_csv(OUT / "exchangeability_check.csv", index=False)
pd.set_option("display.width", 200)
print(T.round(4).to_string(index=False)); print(shares_tab.round(3).to_string(index=False)); print(L.round(3).to_string(index=False))
print({k: round(float(v), 3) for k, v in check.items()}, "q =", q)
