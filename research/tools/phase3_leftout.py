"""Phase 3: the questions left out of the paired comparison because system D ran out of GPU memory (very long box lists).
Reports what they are and how the detector (A) and the direct VLM (B) do on them, so the effect of leaving them out is visible.

Run:  python tools/phase3_leftout.py   ->  experiments/phase3/results/left_out_sensitivity.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "experiments/phase3"
lo = set(pd.read_csv(R / "results/left_out_gpu_errors.csv").qid)
q = pd.read_csv(ROOT / "countset/vlm_queries.csv").sample(frac=1, random_state=0).reset_index(drop=True)   # the notebook's query order
q["qid"] = range(len(q))
f = q[q.qid.isin(lo)]
P = {}
for name, key in (("in-domain_VisDrone", "visdrone"), ("in-domain_UAVDT", "uavdt"), ("shift_to_CARPK", "carpk")):
    d = pd.read_csv(ROOT / f"experiments/phase4/results/paired__{name}.csv.gz")
    d = d[d.region == "whole"]
    P[key] = {(str(r.image), r.target): r.yhat for r in d.itertuples()}
det_err = [abs(P[r.dataset][(str(r.image), r.target)] - r["count"]) for _, r in f.iterrows()]
a = pd.concat([pd.read_csv(R / "raw/run1/vlm_answers.csv").assign(run=0), pd.read_csv(R / "raw/rerun/vlm_answers.csv").assign(run=1)])
b = a[(a.system == "B") & a.qid.isin(lo) & a.error.isna()].sort_values("run").drop_duplicates("qid", keep="last")
fd = pd.read_csv(R / "raw/run1/detector_counts_for_d.csv")
out = dict(n_left_out=len(f), share_over_50=(f.count_bin == ">50").mean(), median_true_count=f["count"].median(),
           datasets=";".join(f"{k}:{v}" for k, v in f.dataset.value_counts().items()),
           detector_MAE=sum(det_err) / len(det_err), B_answered=len(b), B_MAE=(b.parsed.fillna(0) - b["count"]).abs().mean(),
           boxes_listed_median=fd[fd.qid.isin(lo)].detector_count.median(), boxes_listed_median_others=fd[~fd.qid.isin(lo)].detector_count.median())
pd.DataFrame([out]).to_csv(R / "results/left_out_sensitivity.csv", index=False)
print(out)
