"""Phase 3 figure: mean absolute count error by true count and by object size, detector against the three VLM systems (paper version).
The empty-scene bin (true count 0, 10 questions) and the no-object size bin are left out because a log axis cannot show an error of 0;
their numbers are in results/metrics.csv.

Run:  python tools/phase3_figure.py   ->  experiments/phase3/figures/error_vs_count_and_size.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = pd.read_csv(ROOT / "experiments/phase3/results/metrics.csv")
SYS = [("A: detector, dev-tuned", "Detector (per-class thresholds)", "#1f6fb4", "-"),
       ("D: VLM grounded", "D: VLM given detector boxes", "#2e8b57", "-"),
       ("B: VLM direct", "B: VLM, direct answer", "#d9822b", "-"),
       ("C: VLM localize", "C: VLM, one point per object", "#7a4fa3", "--")]
plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
panels = ((axes[0], "count_bin", ["1-8", "9-20", "21-50", ">50"], "True count"),
          (axes[1], "size_bin", ["small", "mid", "large"], "Median object size (tertiles from VisDrone val)"))
for ax, by, order, xlabel in panels:
    for k, (s, lab, col, ls) in enumerate(SYS):
        m = M[(M.system == s) & (M.by == by)].set_index("value").reindex(order)
        x = np.arange(len(order)) + (k - 1.5) * 0.035                       # small offsets so the intervals do not overlap
        ax.errorbar(x, m.MAE, yerr=[(m.MAE - m.MAE_lo).clip(lower=0), (m.MAE_hi - m.MAE).clip(lower=0)], color=col, ls=ls, lw=1.6,
                    marker="o", ms=5, capsize=2, label=lab)
    n = M[(M.system == SYS[0][0]) & (M.by == by)].set_index("value").reindex(order)["n"]
    ax.set_xticks(range(len(order)), [f"{o}\n(n={int(v)})" for o, v in zip(order, n)])
    ax.set_xlabel(xlabel); ax.set_yscale("log")
    ax.grid(axis="y", color="#e3e3e3", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Mean absolute count error (log scale)")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
out = ROOT / "experiments/phase3/figures/error_vs_count_and_size"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
print("wrote", out.with_suffix(".pdf"))
