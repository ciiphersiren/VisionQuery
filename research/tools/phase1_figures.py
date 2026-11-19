"""Phase 1 figures from experiments/phase1/eval/results: transfer matrix heatmap and the YOLO26s (UAVDT) training curves.

Run:  python tools/phase1_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dronekit import plots  # noqa: E402

E = ROOT / "experiments/phase1/eval"
M = pd.read_csv(E / "results/transfer_matrix.csv")
M = M[M.van_rule == "van=car"]
ROWS = ["YOLO26m-Tile (VisDrone)", "YOLO26s (VisDrone)", "YOLO26m-Tile (UAVDT)", "YOLO26s (UAVDT)", "YOLO26m-Tile (UAVDT), last epoch", "YOLO26s (UAVDT), last epoch"]
COLS = ["visdrone/test-dev", "uavdt/test", "carpk/test", "carpk/train"]
LBL = ["VisDrone\ntest-dev", "UAVDT\ntest", "CARPK\ntest", "CARPK\n6 other lots"]
plt = plots._plt()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
for a, col, title in ((ax[0], "AP50", "AP50"), (ax[1], "mAP50-95", "mAP50-95")):
    G = M.pivot(index="model", columns="target", values=col).reindex(index=ROWS, columns=COLS)
    a.imshow(G.to_numpy(float), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            v = G.iloc[i, j]
            a.text(j, i, "not run" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=9, color="white" if not np.isnan(v) and v > .55 else "#10303a")
    a.set_xticks(range(len(COLS)), LBL, fontsize=8)
    a.set_yticks(range(len(ROWS)), ROWS if a is ax[0] else [""] * len(ROWS), fontsize=8)
    a.set_title(title + " on shared classes (VisDrone van = car)", fontsize=10)
    a.spines[:].set_visible(False)
    a.grid(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(E / f"figures/transfer_matrix.{ext}", dpi=300)
plots.plot_training_curves(ROOT / "experiments/phase1/results/retrain_yolo26s/results.csv", E / "figures/training_YOLO26s_UAVDT.png", "YOLO26s (UAVDT): training curves")
print("figures written")
