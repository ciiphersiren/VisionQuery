"""Figures: PR curves, normalized confusion matrix, per-class bars, training curves.

Paper figures (white background). Colors follow a validated categorical order; one-series
panels use slot 1, val vs test-dev comparisons use slots 1 and 2 with a legend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_CMAP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_pr_curves(curves, class_names, path, title=""):
    """curves: {label: (recall (R,), precision (R, nc))}, e.g. {"val": ..., "test-dev": ...}.
    Small multiples, one panel per class (10 classes would exceed the 8 categorical hues)."""
    plt = _plt()
    nc = len(class_names); ncol = 5; nrow = int(np.ceil(nc / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.2 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for k, c in enumerate(class_names):
        ax = axes[k]
        for j, (lbl, (rec, prec)) in enumerate(curves.items()):
            p = prec[:, k]
            ap = np.nanmean(p) if np.isfinite(p).any() else float("nan")
            ax.plot(rec, p, color=SERIES[j], lw=1.6, label=f"{lbl} (AP50 {ap:.2f})")
        ax.set_title(c, fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6.5, loc="lower left")
    for ax in axes[nc:]:
        ax.axis("off")
    for ax in axes[(nrow - 1) * ncol:]:
        ax.set_xlabel("recall")
    for ax in axes[::ncol]:
        ax.set_ylabel("precision")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm_norm, class_names, path, title="", conf=None):
    """Row-normalized confusion matrix (rows = true class + 'background' FP row)."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_CMAP)
    labels = list(class_names) + ["background"]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * n + 1.6, 0.55 * n + 1.2))
    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(n), labels, rotation=45, ha="right"); ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    sub = f" (conf ≥ {conf:.2f}, IoU ≥ 0.5)" if conf is not None else ""
    ax.set_title(title + sub, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_per_class_bars(df, path, title="", cols=("val", "test-dev")):
    """df: index = class names, columns include `cols` (AP50 values). Grouped horizontal bars."""
    plt = _plt()
    n = len(df); h = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * n + 1.0))
    y = np.arange(n)
    for j, c in enumerate(cols):
        if c not in df:
            continue
        ax.barh(y + (j - (len(cols) - 1) / 2) * (h + 0.02), df[c].values, height=h, color=SERIES[j], label=c)
    ax.set_yticks(y, df.index); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("AP50"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=len(cols))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_epoch_scan(df, seq_df, class_names, path, title=""):
    """Per-epoch val AP (IoU 0.50:0.95), three panels on one shared scale: the same detections under three
    scorers; eval-v2 per class; eval-v2 per sequence. df: one row per epoch; seq_df: epoch, sequence, AP."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharex=True, sharey=True)
    scorers = [("ultra_AP", "Ultralytics validator"), ("noign_AP", "eval-v2, ignore regions removed"),
               ("ev2_AP", "eval-v2, ignore regions as crowd")]
    for j, (col, lbl) in enumerate(scorers):
        axes[0].plot(df["epoch"], df[col], color=SERIES[j], lw=2, label=lbl)
        k = df[col].idxmax()
        axes[0].plot(df["epoch"][k], df[col][k], "o", color=SERIES[j], ms=8, mec="white", mew=2, zorder=5)
    axes[0].set_title("Scorer (dot = best epoch)", fontsize=9.5)
    for j, c in enumerate(class_names):
        axes[1].plot(df["epoch"], df[f"AP_{c}"], color=SERIES[j], lw=2, label=c)
    axes[1].set_title("Eval-v2, per class", fontsize=9.5)
    for j, (s, g) in enumerate(seq_df.groupby("sequence")):
        axes[2].plot(g["epoch"], g["AP"], color=SERIES[j], lw=2, label=s)
    axes[2].set_title("Eval-v2, per val sequence", fontsize=9.5)
    axes[0].set_ylabel("AP (IoU 0.50:0.95)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2 if ax is not axes[0] else 1)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curves(results_csv, path, title=""):
    """Ultralytics results.csv -> small multiples: each loss and each validator metric in its own panel.
    Note: these metrics come from the Ultralytics validator, not the eval-v2 scorer."""
    import pandas as pd
    plt = _plt()
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    x = df["epoch"] if "epoch" in df else np.arange(1, len(df) + 1)
    cols = [c for c in df.columns if c.startswith(("train/", "val/", "metrics/"))]
    ncol = 4; nrow = int(np.ceil(len(cols) / ncol)) or 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.0 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        ax.plot(x, df[c], color=SERIES[0], lw=1.6, marker="o", ms=3)
        ax.set_title(c.replace("metrics/", "").replace("(B)", ""), fontsize=8.5)
    for ax in axes[len(cols):]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel("epoch")
    fig.suptitle(title + "  (validator metrics: Ultralytics, imgsz from training)", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, path)
