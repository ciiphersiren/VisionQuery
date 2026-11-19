"""Training provenance from Ultralytics run files, and small markdown-table helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_results_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def read_run(weights=None, results_csv=None, args_yaml=None):
    """Collect what a training run recorded: checkpoint train_args, args.yaml, results.csv.
    Every value comes from those files; missing ones stay None."""
    info, ta = {}, {}
    if weights is not None and Path(weights).exists():
        try:
            import torch
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            ta = dict(ck.get("train_args") or {})
            info.update(ultralytics_version=ck.get("version"), checkpoint_date=ck.get("date"))
        except Exception as e:                       # keep going; provenance is best-effort per field
            info["checkpoint_error"] = repr(e)[:200]
    if args_yaml is not None and Path(args_yaml).exists():
        import yaml
        ta = {**ta, **(yaml.safe_load(Path(args_yaml).read_text()) or {})}
    for k in ("model", "data", "imgsz", "batch", "optimizer", "lr0", "epochs", "time", "seed", "device"):
        info[f"arg_{k}"] = ta.get(k)
    if results_csv is not None and Path(results_csv).exists():
        df = read_results_csv(results_csv)
        info["epochs_completed"] = int(len(df))
        if "time" in df:
            info["train_hours"] = round(float(df["time"].iloc[-1]) / 3600, 2)
        key = "metrics/mAP50-95(B)"
        if key in df:
            b = int(df[key].idxmax())
            info["best_epoch"] = int(df["epoch"].iloc[b]) if "epoch" in df else b + 1
            info["best_val_mAP50-95_ultralytics"] = round(float(df[key].iloc[b]), 4)
            if "metrics/mAP50(B)" in df:
                info["best_val_mAP50_ultralytics"] = round(float(df["metrics/mAP50(B)"].iloc[b]), 4)
    return info


def md_table(df, floatfmt="{:.3f}"):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return "" if v is None else str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(lines)
