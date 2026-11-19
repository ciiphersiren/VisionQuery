"""CPU latency of every app model on 10 VisDrone test-dev images (first image per sequence, sorted), one warm-up image first.
Writes experiments/latency/cpu_latency.csv. Hardware is recorded from the OS.

Run:  python tools/cpu_latency.py --visdrone DIR
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from skyquery.detector import Detector  # noqa: E402
from skyquery.models import load_models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--visdrone", type=Path, required=True)
ap.add_argument("--n", type=int, default=10)
A = ap.parse_args()
imgs = sorted((A.visdrone / "VisDrone2019-DET-test-dev/images").glob("*.jpg"))
seen, pick = set(), []
for p in imgs:
    s = p.stem.split("_")[0]
    if s not in seen:
        seen.add(s); pick.append(p)
pick = pick[:A.n + 1]
models, _ = load_models(ROOT.parent / "configs/models.json", ROOT.parent, "", "")
rows = []
for mid, spec in models.items():
    if not spec.available:
        continue
    for pipe in spec.pipelines:
        det = Detector(spec.weights, pipeline=pipe, device="cpu", conf_floor=0.05, max_det=1000)
        det(cv2.imread(str(pick[0])))
        ts = []
        for p in pick[1:]:
            im = cv2.imread(str(p))
            t0 = time.perf_counter(); d = det(im); ts.append(time.perf_counter() - t0)
        rows.append(dict(model=spec.name, pipeline=pipe, images=len(ts), mean_s=round(float(np.mean(ts)), 2), median_s=round(float(np.median(ts)), 2),
                         min_s=round(float(np.min(ts)), 2), max_s=round(float(np.max(ts)), 2)))
        print(rows[-1], flush=True)
out = ROOT / "experiments/latency"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df["cpu"] = platform.processor() or platform.machine()
df["threads"] = os.cpu_count()
df.to_csv(out / "cpu_latency.csv", index=False)
print(df.to_string())
