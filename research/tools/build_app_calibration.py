"""Write configs/models.json (repository root) and one calibration file per added model, from the capstone's result files.
Nothing here looks at a test split: thresholds and conformal q come from val (VisDrone val, UAVDT val).

Run:  python tools/build_app_calibration.py --uavdt-gt DIR --uavdt-attr DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpu_common import UV, load_merged

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from dronekit import countset as K, data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--uavdt-gt", type=Path, required=True)
ap.add_argument("--uavdt-attr", type=Path, required=True)
A = ap.parse_args()
CFG = ROOT.parent / "configs"
TH = json.loads((ROOT / "experiments/phase4/results/thresholds.json").read_text())


def calib(path, pipeline, thresholds, source, conformal=None):
    d = {"count_thresholds": thresholds, "exist_thresholds": thresholds, "global_threshold": float(np.median(list(thresholds.values()))), "source": source}
    if conformal:
        d["conformal"] = conformal
    (CFG / path).write_text(json.dumps({"version": "capstone-2026-11-09", "pipelines": {pipeline: d}}, indent=1))


# YOLO26m-Tile (VisDrone): Phase 4 per-class thresholds (VisDrone val, lowest whole-image MAE) and 90% conformal q
cov = pd.read_csv(ROOT / "experiments/phase4/results/coverage.csv")
c = cov[(cov.check == "in-domain VisDrone") & (cov.method == "sequence") & (cov.score == "sqrt") & (cov.mondrian_by == "target") & np.isclose(cov.target_coverage, .9)]
q = {r.stratum: round(float(r.q), 4) for r in c.itertuples() if r.stratum in TH["tile_vd"] and np.isfinite(r.q)}
calib("calibration_yolo26m_tile_visdrone.json", "sliced", TH["tile_vd"], "experiments/phase4/results/thresholds.json (VisDrone val); conformal q from experiments/phase4/results/coverage.csv",
      {"target": 0.9, "score": "|y - n| / sqrt(n + 1)", "method": "sequence-grouped split conformal, per class, calibrated on VisDrone val", "label": "90% range", "q": q})

# Ensemble: no counting thresholds were ever tuned for it; Phase 0's val best-F1 confidence for every class
b = pd.read_csv(ROOT / "experiments/phase0/results/basic_metrics.csv")
conf = float(b[(b.model == "Ensemble") & (b.split == "val") & (b.setup == "best_sliced")].conf.iloc[0])
calib("calibration_ensemble_visdrone.json", "sliced", {cl: conf for cl in TH["tile_vd"]}, f"experiments/phase0/results/basic_metrics.csv: Ensemble val best-F1 confidence {conf}, one value for all classes; not tuned for counting")

# YOLO26m-Tile (UAVDT): Phase 4 thresholds (UAVDT val)
calib("calibration_yolo26m_tile_uavdt.json", "fi1024", TH["tile_uv"], "experiments/phase4/results/thresholds.json (UAVDT val, 5 sequences). Bus threshold sits at the top of the grid: val had too few buses")

# YOLO26s (UAVDT): tuned here the same way, on UAVDT val
man = json.loads((ROOT / "experiments/phase1/results/retrain_tile/uavdt_split.json").read_text())
ix = pd.read_csv(ROOT / "experiments/phase1/eval/results/image_index.csv")
sizes = {s: (int(w), int(h)) for s, w, h in ix[ix.dataset == "uavdt"][["sequence", "width", "height"]].drop_duplicates().itertuples(index=False)}
recs = []
for s in man["val_seqs"]:
    recs += data.load_uavdt(Path("x"), A.uavdt_gt, A.uavdt_attr, [s], 10, "val", size=sizes[s]).images
val = data.Split("uavdt", "val", list(UV), recs)
assert len(val) == 367
dets = load_merged(ROOT / "experiments/phase1/eval/merged", "s_uv", ("uavdt", "val"))
th = K.tune_thresholds(val, dets, [0, 1, 2])
calib("calibration_yolo26s_uavdt.json", "fi1024", {UV[k]: v for k, v in th.items()}, "tuned by tools/build_app_calibration.py on UAVDT val (lowest whole-image MAE), same rule as Phase 4")

VD = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
models = {
    "default": "yolo26s_visdrone",
    "models": {
        "yolo26s_visdrone": {"name": "YOLO26s (VisDrone)", "weights": "weights/yolo26s_visdrone_sahi.pt", "classes": VD, "pipelines": ["fi1280", "fi1280+fi1920"],
                             "calibration": "configs/calibration.json", "notes": "Teammate's model and calibration. Fast on CPU."},
        "yolo26m_tile_visdrone": {"name": "YOLO26m-Tile (VisDrone)", "weights": "weights/yolo26m_tile_visdrone.pt", "classes": VD, "pipelines": ["sliced"],
                                  "calibration": "configs/calibration_yolo26m_tile_visdrone.json",
                                  "notes": "Strongest single model (test-dev AP50 0.540). Counts carry 90% conformal ranges. Sliced: slow on CPU for a new photo."},
        "ensemble_visdrone": {"name": "Ensemble: YOLO26m-Tile + YOLOE-26m-FT", "weights": ["weights/yolo26m_tile_visdrone.pt", "weights/yoloe26m_e2_full.pt"], "classes": VD,
                              "pipelines": ["sliced"], "calibration": "configs/calibration_ensemble_visdrone.json",
                              "notes": "Best AP (test-dev AP50 0.555), about twice the time. One threshold for all classes, not tuned for counting."},
        "yolo26m_tile_uavdt": {"name": "YOLO26m-Tile (UAVDT)", "weights": "weights/yolo26m_tile_uavdt.pt", "trained_on": "UAVDT", "classes": ["car", "truck", "bus"],
                               "pipelines": ["fi1024"], "calibration": "configs/calibration_yolo26m_tile_uavdt.json",
                               "notes": "Car, truck and bus only. Selected at epoch 1 by the project's rule. Hardly ever counts buses."},
        "yolo26s_uavdt": {"name": "YOLO26s (UAVDT)", "weights": "weights/yolo26s_uavdt.pt", "trained_on": "UAVDT", "classes": ["car", "truck", "bus"],
                          "pipelines": ["fi1024"], "calibration": "configs/calibration_yolo26s_uavdt.json", "notes": "Car, truck and bus only. Selected at epoch 1 by the project's rule."},
    },
}
(CFG / "models.json").write_text(json.dumps(models, indent=1))
print("conformal q:", q)
print("ensemble conf:", conf, "| tile_uv:", TH["tile_uv"], "| s_uv:", {UV[k]: v for k, v in th.items()})
