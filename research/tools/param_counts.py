"""Parameter counts of the released weights, stored and after Ultralytics fuses batch normalization into the convolutions.
Run:  python tools/param_counts.py   ->  experiments/params/param_counts.csv
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from ultralytics import YOLO, YOLOE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "skyquery/weights"
rows = []
for name, f, cls in (("YOLO26m-Tile", "yolo26m_tile_visdrone.pt", YOLO), ("YOLOE-26m-FT", "yoloe26m_e2_full.pt", YOLOE), ("YOLO26s", "yolo26s_visdrone_sahi.pt", YOLO)):
    m = cls(str(W / f)).model
    stored = sum(p.numel() for p in m.parameters())
    m.fuse()
    rows.append(dict(model=name, weights=f, params_stored=stored, params_fused=sum(p.numel() for p in m.parameters())))
(ROOT / "experiments/params").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT / "experiments/params/param_counts.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
