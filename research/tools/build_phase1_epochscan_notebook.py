"""Build notebooks/phase1_epochscan.ipynb: score every saved epoch of uavdt_y26m_tiles on UAVDT val with eval v2.

Run:  python tools/build_phase1_epochscan_notebook.py
"""
from nb_common import EPOCH_INFER_CELL, EPOCH_SCORE_CELL, EPOCH_SUMMARY_CELL, UAVDT_FETCH_CELL, VAL_TABLE_CODE, Notebook

nb = Notebook()
nb.md(r"""
# Phase 1, step 1: why does the UAVDT run peak at epoch 1?

The first UAVDT run's `best.pt` is epoch 1 under the Ultralytics validator. This notebook scores **every saved epoch** on UAVDT
val with the eval-v2 scorer. It uses `full@1024`, the validator's image size, so that only the scorer changes. Each epoch's
detections are scored twice: ignore regions as crowd (eval v2), and ignore regions removed (behaves like the validator).
**UAVDT test is not loaded here.**

## How to run (about 30-60 min)
- **Inputs:** attach the saved output of the `phase1_train` notebook run (*Add Input → Your Work → Notebooks*). It must contain
  `runs/uavdt_y26m_tiles/weights/epoch0.pt` and `uavdt_split.json`.
- **Settings:** Accelerator **GPU T4 x2** (one GPU is used), **Internet ON** (re-downloads UAVDT, about 6 GB).
- *Save Version → Save & Run All*.
- **If Google Drive refuses the download:** upload `UAV-benchmark-M` as a private Kaggle Dataset and attach it too.
- **Send back** from `/kaggle/working/phase1_epochscan/`: `epoch_scan_summary.json`, `epoch_scan.csv`, `per_sequence.csv`,
  `val_sequences.csv` and `figures/epoch_scan.png`.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest gdown')

nb.code(r'''
# --- Config ----------------------------------------------------------------
import os, sys, re, json, shutil, zipfile, tarfile, subprocess, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, plots, score

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
RAW = Path(os.environ.get("DK_UAVDT_RAW", "/tmp/UAVDT/raw"))       # downloads + extracted release (lost at session end)
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "phase1_epochscan"
CUDA = torch.cuda.is_available()
SMOKE = not CUDA                 # no GPU: first and last checkpoints on a few frames, only to check the pipeline
RUN_NAME = "uavdt_y26m_tiles"
PASS = "full@1024"               # the validator's imgsz, so that only the scorer differs from results.csv
SETUP = dict(passes=[PASS], method=None)
IGNORE_CONF = 0.25               # for the share of detections inside ignore regions
EARLY_PEAK = 2                   # the retrain rule: best eval-v2 epoch <= this
EXPECTED_VAL_FRAMES = 414
(OUT / "cache").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)

WDIR = next(iter(sorted(INPUT.rglob(f"{RUN_NAME}/weights/epoch0.pt"))), None)
assert WDIR, f"attach the phase1_train output: no {RUN_NAME}/weights/epoch0.pt under {INPUT}"
WDIR = WDIR.parent
MANIFEST = json.loads(next(iter(sorted(INPUT.rglob("uavdt_split.json")))).read_text())
RESULTS = pd.read_csv(WDIR.parent / "results.csv"); RESULTS.columns = [c.strip() for c in RESULTS.columns]
print("Python", platform.python_version(), "| torch", torch.__version__, "| CUDA", CUDA, "| SMOKE", SMOKE)
print("weights:", WDIR, "| results.csv rows:", len(RESULTS))
''')

nb.code(UAVDT_FETCH_CELL)

nb.code(r'''
# --- Val split: exactly the frames the training run validated on (test sequences are never loaded) ---
VAL = data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, MANIFEST["val_seqs"], MANIFEST["strides"]["val"], "val")
assert [r.key for r in VAL.images] == MANIFEST["frames"]["val"], "val frames differ from the training run: send the log"
assert not set(MANIFEST["val_seqs"]) & set(MANIFEST["test_seqs"])
if not SMOKE:
    assert len(VAL) == EXPECTED_VAL_FRAMES, len(VAL)
else:
    VAL = data.Split(VAL.dataset, VAL.name, VAL.class_names, VAL.images[::max(1, len(VAL) // 10)])
missing = [r.key for r in VAL.images if not r.path.exists()]
assert not missing, f"{len(missing)} val frames without an image, e.g. {missing[:3]}"
NOIGN = data.without_ignore(VAL)
SEQS = data.by_sequence(VAL)

''' + VAL_TABLE_CODE)

nb.code(EPOCH_INFER_CELL)

nb.code(EPOCH_SCORE_CELL)

nb.code(EPOCH_SUMMARY_CELL + r'''
print("\nRETRAIN (eval-v2 peak in the first", EARLY_PEAK, "epochs):", summary["eval_v2_peaks_early"],
      "| smoke run, ignore this" if SMOKE else "")
if ON_KAGGLE:
    !du -sh {OUT} {OUT}/*
''')

nb.save("notebooks/phase1_epochscan.ipynb")
