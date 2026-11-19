"""Build notebooks/phase1_train.ipynb: UAVDT download + conversion + D2 tiles + YOLO26m-Tile (UAVDT) training.

Run:  python tools/build_phase1_train_notebook.py
"""
from nb_common import UAVDT_FETCH_CELL, Notebook

nb = Notebook()
nb.md(r"""
# Phase 1a: train YOLO26m-Tile (UAVDT)

Second training dataset (supervisor's request) and the source of the cross-dataset shift for Phase 4.
Same recipe as YOLO26m-Tile (VisDrone): COCO `yolo26m.pt`, ignore regions painted gray, 640/960 px tiles with 25% overlap,
60% visibility rule, 10% empty tiles, imgsz 1024, batch 16, 2x T4, `time=` budget, `save_period=1`.

**Data (UAVDT-M, official release, Google Drive):** 50 sequences, official split 30 train / 20 test (from the
`M_attr/train|test` folders). Train frames: 25 train sequences, every 5th frame. Validation (picks `best.pt`):
5 train sequences held out (seed 0), every 10th frame. **Test sequences are not used in this notebook.**

## How to run (second Kaggle account)
- No inputs needed. Settings: **Internet ON**.
- **Step 1, CPU check (about 20–30 min):** Accelerator **None** → *Save Version → Save & Run All*. Downloads UAVDT (~6 GB),
  converts it, builds tiles for the full training set, draws a ground-truth check image, and trains `yolo26n` for 1 epoch
  on a handful of tiles. The real training cell is skipped without a GPU. Send back the log and
  `phase1_train/figures/uavdt_gt_check.png`.
- **Step 2, training (about 9 h):** Accelerator **GPU T4 x2** → *Save & Run All*.
- **If Google Drive refuses the download** ("Too many users have viewed or downloaded this file"): download
  `UAV-benchmark-M` once on a PC from the link in the Config cell, upload it as a private Kaggle Dataset, attach it, and re-run.
  The notebook uses an attached copy automatically.
- **Output:** `/kaggle/working/phase1_train/`: `yolo26m_tile_uavdt.pt`, the run folder (`results.csv`, `args.yaml`, per-epoch
  checkpoints), `uavdt_split.json` (exact sequences and frames used) and dataset statistics.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest gdown')

nb.code(r'''
# --- Config ----------------------------------------------------------------
import os, sys, json, shutil, zipfile, tarfile, subprocess, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, tiles

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
RAW = Path(os.environ.get("DK_UAVDT_RAW", "/tmp/UAVDT/raw"))       # downloads + extracted release (lost at session end)
BUILD = Path(os.environ.get("DK_UAVDT_BUILD", "/tmp/UAVDT/build"))  # tiles + yaml
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "phase1_train"
CUDA = torch.cuda.is_available()
RUN_TRAIN = CUDA and os.environ.get("DK_SKIP_TRAIN") != "1"

TRAIN_STRIDE, VAL_STRIDE, TEST_STRIDE = 5, 10, 10
N_VAL_SEQS, SEED = 5, 0
TIME_H = 8.0                     # Phase 1: <= 8 h per session
RUN_NAME = "uavdt_y26m_tiles"
print("Python", platform.python_version(), "| torch", torch.__version__, "| CUDA", CUDA, "| GPUs", torch.cuda.device_count())
''')

nb.code(UAVDT_FETCH_CELL)

nb.code(r'''
# --- Split: official train/test; 5 train sequences held out for validation ------
ATTRS, OFFICIAL = data.read_uavdt_attrs(ATTR)
train_all = sorted(s for s, v in OFFICIAL.items() if v == "train")
test_seqs = sorted(s for s, v in OFFICIAL.items() if v == "test")
assert len(train_all) == 30 and len(test_seqs) == 20, (len(train_all), len(test_seqs))
val_seqs = sorted(np.random.default_rng(SEED).choice(train_all, N_VAL_SEQS, replace=False).tolist())
train_seqs = [s for s in train_all if s not in val_seqs]
assert not set(train_seqs) & set(test_seqs) and not set(val_seqs) & set(test_seqs)
missing = [s for s in train_all + test_seqs if not data.find_uavdt_frames(IMG_ROOT, s)]
assert not missing, f"sequences without images: {missing}"

SPL = {"train": data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, train_seqs, TRAIN_STRIDE, "train"),
       "val": data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, val_seqs, VAL_STRIDE, "val"),
       "test": data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, test_seqs, TEST_STRIDE, "test")}
for s, sp in SPL.items():                           # every frame must exist on disk and match the annotation size
    bad = [r.key for r in sp.images if not r.path.exists()]
    assert not bad, f"{s}: {len(bad)} annotated frames without an image, e.g. {bad[:3]}"
STATS = pd.DataFrame({s: {**data.split_stats(sp), "stride": st} for (s, sp), st in
                      zip(SPL.items(), (TRAIN_STRIDE, VAL_STRIDE, TEST_STRIDE))}).T
print("validation sequences:", val_seqs)
print(STATS[["images", "sequences", "objects", "objects_per_image", "tiny", "small", "medium_plus",
             "share_car", "share_truck", "share_bus", "stride"]].round(3))
attr_tab = pd.DataFrame(ATTRS).T.assign(split=pd.Series(OFFICIAL)).assign(
    role=lambda d: np.where(d.index.isin(val_seqs), "val (held out from train)", d["split"]))
print(attr_tab.groupby(["role", "altitude"]).size().unstack(fill_value=0))
''')

nb.code(r'''
# --- Visual check: ground truth drawn on real frames (boxes must sit on vehicles) ---
import cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
(OUT / "figures").mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
picks = [SPL["train"].images[0], SPL["train"].images[len(SPL["train"]) // 2], SPL["val"].images[0], SPL["test"].images[-1]]
for ax, r in zip(axes.ravel(), picks):
    im = cv2.imread(str(r.path))[:, :, ::-1].copy()
    for c, x1, y1, x2, y2, *_ in r.boxes:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), [(0, 255, 0), (255, 160, 0), (255, 0, 255)][int(c)], 2)
    for x1, y1, x2, y2 in r.ignore:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    ax.imshow(im); ax.set_title(f"{r.key} ({r.attrs.get('altitude')}, {r.attrs.get('weather')}): "
                                f"{len(r.boxes)} boxes, {len(r.ignore)} ignore (red)", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "figures" / "uavdt_gt_check.png", dpi=110); plt.close()
print("saved", OUT / "figures" / "uavdt_gt_check.png", "(green car, orange truck, magenta bus)")
''')

nb.code(r'''
# --- Build training data: D2 recipe on train frames; val frames as plain images + labels ---
shutil.rmtree(BUILD, ignore_errors=True)
n_full, n_tiles = tiles.build_d2(SPL["train"], BUILD / "d2")
print(f"D2 train: {n_full} full frames + {n_tiles} tiles")
if CUDA and TRAIN_STRIDE == 5:   # expected from the official GT (annotated frames only, 1024 x 540 frames); stop before an 8 h run on the wrong set
    assert (n_full, n_tiles) == (3951, 15688), "unexpected D2 size: stop here and send the log"
vd = BUILD / "val"; (vd / "images").mkdir(parents=True); (vd / "labels").mkdir(parents=True)
for r in SPL["val"].images:
    try:
        os.symlink(r.path, vd / "images" / f"{r.key}.jpg")
    except OSError:                                  # no symlink rights (Windows): copy instead
        shutil.copy(r.path, vd / "images" / f"{r.key}.jpg")
    (vd / "labels" / f"{r.key}.txt").write_text("\n".join(tiles.yolo_lines(r.boxes, r.width, r.height)))
YAML = tiles.write_yaml(BUILD / "uavdt_d2.yaml", BUILD, ["d2/full/images", "d2/tiles/images"], "val/images", data.UAVDT_CLASSES)
print(f"D2 (UAVDT): {n_full} masked full frames + {n_tiles} tiles = {n_full + n_tiles} training images; "
      f"val {len(SPL['val'])} frames\n" + YAML.read_text())

manifest = dict(dataset="UAVDT-M (official Google Drive release)", classes=data.UAVDT_CLASSES,
                train_seqs=train_seqs, val_seqs=val_seqs, test_seqs=test_seqs, seed=SEED,
                strides=dict(train=TRAIN_STRIDE, val=VAL_STRIDE, test=TEST_STRIDE),
                frames={s: [r.key for r in sp.images] for s, sp in SPL.items()},
                d2=dict(full=n_full, tiles=n_tiles, tile_sizes=list(tiles.TILE_SIZES), overlap=tiles.OVERLAP,
                        min_vis=tiles.MIN_VIS, empty_keep=tiles.EMPTY_KEEP, grey=tiles.GREY))
(OUT / "uavdt_split.json").write_text(json.dumps(manifest, indent=1))
STATS.assign(d2_training_images=[n_full + n_tiles, None, None]).to_csv(OUT / "uavdt_split_stats.csv")
attr_tab.to_csv(OUT / "uavdt_sequence_attributes.csv")
!du -sh {BUILD}
''')

nb.code(r'''
# --- Smoke training (CPU or GPU): yolo26n, 1 epoch, a handful of tiles --------
sm = BUILD / "smoke"; shutil.rmtree(sm, ignore_errors=True)
for sub in ("train", "val"):
    (sm / sub / "images").mkdir(parents=True); (sm / sub / "labels").mkdir(parents=True)
src = sorted((BUILD / "d2/tiles/labels").glob("*.txt"))[:16]
for i, lab in enumerate(src):
    sub = "val" if i < 4 else "train"
    shutil.copy(lab, sm / sub / "labels" / lab.name)
    shutil.copy(BUILD / "d2/tiles/images" / f"{lab.stem}.jpg", sm / sub / "images" / f"{lab.stem}.jpg")
smoke_yaml = tiles.write_yaml(sm / "smoke.yaml", sm, ["train/images"], "val/images", data.UAVDT_CLASSES)
YOLO_CLI = shutil.which("yolo") or str(next(Path(sys.executable).parent.glob("yolo*")))
r = subprocess.run([YOLO_CLI, "detect", "train", "model=yolo26n.pt", f"data={smoke_yaml}", "imgsz=320", "epochs=1", "batch=4",
                    f"device={'0' if CUDA else 'cpu'}", "workers=0", "max_det=1000", "plots=False",
                    f"project={sm / 'runs'}", "name=smoke", "exist_ok=True"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0 and (sm / "runs/smoke/weights/last.pt").exists(), "smoke training failed: send the log"
print("smoke training passed")
''')

nb.md("## Training (GPU only)")
nb.code(r'''
# Same command as YOLO26m-Tile (VisDrone) in visdrone-training (Cell 5), with time=8 and save_period=1.
# time= overrides epochs: Ultralytics sizes the epoch count and LR schedule to finish in TIME_H hours.
# Resume (only if a session dies): attach that version's output; RESUME finds runs/<RUN_NAME>/weights/last.pt.
RESUME = next(iter(sorted(INPUT.rglob(f"{RUN_NAME}/weights/last.pt"))), None) if INPUT.exists() else None
if RUN_TRAIN:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)          # SAHI may have set it; it breaks two-GPU training
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if RESUME:
        local = OUT / "runs" / RUN_NAME / "weights"; local.mkdir(parents=True, exist_ok=True)
        shutil.copytree(RESUME.parent.parent, OUT / "runs" / RUN_NAME, dirs_exist_ok=True)
        print("resuming from", RESUME)
        !yolo detect train resume model={local / "last.pt"}
    else:
        !yolo detect train model=yolo26m.pt data={YAML} imgsz=1024 batch=16 \
            device=0,1 workers=4 time={TIME_H} epochs=20 patience=100 cos_lr=True close_mosaic=3 \
            mosaic=1.0 mixup=0.1 scale=0.5 fliplr=0.5 max_det=1000 save_period=1 \
            project={OUT / "runs"} name={RUN_NAME} exist_ok=True plots=True seed=0
    best = OUT / "runs" / RUN_NAME / "weights" / "best.pt"
    assert best.exists(), "training produced no best.pt"
    shutil.copy(best, OUT / "yolo26m_tile_uavdt.pt")
    !ls -lh {OUT} {OUT}/runs/{RUN_NAME}/weights | head -20
else:
    print("no GPU: training skipped (data build and smoke training above checked the pipeline)")
''')

nb.code(r'''
# --- Clean-up: keep weights + run files, drop the per-epoch checkpoints only if space is short ---
shutil.rmtree(BUILD / "smoke", ignore_errors=True)
if ON_KAGGLE:
    !du -sh /kaggle/working/* 2>/dev/null
print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=1))
''')

nb.save("notebooks/phase1_train.ipynb")
