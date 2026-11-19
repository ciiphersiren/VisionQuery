"""Build the two UAVDT training notebooks from the same cells; only the recipe differs:
notebooks/phase1_retrain.ipynb (YOLO26m-Tile) and notebooks/phase1_retrain_yolo26s.ipynb (YOLO26s).
Both use the VisDrone optimizer settings and the same sequences, score every epoch with eval v2 and select the checkpoint on that.

Run:  python tools/build_phase1_retrain_notebook.py
"""
from nb_common import EPOCH_INFER_CELL, EPOCH_SCORE_CELL, EPOCH_SUMMARY_CELL, UAVDT_FETCH_CELL, VAL_TABLE_CODE, Notebook

INTRO_TILE = r"""
# Phase 1, step 2: retrain YOLO26m-Tile (UAVDT) with the VisDrone optimizer

The first run's eval-v2 curve peaked at epoch 1, so the retrain rule applies. Changes from the first run:
1. **Optimizer set explicitly:** `optimizer=MuSGD lr0=0.01 momentum=0.9 warmup_bias_lr=0.0`, the four values `optimizer=auto`
   gave YOLO26m-Tile (VisDrone). The rest of the command is unchanged.
2. **New val sequences:** `M0201, M0402, M0604, M0902, M1006`, stratified by altitude and weather (rule in the Config cell).
   Built from train-sequence labels only. **Test labels and images are not loaded in this notebook.**
3. **Checkpoint selected with eval v2:** every epoch is scored, and the best eval-v2 epoch becomes `yolo26m_tile_uavdt.pt`.
   Ultralytics' own `best.pt` is not used.

## How to run (about 10 h, one GPU session)
- No inputs needed. Settings: **GPU T4 x2, Internet ON**. *Save Version → Save & Run All*.
- It stops early if the data size or the optimizer line is wrong. Training takes 8 h, the scan about 1 h.
- **If Google Drive refuses the download:** upload `UAV-benchmark-M` as a private Kaggle Dataset and attach it.
- **Send back** from `/kaggle/working/phase1_retrain/`: `selection.json`, `epoch_scan_summary.json`, `epoch_scan.csv`,
  `per_sequence.csv`, `val_pick.csv`, `val_sequences.csv`, `figures/epoch_scan.png`, `runs/uavdt_y26m_tiles_musgd/args.yaml`
  and `runs/uavdt_y26m_tiles_musgd/results.csv`.
- **Keep** this notebook's output for the next step: it holds `yolo26m_tile_uavdt.pt`.
"""

INTRO_S = r"""
# Phase 1: train YOLO26s (UAVDT) with the teammate's recipe

The second UAVDT model. Same recipe as YOLO26s (VisDrone) in `prompt1.ipynb`: COCO `yolo26s.pt`, full frames + 640 px tiles with 20%
overlap, 30% visibility rule, imgsz 640, batch 32, 40 epochs, seed 42. Same frames and val sequences as YOLO26m-Tile (UAVDT).
**Test labels and images are not loaded in this notebook.** Changes from the VisDrone run, all listed in the capstone report:
1. **Optimizer set explicitly** (`optimizer=MuSGD lr0=0.01 momentum=0.9 warmup_bias_lr=0.0`). `auto` gave MuSGD on VisDrone but would
   give AdamW here, because this training set is smaller.
2. **`patience=100`** instead of 15, so that early stopping on the validator cannot end the run before every epoch is scored.
3. **Checkpoint selected with eval v2**, at `full@1024` as for YOLO26m-Tile (UAVDT).

## How to run (about 3 h, one GPU session)
- No inputs needed. Settings: **GPU T4 x2, Internet ON**. *Save Version → Save & Run All*.
- It stops early if the data size or the optimizer line is wrong.
- **If Google Drive refuses the download:** upload `UAV-benchmark-M` as a private Kaggle Dataset and attach it.
- **Send back** from `/kaggle/working/phase1_retrain_yolo26s/`: `selection.json`, `epoch_scan_summary.json`, `epoch_scan.csv`,
  `per_sequence.csv`, `val_pick.csv`, `val_sequences.csv`, `figures/epoch_scan.png`, `runs/uavdt_y26s_sf/args.yaml`
  and `runs/uavdt_y26s_sf/results.csv`.
- **Keep** this notebook's output for the next step: it holds `yolo26s_uavdt.pt`.
"""

TILE = dict(
    path="notebooks/phase1_retrain.ipynb", intro=INTRO_TILE, out="phase1_retrain", run="uavdt_y26m_tiles_musgd", final="yolo26m_tile_uavdt.pt",
    extra_config="TIME_H = 8.0\n", d2="(4045, 15328)      # counted from the labels, with M0901 at its real width; the same count gave the first run's 3,951 + 15,688",
    root="d2", builder="tiles.build_d2", label="D2", yaml="uavdt_d2.yaml",
    recipe="d2=dict(full=n_full, tiles=n_tiles, tile_sizes=list(tiles.TILE_SIZES), overlap=tiles.OVERLAP,\n"
           "                        min_vis=tiles.MIN_VIS, empty_keep=tiles.EMPTY_KEEP, grey=tiles.GREY)",
    train_comment="# Same command as the first UAVDT run and YOLO26m-Tile (VisDrone), plus the explicit optimizer.",
    train_cmd='!yolo detect train model=yolo26m.pt data={YAML} imgsz=1024 batch=16 {OPT_ARGS} device=0,1 workers=4 time={TIME_H} epochs=20 patience=100 cos_lr=True close_mosaic=3 mosaic=1.0 mixup=0.1 scale=0.5 fliplr=0.5 max_det=1000 save_period=1 project={OUT / "runs"} name={RUN_NAME} exist_ok=True plots=True seed=0 2>&1 | tee {OUT / "train.log"}')

YOLO26S = dict(
    path="notebooks/phase1_retrain_yolo26s.ipynb", intro=INTRO_S, out="phase1_retrain_yolo26s", run="uavdt_y26s_sf", final="yolo26s_uavdt.pt",
    extra_config="", d2="(4045, 8009)       # counted from the labels; the same tiler gives the teammate's logged 2,781 tiles on VisDrone val",
    root="sf", builder="tiles.build_sf", label="SF", yaml="uavdt_sf.yaml",
    recipe="sf=dict(full=n_full, tiles=n_tiles, tile=tiles.SF_TILE, overlap=tiles.SF_OVERLAP, min_vis=tiles.SF_MIN_VIS, min_side=tiles.SF_MIN_SIDE,\n"
           "                        empty_keep=tiles.SF_EMPTY_KEEP, seed=tiles.SF_SEED, grey=tiles.GREY)",
    train_comment="# YOLO26s (VisDrone) settings from prompt1.ipynb, plus the explicit optimizer, patience=100, save_period=1 and max_det=1000.",
    train_cmd='!yolo detect train model=yolo26s.pt data={YAML} imgsz=640 batch=32 {OPT_ARGS} device=0,1 workers=4 epochs=40 patience=100 cos_lr=True close_mosaic=10 mosaic=1.0 mixup=0.0 scale=0.5 fliplr=0.5 max_det=1000 save_period=1 deterministic=False project={OUT / "runs"} name={RUN_NAME} exist_ok=True plots=True seed=42 2>&1 | tee {OUT / "train.log"}')


def build(v):
    def fill(s):
        for k, x in v.items():
            s = s.replace(f"@@{k}@@", x)
        assert "@@" not in s, s
        return s

    nb = Notebook()
    nb.md(v["intro"])
    nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest gdown')

    nb.code(fill(r'''
# --- Config ----------------------------------------------------------------
import os, sys, re, json, shutil, zipfile, tarfile, subprocess, platform
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, plots, score, tiles

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
RAW = Path(os.environ.get("DK_UAVDT_RAW", "/tmp/UAVDT/raw"))       # downloads + extracted release (lost at session end)
BUILD = Path(os.environ.get("DK_UAVDT_BUILD", "/tmp/UAVDT/build"))  # tiles + yaml
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "@@out@@"
CUDA = torch.cuda.is_available()
SMOKE = not CUDA                 # no GPU: yolo26n on a few tiles stands in for training, then a few frames are scanned

TRAIN_STRIDE, VAL_STRIDE = 5, 10
@@extra_config@@RUN_NAME = "@@run@@"
# The values optimizer=auto gave YOLO26m-Tile (VisDrone). trainer.build_optimizer sets momentum and warmup_bias_lr only under
# auto, so naming MuSGD alone would train with the defaults momentum 0.937 and warmup_bias_lr 0.1.
OPTIMIZER = ["optimizer=MuSGD", "lr0=0.01", "momentum=0.9", "warmup_bias_lr=0.0"]
OPTIMIZER_LOG = "MuSGD(lr=0.01, momentum=0.9)"
# Proportional to the 30 official train sequences by weather (17/8/5 -> 3/1/1) and altitude (19/10/1 -> 3/2/0). The only
# high-altitude train sequence stays in train: holding it out would leave training with no high-altitude frames.
VAL_ALLOC = {("medium-alt", "daylight"): 2, ("low-alt", "daylight"): 1, ("medium-alt", "night"): 1, ("low-alt", "fog"): 1}
MIN_RARE_SEQS = 3                # truck and bus must each appear in >= 3 val sequences (first run: bus in 2)
EXPECTED_VAL_SEQS = ["M0201", "M0402", "M0604", "M0902", "M1006"]
EXPECTED_D2 = @@d2@@
EXPECTED_WIDTHS = {"M0901": 960} # train-pool sequences not 1024 px wide
EXPECTED_VAL_FRAMES = 367
PASS = "full@1024"               # the same pass for both UAVDT models, so that selection is comparable
SETUP = dict(passes=[PASS], method=None)
IGNORE_CONF = 0.25
EARLY_PEAK = 2
(OUT / "cache").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
print("Python", platform.python_version(), "| torch", torch.__version__, "| CUDA", CUDA, "| GPUs", torch.cuda.device_count(), "| SMOKE", SMOKE)
'''))

    nb.code(UAVDT_FETCH_CELL)

    nb.code(r'''
# --- Split: pick val from official train sequences only; test sequences are never loaded ----------
ATTRS, OFFICIAL = data.read_uavdt_attrs(ATTR)
train_all = sorted(s for s, v in OFFICIAL.items() if v == "train")
test_seqs = sorted(s for s, v in OFFICIAL.items() if v == "test")
POOL = data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, train_all, VAL_STRIDE, "pool")
TABLE = data.sequence_table(POOL)
val_seqs, distance = data.pick_val_sequences(TABLE, VAL_ALLOC, min_rare_seqs=MIN_RARE_SEQS)
train_seqs = [s for s in train_all if s not in val_seqs]
if ON_KAGGLE:
    assert len(train_all) == 30 and len(test_seqs) == 20, (len(train_all), len(test_seqs))
    assert val_seqs == EXPECTED_VAL_SEQS, f"picked {val_seqs}: stop here and send the log"
assert not set(train_seqs) & set(test_seqs) and not set(val_seqs) & set(test_seqs) and not set(train_seqs) & set(val_seqs)

SPL = {"train": data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, train_seqs, TRAIN_STRIDE, "train"),
       "val": data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, val_seqs, VAL_STRIDE, "val")}
for s, sp in SPL.items():
    bad = [r.key for r in sp.images if not r.path.exists()]
    assert not bad, f"{s}: {len(bad)} annotated frames without an image, e.g. {bad[:3]}"
SIZES = {r.sequence: (r.width, r.height) for r in POOL.images}
other_widths = {s: w for s, (w, h) in SIZES.items() if w != 1024}
print("frame sizes:", sorted(set(SIZES.values())), "| sequences not 1024 px wide:", other_widths)
if ON_KAGGLE:
    assert other_widths == EXPECTED_WIDTHS and {h for _, h in SIZES.values()} == {540}, f"unexpected frame sizes {SIZES}: stop here and send the log"
pick_tab = pd.DataFrame(TABLE).T.assign(role=lambda d: np.where(d.index.isin(val_seqs), "val", "train"),
                                        width=pd.Series({s: w for s, (w, h) in SIZES.items()}))
pick_tab.to_csv(OUT / "val_pick.csv")
print("validation sequences:", val_seqs, "| distance from the train pool:", round(distance, 4))
print(pick_tab.loc[val_seqs, ["altitude", "weather", "images", "objects", "tiny", "share_truck", "share_bus"]])
STATS = pd.DataFrame({"pool": data.split_stats(POOL), **{s: data.split_stats(sp) for s, sp in SPL.items()}}).T
STATS.to_csv(OUT / "uavdt_split_stats.csv")
print(STATS[["images", "sequences", "objects", "tiny", "small", "share_car", "share_truck", "share_bus"]].round(3))
''')

    nb.code(fill(r'''
# --- Build training data: @@label@@ recipe on train frames; val frames as plain images + labels ---
shutil.rmtree(BUILD, ignore_errors=True)
n_full, n_tiles = @@builder@@(SPL["train"], BUILD / "@@root@@")
print(f"@@label@@ train: {n_full} full frames + {n_tiles} tiles")
if ON_KAGGLE:                    # stop before a long run on the wrong set
    assert (n_full, n_tiles) == EXPECTED_D2, "unexpected training-set size: stop here and send the log"
    assert len(SPL["val"]) == EXPECTED_VAL_FRAMES, len(SPL["val"])
vd = BUILD / "val"; (vd / "images").mkdir(parents=True); (vd / "labels").mkdir(parents=True)
for r in SPL["val"].images:
    try:
        os.symlink(r.path, vd / "images" / f"{r.key}.jpg")
    except OSError:                                  # no symlink rights (Windows): copy instead
        shutil.copy(r.path, vd / "images" / f"{r.key}.jpg")
    (vd / "labels" / f"{r.key}.txt").write_text("\n".join(tiles.yolo_lines(r.boxes, r.width, r.height)))
YAML = tiles.write_yaml(BUILD / "@@yaml@@", BUILD, ["@@root@@/full/images", "@@root@@/tiles/images"], "val/images", data.UAVDT_CLASSES)
manifest = dict(dataset="UAVDT-M (official Google Drive release)", classes=data.UAVDT_CLASSES,
                train_seqs=train_seqs, val_seqs=val_seqs, test_seqs=test_seqs, val_alloc={"/".join(k): v for k, v in VAL_ALLOC.items()},
                strides=dict(train=TRAIN_STRIDE, val=VAL_STRIDE), frames={s: [r.key for r in sp.images] for s, sp in SPL.items()},
                @@recipe@@)
(OUT / "uavdt_split.json").write_text(json.dumps(manifest, indent=1))
'''))

    nb.code(fill(r'''
# --- Short check (CPU or GPU): yolo26n, 2 epochs, a handful of tiles, same optimizer flags -----------
sm = BUILD / "smoke"; shutil.rmtree(sm, ignore_errors=True)
for sub in ("train", "val"):
    (sm / sub / "images").mkdir(parents=True); (sm / sub / "labels").mkdir(parents=True)
src = sorted((BUILD / "@@root@@/tiles/labels").glob("*.txt"))[:16]
for i, lab in enumerate(src):
    sub = "val" if i < 4 else "train"
    shutil.copy(lab, sm / sub / "labels" / lab.name)
    shutil.copy(BUILD / "@@root@@/tiles/images" / f"{lab.stem}.jpg", sm / sub / "images" / f"{lab.stem}.jpg")
smoke_yaml = tiles.write_yaml(sm / "smoke.yaml", sm, ["train/images"], "val/images", data.UAVDT_CLASSES)
YOLO_CLI = shutil.which("yolo") or str(next(Path(sys.executable).parent.glob("yolo*")))
r = subprocess.run([YOLO_CLI, "detect", "train", "model=yolo26n.pt", f"data={smoke_yaml}", "imgsz=320", "epochs=2", "batch=4", "save_period=1",
                    *OPTIMIZER, f"device={'0' if CUDA else 'cpu'}", "workers=0", "max_det=1000", "plots=False",
                    f"project={sm / 'runs'}", "name=smoke", "exist_ok=True"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0 and (sm / "runs/smoke/weights/last.pt").exists(), "short check failed: send the log"
assert OPTIMIZER_LOG in r.stdout + r.stderr, f"optimizer line is not {OPTIMIZER_LOG}: stop here and send the log"
print("short check passed:", OPTIMIZER_LOG)
'''))

    nb.md("## Training (GPU only)")
    nb.code(fill(r'''
@@train_comment@@
if not SMOKE:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)          # SAHI may have set it; it breaks two-GPU training
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    OPT_ARGS = " ".join(OPTIMIZER)
    @@train_cmd@@
    WDIR = OUT / "runs" / RUN_NAME / "weights"
    assert (WDIR / "best.pt").exists(), "training produced no weights"
    assert OPTIMIZER_LOG in (OUT / "train.log").read_text(errors="replace"), f"training log has no {OPTIMIZER_LOG}"
else:
    print("no GPU: training skipped; the scan below uses the short check's checkpoints")
    WDIR = sm / "runs/smoke/weights"
RUN_ARGS = yaml.safe_load((WDIR.parent / "args.yaml").read_text())
for k, v in dict(optimizer="MuSGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0).items():
    assert RUN_ARGS[k] == v, f"args.yaml {k} = {RUN_ARGS[k]}, expected {v}"
RESULTS = pd.read_csv(WDIR.parent / "results.csv"); RESULTS.columns = [c.strip() for c in RESULTS.columns]
print("weights:", WDIR, "| results.csv rows:", len(RESULTS))
'''))

    nb.md("## Score every epoch on val and select the checkpoint with eval v2")
    nb.code(r'''
# --- Val split for scoring --------------------------------------------------
VAL = SPL["val"]
if SMOKE:
    VAL = data.Split(VAL.dataset, VAL.name, VAL.class_names, VAL.images[::max(1, len(VAL) // 10)])
NOIGN = data.without_ignore(VAL)
SEQS = data.by_sequence(VAL)

''' + VAL_TABLE_CODE)
    nb.code(EPOCH_INFER_CELL)
    nb.code(EPOCH_SCORE_CELL)
    nb.code(EPOCH_SUMMARY_CELL)

    nb.code(fill(r'''
# --- Selection: highest eval-v2 AP on val. The stripped copy must score the same as the checkpoint it came from ---
from ultralytics.utils.torch_utils import strip_optimizer
top = SCAN.loc[SCAN["ev2_AP"].idxmax()]
SELECTED = int(top["epoch"])
final = OUT / "@@final@@"
shutil.copy(WDIR / f"epoch{SELECTED - 1}.pt", final)
strip_optimizer(str(final))
LOGGER.setLevel(logging.ERROR)
model = YOLO(str(final))
infer.run_and_cache(model, VAL, [PASS], OUT / "cache" / "selected", meta=dict(epoch=SELECTED, checkpoint=final.name, ultralytics=UL_VERSION), **device)
del model
LOGGER.setLevel(logging.INFO)
m_final, _, _ = score.evaluate(VAL, infer.apply_setup(infer.load_cache(OUT / "cache" / "selected")[0], VAL, SETUP)[0])
assert abs(m_final["AP"] - top["ev2_AP"]) < 0.005, (m_final["AP"], top["ev2_AP"])
selection = dict(smoke=SMOKE, rule="highest eval-v2 AP on val, full@1024", selected_epoch=SELECTED, selected_checkpoint=f"epoch{SELECTED - 1}.pt",
                 eval_v2=float(top["ev2_AP"]), eval_v2_AP50=float(top["ev2_AP50"]), per_class_AP={c: float(top["AP_" + c]) for c in VAL.class_names},
                 stripped_copy_eval_v2=float(m_final["AP"]), ultralytics_best_pt_epochs=BEST_ROW, peaks_in_first_epochs=bool(SELECTED <= EARLY_PEAK),
                 val_seqs=val_seqs, optimizer={k: RUN_ARGS[k] for k in ("optimizer", "lr0", "momentum", "warmup_bias_lr")}, epochs_trained=len(RESULTS))
(OUT / "selection.json").write_text(json.dumps(selection, indent=1))
print(json.dumps(selection, indent=1))
shutil.rmtree(BUILD / "smoke", ignore_errors=True)
if ON_KAGGLE:
    !du -sh {OUT} {OUT}/*
'''))
    nb.save(v["path"])


build(TILE)
build(YOLO26S)
