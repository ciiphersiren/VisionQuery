"""Build the two notebooks for the extra-seed runs of YOLO26m-Tile (VisDrone):
  notebooks/seeds_train.ipynb  - one training run with SEED = 1 or 2 (run it twice, once per Kaggle account)
  notebooks/seeds_eval.ipynb   - scores the original model (seed 0) and both new seeds with eval v2 on val and test-dev

Pre-registered before any run (10 Nov 2026): same data (D2, identical tiles), same command as the original run except the seed and the
explicit optimizer that `auto` chose; checkpoint = Ultralytics best.pt, as for the original; each test-dev scored once; reported as mean and
standard deviation over the three runs, with no run dropped.

Run:  python tools/build_seeds_notebooks.py
"""
from nb_common import Notebook

# ============================================================================ training notebook
nb = Notebook()
nb.md(r"""
# Extra seeds: train YOLO26m-Tile (VisDrone) again with another seed

Same data and command as the original YOLO26m-Tile run, except the **seed** and the optimizer written out explicitly (MuSGD, lr 0.01, momentum 0.9,
warm-up bias lr 0.0: the values `optimizer=auto` gave the original run). The tiles are identical, because the tiler is seeded per image.
The checkpoint is Ultralytics' `best.pt`, as for the original run. Fixed before running; see the capstone report.

## How to run (about 11.5 h, one GPU session)
1. In the Config cell below, set `SEED = 1` for the first run and `SEED = 2` for the second (one per Kaggle account, at the same time).
2. Settings: **GPU T4 x2, Internet ON**. No inputs needed. *Save Version → Save & Run All*.
3. It stops early with a message if the training set is not the expected size or the optimizer line is wrong.
4. When it finishes, the **Output** tab holds `seeds_train_s{SEED}/yolo26m_tile_visdrone_seed{SEED}.pt` and `send_back.zip`.
   Keep the weights for `seeds_eval.ipynb`; send `send_back.zip`.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest')

nb.code(r'''
# --- Config ----------------------------------------------------------------
SEED = 1                         # <<< set 1 for the first run, 2 for the second. The original run is seed 0.

import os, sys, json, shutil, zipfile, subprocess, platform, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, tiles

assert SEED in (1, 2), "SEED must be 1 or 2"
ON_KAGGLE = Path("/kaggle").exists()
DATA = Path(os.environ.get("DK_DATA", "/tmp/VisDrone"))            # images in /tmp (lost at session end)
BUILD = Path(os.environ.get("DK_BUILD", "/tmp/VisDrone_D2"))
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / f"seeds_train_s{SEED}"
CUDA = torch.cuda.is_available()
SMOKE = not CUDA                 # no GPU: a few val images stand in for the training set and yolo26n for the model
TIME_H = 10.5                    # the original run's budget
RUN_NAME = f"visdrone_y26m_tiles_seed{SEED}"
OPTIMIZER = ["optimizer=MuSGD", "lr0=0.01", "momentum=0.9", "warmup_bias_lr=0.0"]
OPTIMIZER_LOG = "MuSGD(lr=0.01, momentum=0.9)"
EXPECTED_D2 = (6471, 63745)
OUT.mkdir(parents=True, exist_ok=True)
print("SEED", SEED, "| Python", platform.python_version(), "| torch", torch.__version__, "| GPUs", torch.cuda.device_count(), "| SMOKE", SMOKE)
''')

nb.code(r'''
# --- Data: VisDrone train (D2 tiles) and val (validation during training, as in the original run) ---
t0 = time.time()
if SMOKE:
    src = data.load_visdrone(data.download_visdrone(DATA, "val"), "val", training_rules=True)
    train = data.Split(src.dataset, "train", src.class_names, src.images[:6])
    val = data.Split(src.dataset, "val", src.class_names, src.images[6:10])
else:
    train = data.load_visdrone(data.download_visdrone(DATA, "train"), "train", training_rules=True)
    val = data.load_visdrone(data.download_visdrone(DATA, "val"), "val", training_rules=True)
shutil.rmtree(BUILD, ignore_errors=True)
n_full, n_tiles = tiles.build_d2(train, BUILD / "d2", workers=0 if SMOKE else None)   # no process pool in the CPU check (Windows)
print(f"D2: {n_full} full frames + {n_tiles} tiles ({(time.time() - t0) / 60:.1f} min)")
if not SMOKE:
    assert (n_full, n_tiles) == EXPECTED_D2, f"training set is {(n_full, n_tiles)}, expected {EXPECTED_D2}: stop here and send the log"
vd = BUILD / "val"; (vd / "images").mkdir(parents=True); (vd / "labels").mkdir(parents=True)
for r in val.images:
    try:
        os.symlink(r.path, vd / "images" / f"{r.key}.jpg")
    except OSError:
        shutil.copy(r.path, vd / "images" / f"{r.key}.jpg")
    (vd / "labels" / f"{r.key}.txt").write_text("\n".join(tiles.yolo_lines(r.boxes, r.width, r.height)))
YAML = tiles.write_yaml(BUILD / "visdrone_d2.yaml", BUILD, ["d2/full/images", "d2/tiles/images"], "val/images", data.VISDRONE_CLASSES)
(OUT / "d2_manifest.json").write_text(json.dumps(dict(seed=SEED, full=n_full, tiles=n_tiles, tile_sizes=list(tiles.TILE_SIZES), overlap=tiles.OVERLAP,
                                                      min_vis=tiles.MIN_VIS, empty_keep=tiles.EMPTY_KEEP, grey=tiles.GREY), indent=1))
''')

nb.code(r'''
# --- Short check (CPU or GPU): yolo26n, 1 epoch, a few tiles, same optimizer flags ----------------
sm = BUILD / "smoke"; shutil.rmtree(sm, ignore_errors=True)
for sub in ("train", "val"):
    (sm / sub / "images").mkdir(parents=True); (sm / sub / "labels").mkdir(parents=True)
for i, lab in enumerate(sorted((BUILD / "d2/tiles/labels").glob("*.txt"))[:12]):
    sub = "val" if i < 3 else "train"
    shutil.copy(lab, sm / sub / "labels" / lab.name)
    shutil.copy(BUILD / "d2/tiles/images" / f"{lab.stem}.jpg", sm / sub / "images" / f"{lab.stem}.jpg")
smoke_yaml = tiles.write_yaml(sm / "smoke.yaml", sm, ["train/images"], "val/images", data.VISDRONE_CLASSES)
YOLO_CLI = shutil.which("yolo") or str(next(Path(sys.executable).parent.glob("yolo*")))
r = subprocess.run([YOLO_CLI, "detect", "train", "model=yolo26n.pt", f"data={smoke_yaml}", "imgsz=320", "epochs=1", "batch=4", *OPTIMIZER,
                    f"device={'0' if CUDA else 'cpu'}", "workers=0", "max_det=1000", "plots=False", f"seed={SEED}",
                    f"project={sm / 'runs'}", "name=smoke", "exist_ok=True"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-1200:], r.stderr[-1200:])
assert r.returncode == 0 and (sm / "runs/smoke/weights/best.pt").exists(), "short check failed: send the log"
assert OPTIMIZER_LOG in r.stdout + r.stderr, f"optimizer line is not {OPTIMIZER_LOG}: stop here and send the log"
print("short check passed:", OPTIMIZER_LOG)
''')

nb.md("## Training (GPU only, about 10.5 h)")
nb.code(r'''
# Same command as the original YOLO26m-Tile run (imgsz 1024, batch 16, 10.5 h, mosaic off for the last 3 epochs, mixup 0.1), with the seed.
if not SMOKE:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)          # SAHI may have set it; it breaks two-GPU training
    OPT_ARGS = " ".join(OPTIMIZER)
    !yolo detect train model=yolo26m.pt data={YAML} imgsz=1024 batch=16 {OPT_ARGS} device=0,1 workers=4 time={TIME_H} epochs=20 patience=100 cos_lr=True close_mosaic=3 mosaic=1.0 mixup=0.1 scale=0.5 fliplr=0.5 max_det=1000 save_period=1 project={OUT / "runs"} name={RUN_NAME} exist_ok=True plots=True seed={SEED} 2>&1 | tee {OUT / "train.log"}
    WDIR = OUT / "runs" / RUN_NAME / "weights"
    assert (WDIR / "best.pt").exists(), "training produced no weights: send train.log"
    assert OPTIMIZER_LOG in (OUT / "train.log").read_text(errors="replace"), f"training log has no {OPTIMIZER_LOG}"
else:
    print("no GPU: training skipped; the short check's weights stand in")
    WDIR = sm / "runs/smoke/weights"
ARGS = yaml.safe_load((WDIR.parent / "args.yaml").read_text())
assert int(ARGS["seed"]) == SEED, f"args.yaml seed = {ARGS['seed']}"
for k, v in dict(optimizer="MuSGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0).items():
    assert ARGS[k] == v, f"args.yaml {k} = {ARGS[k]}, expected {v}"
''')

nb.code(r'''
# --- Keep best.pt (the original run's rule), stripped of optimizer state; bundle what comes back ---
from ultralytics.utils.torch_utils import strip_optimizer
final = OUT / f"yolo26m_tile_visdrone_seed{SEED}.pt"
shutil.copy(WDIR / "best.pt", final); strip_optimizer(str(final))
res = pd.read_csv(WDIR.parent / "results.csv"); res.columns = [c.strip() for c in res.columns]
fit = res["metrics/mAP50(B)"] * 0.1 + res["metrics/mAP50-95(B)"] * 0.9 if "metrics/mAP50-95(B)" in res else res.iloc[:, 0]
info = dict(seed=SEED, smoke=SMOKE, epochs_trained=len(res), best_epoch_ultralytics=int(fit.idxmax()) + 1, time_h=TIME_H,
            optimizer={k: ARGS[k] for k in ("optimizer", "lr0", "momentum", "warmup_bias_lr")}, ultralytics=__import__("ultralytics").__version__)
(OUT / "train_info.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))
with zipfile.ZipFile(OUT / "send_back.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in [OUT / "train_info.json", OUT / "d2_manifest.json", WDIR.parent / "results.csv", WDIR.parent / "args.yaml", OUT / "train.log"]:
        if f.exists():
            z.write(f, f.name)
shutil.rmtree(BUILD / "smoke", ignore_errors=True)
print("done:", final, f"{final.stat().st_size / 1e6:.1f} MB")
''')
nb.save("notebooks/seeds_train.ipynb")

# ============================================================================ evaluation notebook
nb = Notebook()
nb.md(r"""
# Extra seeds: score YOLO26m-Tile seeds 0, 1 and 2 with eval v2

Scores the original model (seed 0) and the two new seeds on VisDrone val and test-dev, whole image at 1536 and the best sliced setup, with the
paper's scorer. Seed 0 must reproduce its published numbers (val 0.667, test-dev 0.540 AP50, best sliced); otherwise the notebook stops.

## How to run (about 1.5 h)
1. **Inputs:** attach (a) the weights Dataset used by the other notebooks (it holds `yolo26m_tile_visdrone.pt`) and (b) the outputs of both
   `seeds_train` runs (`yolo26m_tile_visdrone_seed1.pt`, `..._seed2.pt`). From the other account, share its notebook output or upload the
   `.pt` file as a private Dataset first.
2. Settings: **GPU T4 x2, Internet ON**. *Save Version → Save & Run All*.
3. **Send back** `/kaggle/working/seeds_eval/send_back.zip`.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest')

nb.code(r'''
# --- Config ----------------------------------------------------------------
import os, sys, gc, json, zipfile, logging, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, score
logging.getLogger("sahi").setLevel(logging.WARNING)

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
DATA = Path(os.environ.get("DK_DATA", "/tmp/VisDrone"))
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "seeds_eval"
CUDA = torch.cuda.is_available()
SMOKE = not CUDA
DEVICE, HALF = (0, True) if CUDA else ("cpu", False)
PUBLISHED = {("val", "best_sliced"): 0.667, ("test-dev", "best_sliced"): 0.540}   # seed 0
TOL = 0.003
(OUT / "cache").mkdir(parents=True, exist_ok=True)

def find(name):
    hits = sorted(INPUT.rglob(name)) if INPUT.exists() else []
    return hits[0] if hits else None

WEIGHTS = {0: find("yolo26m_tile_visdrone.pt"), 1: find("yolo26m_tile_visdrone_seed1.pt"), 2: find("yolo26m_tile_visdrone_seed2.pt")}
print({k: str(v) for k, v in WEIGHTS.items()})
missing = [k for k, v in WEIGHTS.items() if v is None]
if ON_KAGGLE:
    assert not missing, f"weights for seed(s) {missing} not attached: see the instructions at the top"
WEIGHTS = {k: v for k, v in WEIGHTS.items() if v is not None}
''')

nb.code(r'''
# --- Data: VisDrone val + test-dev (official annotations, scoring rules) ----
SPLITS = {s: data.load_visdrone(data.download_visdrone(DATA, s), s) for s in (("val",) if SMOKE else ("val", "test-dev"))}
if SMOKE:
    SPLITS = {s: data.Split(sp.dataset, sp.name, sp.class_names, sp.images[:3]) for s, sp in SPLITS.items()}
print({s: len(sp) for s, sp in SPLITS.items()})
''')

nb.code(r'''
# --- Run every pass once per model and split, then score both setups -------
from ultralytics import YOLO
PASSES = infer.SETUPS["best_sliced"]["passes"]            # s512@1280 + full@1536; whole1536 reuses full@1536
rows, pcs = [], []
for seed, w in WEIGHTS.items():
    m = YOLO(str(w))
    names = [m.names[i] for i in range(len(m.names))]
    assert names == dronekit.VISDRONE_CLASSES, f"seed {seed}: classes {names}"
    for sname, sp in SPLITS.items():
        path = OUT / "cache" / f"seed{seed}__{sname}"
        infer.run_and_cache(m, sp, PASSES, path, cuda=CUDA, device=DEVICE, half=HALF, meta=dict(seed=seed, weights=str(w)))
        raw, meta = infer.load_cache(path)
        for setup in ("whole1536", "best_sliced"):
            dets, merge_ms = infer.apply_setup(raw, sp, setup)
            met, pc, _ = score.evaluate(sp, dets)
            rows.append(dict(seed=seed, split=sname, setup=setup, n_images=len(sp), **met, ms_per_img=infer.setup_ms(meta, setup, merge_ms)))
            pcs += [dict(seed=seed, split=sname, setup=setup, cls=c, AP50=v) for c, v in pc.items()]
            print(f"seed {seed} {sname:9s} {setup:12s} AP50 {met['AP50']:.3f}  AP {met['AP']:.3f}")
    del m; gc.collect()
    if CUDA: torch.cuda.empty_cache()
R = pd.DataFrame(rows); R.to_csv(OUT / "seeds_results.csv", index=False)
pd.DataFrame(pcs).to_csv(OUT / "seeds_per_class.csv", index=False)
''')

nb.code(r'''
# --- Seed 0 must reproduce the published numbers; then mean and spread over the seeds ---
if not SMOKE:
    for (sname, setup), ref in PUBLISHED.items():
        got = R[(R.seed == 0) & (R.split == sname) & (R.setup == setup)].AP50.iloc[0]
        assert abs(got - ref) <= TOL, f"seed 0 {sname} {setup}: AP50 {got:.4f}, published {ref}: stop here and send the log"
cols = ["AP50", "AP", "AP50_tiny", "AP50_small", "AP50_medium+"]
cols = [c for c in cols if c in R]
S = R.groupby(["split", "setup"])[cols].agg(["mean", "std", "min", "max"]).round(4)
S.columns = ["_".join(c) for c in S.columns]
S["n_seeds"] = R.groupby(["split", "setup"]).seed.nunique()
S.to_csv(OUT / "seeds_summary.csv"); print(S.to_string())
(OUT / "run_meta.json").write_text(json.dumps(dict(seeds=sorted(WEIGHTS), weights={k: str(v) for k, v in WEIGHTS.items()}, smoke=SMOKE,
                                                   python=platform.python_version(), torch=torch.__version__,
                                                   ultralytics=__import__("ultralytics").__version__), indent=1))
with zipfile.ZipFile(OUT / "send_back.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("seeds_results.csv", "seeds_per_class.csv", "seeds_summary.csv", "run_meta.json"):
        z.write(OUT / f, f)
print("send_back.zip written")
''')
nb.save("notebooks/seeds_eval.ipynb")
