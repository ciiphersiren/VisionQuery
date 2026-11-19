"""Build notebooks/phase0_metrics.ipynb with the dronekit package and tests embedded as %%writefile cells.

Run:  python tools/build_phase0_notebook.py
Re-run after any change to dronekit/ or tests/ so the notebook carries the current code.
When the GitHub repo exists, the %%writefile cells become one `pip install git+...@<commit>` line.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_FILES = ["dronekit/__init__.py", "dronekit/data.py", "dronekit/infer.py", "dronekit/score.py",
             "dronekit/plots.py", "dronekit/report.py",
             "tests/conftest.py", "tests/test_data.py", "tests/test_infer.py", "tests/test_score.py"]

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))


md(r"""
# Phase 0: basic metrics for every trained model (VisDrone val + test-dev)

Supervisor's request: proof that each model was trained, plus its basic metrics.
Everything goes through **dronekit** (embedded below), the one implementation of the evaluation
protocol: 512 px slices resized to 1280 + whole image at 1536, boundary filter,
class-wise NMS 0.6, conf ≥ 0.01, COCO AP at maxDets 500 with ignore regions as crowd boxes.

**Models:** YOLO26m-Tile, YOLOE-26m-FT, YOLOE-26m-LP, YOLOE-26m-ZS (descriptive prompts, the val-chosen set from
`yoloe-1`), YOLO26s (only if its weights are attached), plus the Ensemble (YOLO26m-Tile + YOLOE-26m-FT, WBF 0.65).

## How to run

**Inputs:** Add Input → Your Work → Notebooks, then attach the outputs of
1. `visdrone-training` (YOLO26m-Tile: `runs/K1_y26m_tiles/…`)
2. `yoloe-2` (YOLOE-26m-FT: `yoloe26m_e2_full.pt`, `runs_yoloe/e2_full/…`)
3. `yoloe-1` (YOLOE-26m-LP: `yoloe26m_e1_linprobe.pt`, `runs_yoloe/e1_linprobe/…`)
4. `evaluation` (`eval_v2/results_v2.csv` and `eval_v2/cache/*.pkl`, used only for the regression check)
5. *(optional)* the YOLO26s weights. Set `YOLO26S_PATTERN` in the Config cell to their filename.

Files are found by name with `rglob`, so folder layout doesn't matter. Settings: **Internet ON**.

**Step 1: CPU check (about 15 min, no GPU hours used).** Accelerator **None** → *Save Version → Save & Run All*.
The unit tests, the regression check against the old eval-v2 caches and the 5-image smoke test run;
the full run is skipped automatically because there is no GPU. Send back the log if anything fails.

**Step 2: full run (about 2–2.5 h).** Accelerator **GPU T4 x2** → *Save Version → Save & Run All*.
Only GPU 0 is used, so latencies are single-T4 numbers.

**Output:** everything is in `/kaggle/working/phase0/` (results CSVs, figures, detection caches, weights, run files,
code). From the finished version's Output tab, make one **private Dataset** from it (*New Dataset*) and share
it with the second account. Phases 3–4 read the caches from there, so no detector is re-run on VisDrone.
Download `phase0/results/REPORT_tables.md` and the log of the last cells.
""")

code(r"""
# --- Install (Kaggle) ------------------------------------------------------
from pathlib import Path
if Path("/kaggle").exists():
    !pip install -q "ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest "git+https://github.com/ultralytics/CLIP.git"
Path("dronekit_src/dronekit").mkdir(parents=True, exist_ok=True)
Path("dronekit_src/tests").mkdir(parents=True, exist_ok=True)
""")

md("## dronekit package (generated from the repo by `tools/build_phase0_notebook.py`; do not edit here)")
for f in PKG_FILES:
    code(f"%%writefile dronekit_src/{f}\n" + (ROOT / f).read_text(encoding="utf-8"))

code(r"""
# --- Unit tests (CPU, seconds) ----------------------------------------------
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "dronekit_src/tests"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
assert r.returncode == 0, "unit tests failed: stop here and send the log"
""")

code(r"""
# --- Config ----------------------------------------------------------------
import os, sys, gc, json, time, shutil, pickle, logging, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, score, plots, report
logging.getLogger("sahi").setLevel(logging.WARNING)
try:
    from ultralytics.utils import LOGGER; LOGGER.setLevel(logging.ERROR)   # hides the 'half is deprecated' flood
except Exception:
    pass

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
DATA = Path(os.environ.get("DK_DATA", "/tmp/VisDrone"))            # images live in /tmp (lost at session end)
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "phase0"
CUDA = torch.cuda.is_available()
DEVICE, HALF = (0, True) if CUDA else ("cpu", False)
FULL_N = int(os.environ.get("DK_FULL_N", "0")) or None             # local CPU test only: "full" section on N images
RUN_FULL = bool(CUDA or FULL_N) and os.environ.get("DK_SKIP_FULL") != "1"   # full val + test-dev run needs the GPU
GPU_NAME = torch.cuda.get_device_name(0) if CUDA else "cpu"
SMOKE_N = 5
YOLO26S_PATTERN = None          # e.g. "yolo26s_visdrone_best.pt" once the teammate's weights are attached
CLASSES = dronekit.VISDRONE_CLASSES

def find(pattern):
    hits = sorted(INPUT.rglob(pattern)) if INPUT.exists() else []
    if len(hits) > 1:
        print(f"  note: {len(hits)} files match {pattern!r}; using {hits[0]}")
    return hits[0] if hits else None

# Zero-shot prompt sets, copied from yoloe-1.ipynb. "descriptive" won on val there (AP50 0.261 vs 0.200 for names).
ZS_PROMPTS = {
    "descriptive": {"pedestrian": "person walking or standing", "people": "person sitting", "bicycle": "bicycle",
                    "car": "car", "van": "van", "truck": "truck", "tricycle": "tricycle",
                    "awning-tricycle": "tricycle with a canopy", "bus": "bus", "motor": "motorcycle"},
    "names": {c: c for c in CLASSES},
}
D2 = "VisDrone2019-DET train → D2: ignore-masked full images + 640/960 px tiles (25% overlap)"
MODELS = {
    "YOLO26m-Tile": dict(weights=find("K1_y26m_tiles/weights/best.pt"), results=find("K1_y26m_tiles/results.csv"),
                         args=find("K1_y26m_tiles/args.yaml"), notebook="visdrone-training", init="yolo26m.pt (COCO)",
                         train_data=D2, train_images="70,216 (6,471 full + 63,745 tiles)"),
    "YOLOE-26m-FT": dict(weights=find("yoloe26m_e2_full.pt"), results=find("e2_full/results.csv"),
                         args=find("e2_full/args.yaml"), notebook="yoloe-2", init="yoloe-26m-seg.pt",
                         train_data=D2, train_images="70,216 (6,471 full + 63,745 tiles)"),
    "YOLOE-26m-LP": dict(weights=find("yoloe26m_e1_linprobe.pt"), results=find("e1_linprobe/results.csv"),
                         args=find("e1_linprobe/args.yaml"), notebook="yoloe-1", init="yoloe-26m-seg.pt",
                         train_data=D2, train_images="70,216 (6,471 full + 63,745 tiles)",
                         note="linear probe: 15,420 of 21,788,308 params trainable (yoloe-1 log)"),
    "YOLOE-26m-ZS": dict(weights="yoloe-26m-seg.pt", prompts="descriptive", notebook="none (Ultralytics release)",
                         init="yoloe-26m-seg.pt", train_data="not trained on VisDrone (zero-shot)", train_images="0"),
    "YOLO26s": dict(weights=find(YOLO26S_PATTERN) if YOLO26S_PATTERN else None, notebook="teammate harness",
                    init="yolo26s.pt (COCO)", train_data="VisDrone2019-DET train, 640 px slices (teammate)",
                    train_images="see teammate notebook"),
}
# Extra val-only row: zero-shot with plain class names, whole image only (resolves which prompts gave 0.261).
EXTRA_VAL_ONLY = {"YOLOE-26m-ZS (names)": dict(weights="yoloe-26m-seg.pt", prompts="names", passes=["full@1536"])}
ENSEMBLE = ("YOLO26m-Tile", "YOLOE-26m-FT")
SETUPS = ["whole1536", "best_sliced"]
PASSES = infer.SETUPS["best_sliced"]["passes"]            # ["s512@1280", "full@1536"]; whole1536 reuses full@1536

print("Python", platform.python_version(), "| torch", torch.__version__, "| CUDA", CUDA, "| GPUs", torch.cuda.device_count())
for k, v in MODELS.items():
    print(f"{k:14s} weights={v['weights']}\n{'':14s} results={v.get('results')}  args={v.get('args')}")
missing = [k for k in ("YOLO26m-Tile", "YOLOE-26m-FT", "YOLOE-26m-LP") if MODELS[k]["weights"] is None]
if missing:
    print("\n!! weights not found for", missing, ": attach the notebook outputs listed at the top (they will be skipped)")
""")

code(r"""
# --- Data: VisDrone val + test-dev (official annotations) -------------------
SPLITS = {}
for s in ("val", "test-dev"):
    d = DATA / f"VisDrone2019-DET-{s}"
    if not (d / "annotations").exists() and (ON_KAGGLE or s == "val"):
        d = data.download_visdrone(DATA, s)
    if (d / "annotations").exists():
        SPLITS[s] = data.load_visdrone(d, s)
STATS = pd.DataFrame({s: data.split_stats(sp) for s, sp in SPLITS.items()}).T
print(STATS[["images", "sequences", "objects", "objects_per_image", "tiny", "small", "medium_plus", "share_bus"]].round(3))
# expected: val 548 / 76 / 70.7 per image / tiny 0.308 / small 0.377 / bus 0.006;
#                              test-dev 1610 / 61 / 46.6 / 0.364 / 0.313 / 0.039
""")

code(r"""
# --- Helpers: load a model, run + cache its passes, score every setup -------
def load_model(name, cfg):
    from ultralytics import YOLO, YOLOE
    if cfg.get("prompts"):
        m = YOLOE(str(cfg["weights"]))
        m.set_classes([ZS_PROMPTS[cfg["prompts"]][c] for c in CLASSES])   # one prompt per class, in class order
        cls_map = None
    else:
        m = YOLO(str(cfg["weights"]))                 # 'yoloe' in the filename -> loads as YOLOE
        names = [m.names[i] for i in range(len(m.names))]
        if names == CLASSES:
            cls_map = None
        elif set(names) == set(CLASSES):
            cls_map = {i: CLASSES.index(n) for i, n in enumerate(names)}
        else:
            raise ValueError(f"{name}: model classes {names} are not the VisDrone classes")
    m.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, half=HALF, device=DEVICE, verbose=False)   # warm-up
    return m, cls_map


def run_models(models, split, cache_dir):
    # One cache per model x split: every raw pass down to conf 0.01, with slice boxes and timings.
    paths = {}
    for name, cfg in models.items():
        if cfg.get("weights") is None:
            continue
        path = cache_dir / f"{name.replace(' ', '_')}__{split.name}"
        if not path.with_suffix(".parquet").exists():
            m, cmap = load_model(name, cfg)
            infer.run_and_cache(m, split, cfg.get("passes", PASSES), path, cuda=CUDA, device=DEVICE, half=HALF,
                                cls_map=cmap, meta=dict(model=name, weights=str(cfg["weights"]),
                                                        prompts=cfg.get("prompts"), device=str(DEVICE),
                                                        gpu=GPU_NAME))
            del m; gc.collect()
            if CUDA: torch.cuda.empty_cache()
        paths[name] = path
    return paths


def save_dets(dets, split, path):
    rows = [np.c_[np.full(len(d), i), d] for i, d in enumerate(dets[r.key] for r in split.images) if len(d)]
    a = np.concatenate(rows) if rows else np.zeros((0, 7))
    df = pd.DataFrame(a[:, 1:], columns=["x1", "y1", "x2", "y2", "score", "cls"]).astype(np.float32)
    df.insert(0, "image", pd.Categorical([split.images[int(i)].key for i in a[:, 0]]))
    df["cls"] = df["cls"].astype(np.int16)
    df.to_parquet(path, compression="zstd", index=False)


def score_everything(caches, splits, out):
    # caches: {split_name: {model: cache path}}. val is processed first so its best-F1 confidence
    # can be reused unchanged on test-dev.
    res_dir, fig_dir, det_dir = out / "results", out / "figures", out / "merged_dets"
    for d in (res_dir, fig_dir, det_dir): d.mkdir(parents=True, exist_ok=True)
    rows, pcs, curves, cms, conf_star, keep = [], [], {}, {}, {}, {}

    def record(name, sname, setup, dets, ms):
        sp = splits[sname]
        m, pc, E = score.evaluate(sp, dets)
        if sname == "val" or (name, setup) not in conf_star:
            conf_star[(name, setup)] = score.best_f1_conf(E) if E is not None else 0.25
        c = conf_star[(name, setup)]
        prc, pr25 = score.pr_at_conf(E, c), score.pr_at_conf(E, 0.25)
        rows.append(dict(model=name, split=sname, setup=setup, n_images=len(sp),
                         precision=prc["precision"], recall=prc["recall"], f1=prc["f1"], conf=c,
                         P_at_0_25=pr25["precision"], R_at_0_25=pr25["recall"],
                         mAP50=m["AP50"], **{"mAP50-95": m["AP"]}, AP75=m.get("AP75"),
                         AP50_tiny=m.get("AP50_tiny"), AP50_small=m.get("AP50_small"),
                         **{"AP50_medium+": m.get("AP50_medium+")}, ms_per_img=round(ms, 1)))
        for k, v in pc.items():
            pcs.append(dict(model=name, split=sname, setup=setup, cls=k, AP50=v))
        if setup == "best_sliced" and E is not None:
            curves[(name, sname)] = score.pr_curves(E)
            cms[(name, sname)] = (score.confusion_matrix(sp, dets, c), c)
        print(f"  {name:22s} {sname:8s} {setup:12s} mAP50={m['AP50']:.4f} mAP50-95={m['AP']:.4f} "
              f"P={prc['precision']:.3f} R={prc['recall']:.3f} @conf {c:.2f}  {ms:.0f} ms")

    for sname in [s for s in ("val", "test-dev") if s in caches]:
        sp = splits[sname]
        for name, path in caches[sname].items():
            raw, meta = infer.load_cache(path)
            for setup in SETUPS:
                if not all(p in raw for p in infer.SETUPS[setup]["passes"]):
                    continue
                dets, mms = infer.apply_setup(raw, sp, setup)
                record(name, sname, setup, dets, infer.setup_ms(meta, setup, mms))
                save_dets(dets, sp, det_dir / f"{name.replace(' ', '_')}__{sname}__{setup}.parquet")
                if setup == "best_sliced" and name in ENSEMBLE:
                    keep[(name, sname)] = (dets, infer.setup_ms(meta, setup, mms))
            del raw; gc.collect()
        if all((n, sname) in keep for n in ENSEMBLE):
            t0 = time.perf_counter()
            dets = infer.ensemble([keep[(n, sname)][0] for n in ENSEMBLE], sp)
            ms = sum(keep[(n, sname)][1] for n in ENSEMBLE) + 1000 * (time.perf_counter() - t0) / len(sp)
            record("Ensemble", sname, "best_sliced", dets, ms)
            save_dets(dets, sp, det_dir / f"Ensemble__{sname}__best_sliced.parquet")

    summary = pd.DataFrame(rows)
    per_class = pd.DataFrame(pcs)
    summary.to_csv(res_dir / "basic_metrics.csv", index=False)
    per_class.to_csv(res_dir / "per_class_ap50_long.csv", index=False)
    for (name, sname), (cm, c) in cms.items():
        pd.DataFrame(cm, index=CLASSES + ["background"], columns=CLASSES + ["background"]).to_csv(
            res_dir / f"confusion_{name.replace(' ', '_')}__{sname}.csv")
    return summary, per_class, curves, cms
""")

code(r"""
# --- Regression check: dronekit vs the old eval-v2 numbers (CPU, uses the attached evaluation output) ---
# Scores the *old* cached passes from evaluation(1).ipynb with dronekit. Expect an exact match with results_v2.csv:
# same detections in, same protocol -> same AP. Any difference means the package changed the protocol.
OLD_CSV = find("results_v2.csv")
REGRESSION = []
if OLD_CSV is None:
    print("evaluation output not attached: regression check skipped")
else:
    old = pd.read_csv(OLD_CSV)
    for tag in ("K1", "E2", "E1"):
        for sname, sp in SPLITS.items():
            raw = {}
            for p in PASSES:
                f = find(f"{tag}_{sname}_{p}_all.pkl")
                if f is not None:
                    out_, _sec = pickle.load(open(f, "rb"))        # {image id (position): (dets, slice boxes)}
                    raw[p] = {sp.images[i].key: v for i, v in out_.items()}
            for setup, (passes, merge) in {"whole1536": ("full@1536", "NMS-IOU-0.5"),
                                           "best_sliced": ("s512@1280+full@1536", "NMS-IOU-0.6+edge")}.items():
                ref = old[(old["model"] == tag) & (old["split"] == sname) & (old["passes"] == passes) & (old["merge"] == merge)]
                if ref.empty or not all(p in raw for p in infer.SETUPS[setup]["passes"]):
                    continue
                dets, _ = infer.apply_setup(raw, sp, setup)
                m, _, _ = score.evaluate(sp, dets)
                REGRESSION.append(dict(model=tag, split=sname, setup=setup, old_AP50=ref.AP50.iloc[0], new_AP50=round(m["AP50"], 4),
                                       old_AP=ref.AP.iloc[0], new_AP=round(m["AP"], 4)))
            del raw; gc.collect()
    REG = pd.DataFrame(REGRESSION)
    print(REG.to_string(index=False) if len(REG) else "no old caches found (attach eval_v2/cache)")
    if len(REG):
        bad = REG[(REG.old_AP50 - REG.new_AP50).abs() > 0.0011]
        assert bad.empty, f"dronekit does not reproduce eval v2:\n{bad}"
        print("regression check passed: dronekit reproduces eval v2")
""")

code(r"""
# --- 5-image smoke test (runs on CPU or GPU; finishes the whole pipeline on 5 val images) ---
SMOKE = WORK / "phase0_smoke"
shutil.rmtree(SMOKE, ignore_errors=True)
smoke_split = {"val": SPLITS["val"].subset(SMOKE_N)}
smoke_models = {k: v for k, v in MODELS.items() if v.get("weights") is not None}
t0 = time.time()
smoke_caches = {"val": run_models(smoke_models, smoke_split["val"], SMOKE / "caches")}
s_sum, s_pc, s_curves, s_cms = score_everything(smoke_caches, smoke_split, SMOKE)
for name in list(s_curves)[:1]:
    plots.plot_pr_curves({"val": s_curves[name]}, CLASSES, SMOKE / "figures" / "pr.png", title=f"{name[0]} (smoke)")
    plots.plot_confusion(score.normalize_rows(s_cms[name][0]), CLASSES, SMOKE / "figures" / "cm.png", "smoke", s_cms[name][1])
assert len(s_sum) >= 2 * len(smoke_models), "missing rows"
assert s_sum["mAP50"].between(0, 1).all()
raw0, meta0 = infer.load_cache(next(iter(smoke_caches["val"].values())))
assert set(raw0) == set(PASSES) and len(raw0["full@1536"]) <= SMOKE_N
print(f"\nsmoke test passed in {time.time() - t0:.0f} s ({len(smoke_models)} models)")
s_sum[["model", "setup", "mAP50", "mAP50-95", "precision", "recall", "ms_per_img"]]
""")

md("## Full run (GPU only): val + test-dev for every model")

code(r"""
# Test-dev rule: each configuration is scored on test-dev once. YOLO26m-Tile, YOLOE-26m-FT and the
# Ensemble repeat configurations already reported; YOLOE-26m-LP and YOLOE-26m-ZS are scored on test-dev for the first time.
CACHES = {}
if RUN_FULL:
    if FULL_N:
        SPLITS = {s: sp.subset(FULL_N) for s, sp in SPLITS.items()}
    for sname, sp in SPLITS.items():
        print(f"== {sname}: {len(sp)} images")
        CACHES[sname] = run_models({k: v for k, v in MODELS.items() if v.get("weights") is not None}, sp, OUT / "caches")
    CACHES["val"].update(run_models(EXTRA_VAL_ONLY, SPLITS["val"], OUT / "caches"))
else:
    print("no GPU: full run skipped (the smoke test above checked the pipeline)")
""")

code(r"""
# --- Score: whole1536 and best_sliced for every model, plus the Ensemble ------
if RUN_FULL:
    SUMMARY, PER_CLASS, CURVES, CMS = score_everything(CACHES, SPLITS, OUT)
    cols = ["model", "split", "setup", "precision", "recall", "conf", "mAP50", "mAP50-95",
            "AP50_tiny", "AP50_small", "AP50_medium+", "ms_per_img"]
    display(SUMMARY[cols].round(3))
""")

code(r"""
# --- Per-class AP50: val vs test-dev side by side (best sliced), with GT counts per class ---
if RUN_FULL:
    res = OUT / "results"
    counts = {s: pd.Series(np.concatenate([r.boxes[:, 0] for r in sp.images]).astype(int)).value_counts()
              for s, sp in SPLITS.items()}
    wide = pd.DataFrame(index=CLASSES)
    for s in SPLITS:
        wide[f"n_{s}"] = [int(counts[s].get(k, 0)) for k in range(len(CLASSES))]
        wide[f"share_{s}"] = (wide[f"n_{s}"] / wide[f"n_{s}"].sum()).round(4)
    bs = PER_CLASS[PER_CLASS.setup == "best_sliced"]
    for name in bs.model.unique():
        for s in SPLITS:
            wide[f"{name} {s}"] = bs[(bs.model == name) & (bs.split == s)].set_index("cls").AP50.reindex(CLASSES).values
        if all(f"{name} {s}" in wide for s in ("val", "test-dev")):
            wide[f"{name} drop"] = wide[f"{name} val"] - wide[f"{name} test-dev"]
    wide.index.name = "class"
    wide.to_csv(res / "per_class_val_vs_testdev.csv")
    display(wide.round(3))
""")

code(r"""
# --- Figures: PR curves, confusion matrices, per-class bars, training curves ---
if RUN_FULL:
    fig = OUT / "figures"
    for name in SUMMARY.model.unique():
        fn = name.replace(" ", "_")
        cv = {s: CURVES[(name, s)] for s in ("val", "test-dev") if (name, s) in CURVES}
        if cv:
            plots.plot_pr_curves(cv, CLASSES, fig / f"pr_{fn}.png", title=f"{name}: PR curves at IoU 0.5 (best sliced)")
        for s in ("val", "test-dev"):
            if (name, s) in CMS:
                cm, c = CMS[(name, s)]
                plots.plot_confusion(score.normalize_rows(cm), CLASSES, fig / f"confusion_{fn}__{s}.png", f"{name}, {s}", c)
        cols_ = [c for c in (f"{name} val", f"{name} test-dev") if c in wide]
        if cols_:
            plots.plot_per_class_bars(wide[cols_].rename(columns=lambda c: c.split(" ")[-1]), fig / f"per_class_{fn}.png",
                                      title=f"{name}: per-class AP50 (best sliced)",
                                      cols=[c.split(" ")[-1] for c in cols_])
    for name, cfg in MODELS.items():
        if cfg.get("results"):
            plots.plot_training_curves(cfg["results"], fig / f"training_{name}.png", title=name)
    print(*sorted(p.name for p in fig.glob("*.png")), sep="\n")
""")

code(r"""
# --- Training provenance --------------------------------------------------------
prov = []
for name, cfg in MODELS.items():
    w = cfg.get("weights")
    trained_by_us = w and not cfg.get("prompts")        # the zero-shot release was not trained by us
    info = report.read_run(w if Path(str(w)).exists() else None, cfg.get("results"), cfg.get("args")) if trained_by_us else {}
    prov.append(dict(model=name, training_dataset=cfg["train_data"], split="train" if cfg["train_images"] != "0" else "-",
                     image_count=cfg["train_images"], epochs=info.get("epochs_completed"), train_hours=info.get("train_hours"),
                     best_epoch=info.get("best_epoch"), init=cfg["init"], weights_file=Path(str(w)).name if w else "not attached",
                     notebook=cfg["notebook"], note=cfg.get("note", ""), **{k: v for k, v in info.items() if k.startswith("arg_")},
                     ultralytics_version=info.get("ultralytics_version"),
                     **{"val mAP50 (Ultralytics validator)": info.get("best_val_mAP50_ultralytics"),
                        "val mAP50-95 (Ultralytics validator)": info.get("best_val_mAP50-95_ultralytics")}))
PROV = pd.DataFrame(prov)
(OUT / "results").mkdir(parents=True, exist_ok=True)
PROV.to_csv(OUT / "results" / "training_provenance.csv", index=False)
display(PROV)
""")

code(r"""
# --- Bundle: weights + run files + code next to the results, then the markdown tables ---
if RUN_FULL:
    for name, cfg in MODELS.items():
        d = OUT / "runs" / name; d.mkdir(parents=True, exist_ok=True)
        for k in ("results", "args"):
            if cfg.get(k): shutil.copy(cfg[k], d / Path(cfg[k]).name)
        w = cfg.get("weights")
        if w and Path(str(w)).exists() and not cfg.get("prompts"):
            dst = "yolo26m_tile_visdrone.pt" if name == "YOLO26m-Tile" else Path(w).name   # keep 'yoloe' in YOLOE names
            (OUT / "weights").mkdir(exist_ok=True); shutil.copy(w, OUT / "weights" / dst)
    shutil.copytree("dronekit_src", OUT / "dronekit_src", dirs_exist_ok=True)
    if len(REGRESSION): pd.DataFrame(REGRESSION).to_csv(OUT / "results" / "regression_vs_eval_v2.csv", index=False)
    STATS.to_csv(OUT / "results" / "split_stats.csv")
    cols = ["model", "split", "setup", "precision", "recall", "conf", "mAP50", "mAP50-95",
            "AP50_tiny", "AP50_small", "AP50_medium+", "ms_per_img"]
    md_ = ["# Phase 0 tables (generated)", "", f"dronekit {dronekit.__version__}, GPU: {GPU_NAME}", "",
           "## Basic metrics (precision/recall at the val best-F1 confidence, IoU 0.5)", report.md_table(SUMMARY[cols]), "",
           "## Per-class AP50, best sliced", report.md_table(wide.reset_index()), "",
           "## Training provenance", report.md_table(PROV), "",
           "## Regression vs eval v2", report.md_table(pd.DataFrame(REGRESSION)) if len(REGRESSION) else "not run"]
    (OUT / "results" / "REPORT_tables.md").write_text("\n".join(md_), encoding="utf-8")
    meta = dict(dronekit=dronekit.__version__, torch=torch.__version__, python=platform.python_version(),
                gpu=GPU_NAME, settings=dict(conf=infer.CONF, max_det=infer.MAX_DET, overlap=infer.OVERLAP,
                setups=infer.SETUPS, ensemble_wbf=infer.ENSEMBLE_WBF_THR, max_dets=score.MAX_DETS, areas=score.AREAS))
    import ultralytics, sahi; meta.update(ultralytics=ultralytics.__version__, sahi=sahi.__version__)
    (OUT / "results" / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    !du -sh {OUT} {OUT}/caches {OUT}/merged_dets {OUT}/weights
    print(open(OUT / "results" / "REPORT_tables.md", encoding="utf-8").read())
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
out = ROOT / "notebooks" / "phase0_metrics.ipynb"
out.parent.mkdir(exist_ok=True)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(cells), "cells")
