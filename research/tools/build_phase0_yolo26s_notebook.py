"""Build notebooks/phase0_yolo26s.ipynb: score only the teammate's YOLO26s on VisDrone with the Phase 0 protocol.

Run:  python tools/build_phase0_yolo26s_notebook.py
"""
from nb_common import Notebook

nb = Notebook()
nb.md(r"""
# Phase 0 add-on: YOLO26s on VisDrone, same scorer as the other models

Fills the YOLO26s row of the Phase 0 tables without re-running the other models. Same dronekit code, same val-chosen confidence
rule. Three setups, all fixed before this run, and none tuned here:
- `whole1536` and `best_sliced`: the two setups every other model was scored with.
- `two_scale`: whole image at 1280 + 1920, NMS IoU 0.5. This is the setup the teammate selected on val (`prompt2.ipynb`), so the model also
  gets its own best recipe.

## How to run (about 1 h)
- **Inputs:** on your PC, rename `yolo26_visdrone_best.zip` to `yolo26s_visdrone.pt` (do not unzip it). Upload it as a private Kaggle
  Dataset and attach it. A dataset where Kaggle unpacked it into a `best/` folder also works.
- **Settings:** Accelerator **GPU T4 x2** (one GPU is used, so latencies match Phase 0), **Internet ON**.
- *Save Version → Save & Run All*. It tests 5 val images first and stops there if anything is wrong.
- **Send back** the whole `/kaggle/working/phase0_yolo26s/results/` folder and `figures/`.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest')

nb.code(r'''
# --- Config ----------------------------------------------------------------
import os, sys, gc, json, shutil, zipfile, logging, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, plots, report, score
logging.getLogger("sahi").setLevel(logging.WARNING)
from ultralytics import YOLO, __version__ as UL_VERSION
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)              # 'half' deprecation warning prints once per image otherwise

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
DATA = Path(os.environ.get("DK_DATA", "/tmp/VisDrone"))
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "phase0_yolo26s"
CUDA = torch.cuda.is_available()
DEVICE, HALF = (0, True) if CUDA else ("cpu", False)
GPU_NAME = torch.cuda.get_device_name(0) if CUDA else "cpu"
RUN_FULL = CUDA
SMOKE_N = int(os.environ.get("DK_SMOKE_N", "5"))
NAME = "YOLO26s"
CLASSES = dronekit.VISDRONE_CLASSES
EXPECTED_PARAMS = (9e6, 11e6)               # YOLO26s is 10.0M params
SETUPS = dict(whole1536=infer.SETUPS["whole1536"], best_sliced=infer.SETUPS["best_sliced"],
              two_scale=dict(passes=["full@1280", "full@1920"], method="NMS", metric="IOU", thr=0.5, edge=False))
PASSES = list(dict.fromkeys(p for s in SETUPS.values() for p in s["passes"]))
for d in ("results", "figures", "caches"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
print("Python", platform.python_version(), "| torch", torch.__version__, "| GPU", GPU_NAME, "| passes", PASSES)
''')

nb.code(r'''
# --- Weights: a renamed .pt, the original .zip, or a folder Kaggle unpacked. All are the same checkpoint -----
WEIGHTS = Path("yolo26s_visdrone.pt")
hit = next(iter(sorted(p for pat in ("yolo26s_visdrone*.pt", "yolo26_visdrone_best*") for p in INPUT.rglob(pat) if p.is_file())), None)
if hit:
    shutil.copy(hit, WEIGHTS)
else:
    pkl = next((p for p in sorted(INPUT.rglob("data.pkl")) if (p.parent / "data").is_dir()), None)
    assert pkl, f"attach the YOLO26s checkpoint: nothing found under {INPUT}"
    with zipfile.ZipFile(WEIGHTS, "w", zipfile.ZIP_STORED) as z:     # torch checkpoints are uncompressed zips
        for p in sorted(pkl.parent.rglob("*")):
            if p.is_file():
                z.write(p, Path(pkl.parent.name) / p.relative_to(pkl.parent))
model = YOLO(str(WEIGHTS))
names = [model.names[i] for i in range(len(model.names))]
assert names == CLASSES, f"model classes {names} are not the VisDrone classes in order"
n_params = sum(p.numel() for p in model.model.parameters())
assert EXPECTED_PARAMS[0] <= n_params <= EXPECTED_PARAMS[1], f"{n_params / 1e6:.1f}M params: this is not YOLO26s, stop and send the log"
TA = model.ckpt.get("train_args") or {}
PROV = dict(model=NAME, params=int(n_params), saved=model.ckpt.get("date"), trained_on_ultralytics=model.ckpt.get("version"),
            scored_on_ultralytics=UL_VERSION, train_args={k: TA.get(k) for k in ("model", "data", "imgsz", "batch", "epochs", "time", "optimizer", "lr0", "seed", "device")})
(OUT / "results" / "provenance_yolo26s.json").write_text(json.dumps(PROV, indent=1, default=str))
print(json.dumps(PROV, indent=1, default=str))
del model
''')

nb.code(r'''
# --- Data: VisDrone val + test-dev -------------------------------------------
SPLITS = {}
for s in ("val", "test-dev"):
    d = DATA / f"VisDrone2019-DET-{s}"
    if not (d / "annotations").exists() and (ON_KAGGLE or s == "val"):
        d = data.download_visdrone(DATA, s)
    if (d / "annotations").exists():
        SPLITS[s] = data.load_visdrone(d, s)
if ON_KAGGLE:
    assert (len(SPLITS["val"]), len(SPLITS["test-dev"])) == (548, 1610), {s: len(sp) for s, sp in SPLITS.items()}
print({s: len(sp) for s, sp in SPLITS.items()})

def run_cache(split, out_dir):
    path = out_dir / f"{NAME}__{split.name}"
    if not path.with_suffix(".parquet").exists():
        model = YOLO(str(WEIGHTS))
        model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, half=HALF, device=DEVICE, verbose=False)   # warm-up, as in Phase 0
        infer.run_and_cache(model, split, PASSES, path, cuda=CUDA, device=DEVICE, half=HALF, meta=dict(model=NAME, gpu=GPU_NAME))
        del model; gc.collect()
        if CUDA:
            torch.cuda.empty_cache()
    return path

def score_all(splits, caches, out):
    # val first: its best-F1 confidence is reused unchanged on test-dev
    rows, pcs, curves, cms, conf = [], [], {}, {}, {}
    for sname in [s for s in ("val", "test-dev") if s in caches]:
        sp = splits[sname]
        raw, meta = infer.load_cache(caches[sname])
        for setup, cfg in SETUPS.items():
            dets, mms = infer.apply_setup(raw, sp, cfg)
            m, pc, E = score.evaluate(sp, dets)
            if sname == "val" or setup not in conf:
                conf[setup] = score.best_f1_conf(E) if E is not None else 0.25
            c = conf[setup]; prc, pr25 = score.pr_at_conf(E, c), score.pr_at_conf(E, 0.25)
            rows.append(dict(model=NAME, split=sname, setup=setup, n_images=len(sp), precision=prc["precision"], recall=prc["recall"], f1=prc["f1"], conf=c,
                             P_at_0_25=pr25["precision"], R_at_0_25=pr25["recall"], mAP50=m["AP50"], **{"mAP50-95": m["AP"]}, AP75=m.get("AP75"),
                             AP50_tiny=m.get("AP50_tiny"), AP50_small=m.get("AP50_small"), **{"AP50_medium+": m.get("AP50_medium+")},
                             ms_per_img=round(infer.setup_ms(meta, cfg, mms), 1)))
            pcs += [dict(model=NAME, split=sname, setup=setup, cls=k, AP50=v) for k, v in pc.items()]
            curves[(setup, sname)] = score.pr_curves(E)
            cms[(setup, sname)] = (score.confusion_matrix(sp, dets, c), c)
            print(f"{sname:8s} {setup:12s} mAP50={m['AP50']:.4f} mAP50-95={m['AP']:.4f} P={prc['precision']:.3f} R={prc['recall']:.3f} @conf {c:.2f} {rows[-1]['ms_per_img']:.0f} ms")
    summary, per_class = pd.DataFrame(rows), pd.DataFrame(pcs)
    summary.to_csv(out / "results" / "basic_metrics_yolo26s.csv", index=False)
    per_class.to_csv(out / "results" / "per_class_ap50_long_yolo26s.csv", index=False)
    for (setup, sname), (cm, c) in cms.items():
        pd.DataFrame(cm, index=CLASSES + ["background"], columns=CLASSES + ["background"]).to_csv(out / "results" / f"confusion_{NAME}__{setup}__{sname}.csv")
    return summary, per_class, curves, cms
''')

nb.code(r'''
# --- 5-image smoke test: the whole pipeline on val images, before any long run --------------------------
SM = WORK / "phase0_yolo26s_smoke"; shutil.rmtree(SM, ignore_errors=True)
for d in ("results", "caches"):
    (SM / d).mkdir(parents=True)
sm_split = {"val": SPLITS["val"].subset(SMOKE_N)}
s_sum, _, _, _ = score_all(sm_split, {"val": run_cache(sm_split["val"], SM / "caches")}, SM)
assert len(s_sum) == len(SETUPS) and s_sum["mAP50"].between(0, 1).all()
print("smoke test passed")
''')

nb.md("## Full run (GPU only): val + test-dev")
nb.code(r'''
# Test-dev rule: these three setups are fixed above and each is scored once here, with no tuning.
if RUN_FULL:
    CACHES = {s: run_cache(sp, OUT / "caches") for s, sp in SPLITS.items()}
    SUMMARY, PER_CLASS, CURVES, CMS = score_all(SPLITS, CACHES, OUT)
    cols = ["model", "split", "setup", "precision", "recall", "conf", "mAP50", "mAP50-95", "AP50_tiny", "AP50_small", "AP50_medium+", "ms_per_img"]
    wide = PER_CLASS[PER_CLASS.setup == "best_sliced"].pivot(index="cls", columns="split", values="AP50").reindex(CLASSES)
    wide["drop"] = wide["val"] - wide["test-dev"]
    wide.index.name = "class"
    wide.to_csv(OUT / "results" / "per_class_val_vs_testdev_yolo26s.csv")
    for (setup, sname), (cm, c) in CMS.items():
        if setup == "best_sliced":
            plots.plot_confusion(score.normalize_rows(cm), CLASSES, OUT / "figures" / f"confusion_{NAME}__{sname}.png", f"{NAME}, {sname}", c)
    plots.plot_pr_curves({s: CURVES[("best_sliced", s)] for s in SPLITS}, CLASSES, OUT / "figures" / f"pr_{NAME}.png", title=f"{NAME}: PR curves at IoU 0.5 (best sliced)")
    plots.plot_per_class_bars(wide[["val", "test-dev"]], OUT / "figures" / f"per_class_{NAME}.png", title=f"{NAME}: per-class AP50 (best sliced)")
    md_ = ["# Phase 0 add-on: YOLO26s (generated)", "", f"dronekit {dronekit.__version__}, GPU: {GPU_NAME}", "",
           "## Basic metrics (precision/recall at the val best-F1 confidence, IoU 0.5)", report.md_table(SUMMARY[cols]), "",
           "## Per-class AP50, best sliced", report.md_table(wide.reset_index())]
    (OUT / "results" / "REPORT_tables_yolo26s.md").write_text("\n".join(md_), encoding="utf-8")
    shutil.rmtree(SM, ignore_errors=True)
    print((OUT / "results" / "REPORT_tables_yolo26s.md").read_text(encoding="utf-8"))
    !du -sh {OUT} {OUT}/*
else:
    print("no GPU: full run skipped (the smoke test above checked the pipeline)")
''')

nb.save("notebooks/phase0_yolo26s.ipynb")
