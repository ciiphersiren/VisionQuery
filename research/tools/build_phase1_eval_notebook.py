"""Build notebooks/phase1_eval.ipynb: Phase 1 steps 3-5 in one run (whole vs sliced on UAVDT val, UAVDT test once, CARPK,
cross-dataset matrix, basic metrics, detection caches).

Run:  python tools/build_phase1_eval_notebook.py
"""
from nb_common import CARPK_FETCH_CELL, UAVDT_FETCH_CELL, Notebook

nb = Notebook()
nb.md(r"""
# Phase 1, steps 3-5: cross-dataset evaluation

Runs every detection job for the paper in one go, on both GPUs:
1. **Step 3:** whole image (`full@1024`) vs sliced (`s512@1024 + full@1536`) for both UAVDT models, on UAVDT val only.
   Sliced is used only if it beats whole by at least 0.005 eval-v2 AP (rule fixed in the Config cell).
2. **UAVDT test and CARPK test, once.** The setups and checkpoints are fixed before any test image is loaded.
3. **Cross-dataset matrix** on car / truck / bus (CARPK: car), with VisDrone `van` as car and as ignore. CARPK test is one parking lot, so
   its train images (6 other lots, which no model trains on) are scored as a separate column.
4. **Basic metrics** for both UAVDT models, in the Phase 0 format.
5. **Detection caches and merged detections** for the CPU phases.

The last-epoch checkpoints are a sensitivity row, declared here before any test result exists: both UAVDT models selected epoch 1, so
the paper must show what full training gives on test as well.

## How to run (about 1 h; an estimate, the first real timing comes from this run)
1. **Weights.** Download these from the notebook outputs, rename them exactly, and upload them together as one private Kaggle Dataset:

| Name in the Dataset | Comes from |
|---|---|
| `yolo26m_tile_visdrone.pt` | `visdrone-training` output: `K1_y26m_tiles/weights/best.pt` |
| `yolo26s_visdrone.pt` | the file in your project folder |
| `yolo26m_tile_uavdt.pt` | `phase1_retrain` output, top folder |
| `yolo26m_tile_uavdt_last.pt` | `phase1_retrain` output: `runs/uavdt_y26m_tiles_musgd/weights/last.pt` |
| `yolo26s_uavdt.pt` | `phase1_retrain_yolo26s` output, top folder |
| `yolo26s_uavdt_last.pt` | `phase1_retrain_yolo26s` output: `runs/uavdt_y26s_sf/weights/last.pt` |

2. **CARPK password.** *Add-ons → Secrets*: add `CARPK_PASSWORD` and attach it to this notebook. It stays out of the code.
3. **Optional, saves about 15 min and adds the Ensemble and zero-shot rows later:** attach the Phase 0 output, if you still have it.
4. **Settings:** GPU T4 x2, Internet ON. *Save Version → Save & Run All*.
5. **It stops early** if a checkpoint does not reproduce its known val score, so a wrong file cannot reach the test sets.
6. **Send back** `/kaggle/working/phase1_eval/send_back.zip` (results, figures, merged detections).
   **Keep** the version: `caches/` holds the raw passes.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest gdown')

nb.code(r'''
# --- Config ----------------------------------------------------------------
import os, sys, gc, json, time, shutil, pickle, zipfile, logging, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, plots, report, score
logging.getLogger("sahi").setLevel(logging.WARNING)

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
RAW = Path(os.environ.get("DK_UAVDT_RAW", "/tmp/UAVDT/raw"))
VIS = Path(os.environ.get("DK_DATA", "/tmp/VisDrone"))
CARPK_RAW = Path(os.environ.get("DK_CARPK_RAW", "/tmp/CARPK"))
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "phase1_eval"
CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count() if CUDA else 0
SMOKE = not CUDA                                   # no GPU: a few frames of each split, stand-in weights
SMOKE_N = 3

VAL_SEQS = ["M0201", "M0402", "M0604", "M0902", "M1006"]   # selected in phase1_retrain from train-sequence labels only
VAL_STRIDE = TEST_STRIDE = 10
EXPECTED_N = {("uavdt", "val"): 367, ("uavdt", "test"): 1665, ("visdrone", "val"): 548, ("visdrone", "test-dev"): 1610}
EXPECTED_SEQS = {"train": 30, "test": 20}

# Step 3 rule, fixed before the run
WHOLE = dict(passes=["full@1024"], method=None)
SLICED = infer.SETUPS["fast"]                      # s512@1024 + full@1536, boundary filter, NMS 0.6
MIN_GAIN = 0.005                                   # eval-v2 AP on UAVDT val; the epoch-selection metric

VD, UV = dronekit.VISDRONE_CLASSES, data.UAVDT_CLASSES
# tag -> weights, trained-on, class list, parameter bounds, display name. Setups are filled in below.
MODELS = {
    "tile_vd": dict(file="yolo26m_tile_visdrone.pt", train="VisDrone", classes=VD, params=(19e6, 22e6), name="YOLO26m-Tile (VisDrone)", cost=1.0),
    "s_vd": dict(file="yolo26s_visdrone.pt", train="VisDrone", classes=VD, params=(9e6, 11e6), name="YOLO26s (VisDrone)", cost=0.55),
    "tile_uv": dict(file="yolo26m_tile_uavdt.pt", train="UAVDT", classes=UV, params=(19e6, 22e6), name="YOLO26m-Tile (UAVDT)", cost=1.0),
    "s_uv": dict(file="yolo26s_uavdt.pt", train="UAVDT", classes=UV, params=(9e6, 11e6), name="YOLO26s (UAVDT)", cost=0.55),
    "tile_uv_last": dict(file="yolo26m_tile_uavdt_last.pt", train="UAVDT", classes=UV, params=(19e6, 22e6), name="YOLO26m-Tile (UAVDT), last epoch", cost=1.0),
    "s_uv_last": dict(file="yolo26s_uavdt_last.pt", train="UAVDT", classes=UV, params=(9e6, 11e6), name="YOLO26s (UAVDT), last epoch", cost=0.55),
}
# Known val scores: a checkpoint that does not reproduce its own stops the run before any test image is scored
# ("selection.json" / "epoch_scan.csv" of the retrains, full@1024; Phase 0 best_sliced).
EXPECTED_VAL = {"tile_uv": ("AP", 0.2793), "s_uv": ("AP", 0.2755), "tile_uv_last": ("AP", 0.1993), "s_uv_last": ("AP", 0.1886),
                "tile_vd": ("AP50", 0.6669), "s_vd": ("AP50", 0.598)}
TOL = 0.005

SHARED = ["car", "truck", "bus"]
VAN_RULES = {"van=car": "car", "van=ignore": "ignore"}
CAR_ONLY = ["car"]
MERGED_MIN_SCORE = 0.05                            # lowest score the count-threshold grid can pick
for d in ("results", "figures", "caches", "merged", "logs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
print("Python", platform.python_version(), "| torch", torch.__version__, "| GPUs", N_GPU, "| SMOKE", SMOKE)
''')

nb.code(UAVDT_FETCH_CELL)

nb.code(CARPK_FETCH_CELL)

nb.code(r'''
# --- Splits. UAVDT and CARPK test images are listed here; nothing is scored until the setups are fixed below ---
ATTRS, OFFICIAL = data.read_uavdt_attrs(ATTR)
test_seqs = sorted(s for s, v in OFFICIAL.items() if v == "test")
assert not set(VAL_SEQS) & set(test_seqs)
SPLITS = {("uavdt", "val"): data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, VAL_SEQS if ON_KAGGLE else [s for s, v in OFFICIAL.items() if v == "train"], VAL_STRIDE, "val"),
          ("uavdt", "test"): data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, test_seqs, TEST_STRIDE, "test"),
          ("carpk", "test"): data.load_carpk(CARPK_ROOT, "test"),
          ("carpk", "train"): data.load_carpk(CARPK_ROOT, "train")}   # no model trains on CARPK; test is a single parking lot, train adds 6 more
for s in ("val", "test-dev"):
    d = VIS / f"VisDrone2019-DET-{s}"
    if not (d / "annotations").exists():
        d = data.download_visdrone(VIS, s)
    SPLITS[("visdrone", s)] = data.load_visdrone(d, s)
if ON_KAGGLE:
    assert {k: len(sp) for k, sp in SPLITS.items() if k in EXPECTED_N} == EXPECTED_N, {k: len(sp) for k, sp in SPLITS.items()}
    assert len(test_seqs) == EXPECTED_SEQS["test"]
    assert (len(SPLITS[("carpk", "test")]), len(SPLITS[("carpk", "train")])) == (459, 989)
for k, sp in SPLITS.items():
    missing = [r.key for r in sp.images if not Path(r.path).exists()]
    assert not missing, f"{k}: {len(missing)} frames without an image, e.g. {missing[:3]}"
if SMOKE:
    SPLITS = {k: sp.subset(SMOKE_N) for k, sp in SPLITS.items()}

# What the CPU phases need and cannot read without the images
idx = [dict(dataset=k[0], split=k[1], image=r.key, sequence=r.sequence, width=r.width, height=r.height,
            altitude=r.attrs.get("altitude"), weather=r.attrs.get("weather")) for k, sp in SPLITS.items() for r in sp.images]
pd.DataFrame(idx).to_csv(OUT / "results" / "image_index.csv", index=False)
STATS = pd.DataFrame({"/".join(k): data.split_stats(sp) for k, sp in SPLITS.items()}).T
STATS.to_csv(OUT / "results" / "split_stats.csv")
print(STATS[["images", "sequences", "objects", "objects_per_image", "tiny", "small", "medium_plus"]].round(3))
''')

nb.code(r'''
# --- Weights: found by exact name; class list and size checked before any GPU time ------------------------------
from ultralytics import YOLO, __version__ as UL_VERSION
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)
prov = {}
for tag, m in MODELS.items():
    hits = sorted(INPUT.rglob(m["file"]))
    assert hits, f"attach {m['file']}: see the table at the top of this notebook"
    m["path"] = hits[0]
    y = YOLO(str(hits[0]))
    names = [y.names[i] for i in sorted(y.names)]
    assert names == m["classes"], f"{m['file']}: classes {names}"
    n = sum(p.numel() for p in y.model.parameters())
    if ON_KAGGLE:
        assert m["params"][0] <= n <= m["params"][1], f"{m['file']}: {n / 1e6:.1f}M params is the wrong model"
    prov[tag] = dict(file=m["file"], name=m["name"], params=int(n), saved=y.ckpt.get("date"), ultralytics=y.ckpt.get("version"))
    del y
(OUT / "results" / "weights_provenance.json").write_text(json.dumps(prov, indent=1, default=str))
print(pd.DataFrame(prov).T[["file", "params", "saved", "ultralytics"]])

# Phase 0 caches, if attached, replace re-running the VisDrone models on VisDrone and add the Ensemble and zero-shot models
for name, tag in {"YOLO26m-Tile": "tile_vd", "YOLO26s": "s_vd", "YOLOE-26m-FT": "ft_vd", "YOLOE-26m-ZS": "zs_vd"}.items():
    for s in ("val", "test-dev"):
        hit = next(iter(sorted(p for p in INPUT.rglob(f"{name}__{s}.parquet"))), None)
        if hit:
            for suf in (".parquet", ".json"):
                shutil.copy(hit.with_suffix(suf), OUT / "caches" / f"{tag}__visdrone__{s}{suf}")
            print("reusing the Phase 0 cache:", hit.name)
''')

nb.code(r'''
# --- GPU jobs: one worker process per GPU; every job writes its own cache, so a rerun skips finished work ----------
WORKER = r"""
import os, pickle, sys
gpu, path = sys.argv[1], sys.argv[2]
if gpu != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
sys.path.insert(0, "dronekit_src")
import logging
import numpy as np
from dronekit import infer
from ultralytics import YOLO
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)
for j in pickle.load(open(path, "rb")):
    model = YOLO(str(j["weights"]))
    kw = dict(cuda=gpu != "cpu", half=gpu != "cpu", device=0 if gpu != "cpu" else "cpu")
    model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, half=kw["half"], device=kw["device"], verbose=False)
    infer.run_and_cache(model, j["split"], j["passes"], j["out"], meta=j["meta"], **kw)
    del model
"""
Path("dk_worker.py").write_text(WORKER)
PASS_COST = {"s512@1280": 380, "s512@1024": 240, "full@1024": 30, "full@1536": 40}      # rough ms per image, for balancing only

def cache_path(tag, key):
    return OUT / "caches" / f"{tag}__{key[0]}__{key[1]}"

def cached_passes(tag, key):
    js = cache_path(tag, key).with_suffix(".json")
    return set(json.loads(js.read_text()).get("passes", [])) if js.exists() else set()

def run_jobs(wanted):
    # wanted: [(tag, split key, passes)]. A cache missing a pass gets it in a second file, merged by load_raw.
    jobs = []
    for tag, key, passes in wanted:
        todo = [p for p in passes if p not in cached_passes(tag, key)]
        part = cache_path(tag, key).with_name(cache_path(tag, key).name + "__" + "+".join(todo))
        if todo and not part.with_suffix(".parquet").exists():
            jobs.append(dict(tag=tag, weights=MODELS[tag]["path"], split=SPLITS[key], passes=todo, out=part, meta=dict(model=MODELS[tag]["name"], ultralytics=UL_VERSION),
                             cost=MODELS[tag]["cost"] * len(SPLITS[key]) * sum(PASS_COST[p] for p in todo)))
    if not jobs:
        return
    n = max(N_GPU, 1)
    plan = [[] for _ in range(n)]; load = [0.0] * n
    for j in sorted(jobs, key=lambda j: -j["cost"]):
        i = int(np.argmin(load)); plan[i].append(j); load[i] += j["cost"]
    print(f"{len(jobs)} jobs; estimated minutes per GPU:", [round(x / 6e4, 1) for x in load])
    t0, procs = time.time(), []
    for i, p in enumerate(plan):
        if p:
            f = OUT / "logs" / f"jobs_{i}_{int(t0)}.pkl"; pickle.dump(p, open(f, "wb"))
            log = open(OUT / "logs" / f"worker_{i}_{int(t0)}.log", "w")
            env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            procs.append((subprocess.Popen([sys.executable, "dk_worker.py", str(i) if CUDA else "cpu", str(f)], stdout=log, stderr=subprocess.STDOUT, env=env), log))
    for pr, log in procs:
        rc = pr.wait(); log.close()
        print(Path(log.name).read_text(errors="replace")[-1500:])
        assert rc == 0, f"worker failed: see {log.name}"
    print(f"done in {(time.time() - t0) / 60:.1f} min")

def load_raw(tag, key):
    # every cache file of this model and split, passes joined
    raw, ms = {}, {}
    for pq in sorted((OUT / "caches").glob(f"{tag}__{key[0]}__{key[1]}*.parquet")):
        r, meta = infer.load_cache(pq.with_suffix(""))
        raw.update(r); ms.update(meta["ms_per_img"])
    return raw, dict(ms_per_img=ms)
''')

nb.md("## Step 3: whole vs sliced on UAVDT val, plus the checkpoint check")
nb.code(r'''
UV_VAL = ("uavdt", "val")
STEP3_PASSES = list(dict.fromkeys(WHOLE["passes"] + SLICED["passes"]))
run_jobs([(t, UV_VAL, STEP3_PASSES) for t in ("tile_uv", "s_uv")] +
         [(t, UV_VAL, WHOLE["passes"]) for t in ("tile_uv_last", "s_uv_last")])
sp = SPLITS[UV_VAL]; rows = []
for t in ("tile_uv", "s_uv"):
    raw, meta = load_raw(t, UV_VAL)
    for lab, cfg in (("whole", WHOLE), ("sliced", SLICED)):
        dets, mms = infer.apply_setup(raw, sp, cfg)
        m, pc, _ = score.evaluate(sp, dets)
        rows.append(dict(model=MODELS[t]["name"], tag=t, setup=lab, AP=m["AP"], AP50=m["AP50"], AP50_tiny=m.get("AP50_tiny"), AP50_small=m.get("AP50_small"), **{"AP50_medium+": m.get("AP50_medium+")},
                         **{f"AP50_{c}": v for c, v in pc.items()}, ms_per_img=round(infer.setup_ms(meta, cfg, mms), 1)))
STEP3 = pd.DataFrame(rows); STEP3.to_csv(OUT / "results" / "step3_whole_vs_sliced.csv", index=False)
print(STEP3.round(3).to_string())

CHOSEN = {}
for t in ("tile_uv", "s_uv"):
    r = STEP3[STEP3.tag == t].set_index("setup")
    CHOSEN[t] = "sliced" if r.loc["sliced", "AP"] - r.loc["whole", "AP"] >= MIN_GAIN else "whole"
SETUP_OF = {"tile_vd": infer.SETUPS["best_sliced"], "s_vd": infer.SETUPS["best_sliced"],
            "tile_uv": SLICED if CHOSEN["tile_uv"] == "sliced" else WHOLE, "s_uv": SLICED if CHOSEN["s_uv"] == "sliced" else WHOLE,
            "tile_uv_last": WHOLE, "s_uv_last": WHOLE}
SETUP_NAME = {"tile_vd": "best_sliced", "s_vd": "best_sliced", "tile_uv": CHOSEN["tile_uv"], "s_uv": CHOSEN["s_uv"], "tile_uv_last": "whole", "s_uv_last": "whole"}
print("val-chosen setups:", CHOSEN)
(OUT / "results" / "setups_chosen.json").write_text(json.dumps(dict(rule=f"sliced only if AP gain >= {MIN_GAIN} on UAVDT val", chosen=CHOSEN, setup_of=SETUP_NAME), indent=1))
''')

nb.code(r'''
# --- Every checkpoint must reproduce its known val score before any test image is scored ---------------------------
run_jobs([(t, ("visdrone", "val"), SETUP_OF[t]["passes"]) for t in ("tile_vd", "s_vd")])
checks = []
for t, (metric, want) in EXPECTED_VAL.items():
    key = ("visdrone", "val") if MODELS[t]["train"] == "VisDrone" else UV_VAL
    cfg = SETUP_OF[t] if MODELS[t]["train"] == "VisDrone" else WHOLE             # the retrain scans used full@1024
    m, _, _ = score.evaluate(SPLITS[key], infer.apply_setup(load_raw(t, key)[0], SPLITS[key], cfg)[0])
    checks.append(dict(model=MODELS[t]["name"], metric=metric, known=want, now=m[metric], diff=m[metric] - want))
CHECKS = pd.DataFrame(checks); CHECKS.to_csv(OUT / "results" / "checkpoint_check.csv", index=False)
print(CHECKS.round(4).to_string())
if ON_KAGGLE:
    bad = CHECKS[CHECKS["diff"].abs() > TOL]
    assert bad.empty, "these checkpoints do not reproduce their val score, so the file is wrong: stop and send this table\n" + bad.to_string()
''')

nb.md("## All remaining inference: test sets touched here, with setups already fixed")
nb.code(r'''
SHIFT_TARGETS = [("uavdt", "val"), ("uavdt", "test"), ("carpk", "test"), ("carpk", "train")]
wanted = []
for t in ("tile_vd", "s_vd"):
    wanted += [(t, k, SETUP_OF[t]["passes"]) for k in SHIFT_TARGETS + [("visdrone", "test-dev")]]
for t in ("tile_uv", "s_uv"):
    wanted += [(t, k, SETUP_OF[t]["passes"]) for k in [("uavdt", "test"), ("carpk", "test"), ("carpk", "train"), ("visdrone", "test-dev")]]
for t in ("tile_uv_last", "s_uv_last"):
    wanted += [(t, k, SETUP_OF[t]["passes"]) for k in [("uavdt", "test"), ("carpk", "test"), ("visdrone", "test-dev")]]
run_jobs(wanted)

# Merged detections per model and split, for the CPU phases
INDEX = []
for t in MODELS:
    for key, sp in SPLITS.items():
        if not list((OUT / "caches").glob(f"{t}__{key[0]}__{key[1]}*.parquet")):
            continue
        dets, mms = infer.apply_setup(load_raw(t, key)[0], sp, SETUP_OF[t])
        fr = [pd.DataFrame(d[d[:, 4] >= MERGED_MIN_SCORE], columns=["x1", "y1", "x2", "y2", "score", "cls"]).assign(image=k) for k, d in dets.items()]
        pd.concat(fr, ignore_index=True).astype(dict(cls="int16", image="category")).to_parquet(OUT / "merged" / f"{t}__{key[0]}__{key[1]}.parquet", compression="zstd", index=False)
        INDEX.append(dict(tag=t, model=MODELS[t]["name"], classes=",".join(MODELS[t]["classes"]), dataset=key[0], split=key[1], setup=SETUP_NAME[t], n_images=len(sp)))
pd.DataFrame(INDEX).to_csv(OUT / "merged" / "index.csv", index=False)
print(len(INDEX), "merged files;", sum(p.stat().st_size for p in (OUT / "merged").glob("*.parquet")) / 1e6, "MB")
''')

nb.md("## Cross-dataset matrix")
nb.code(r'''
def target_view(key, rule_tgt):
    # GT on the shared classes for one target and one van rule
    sp = SPLITS[key]
    if key[0] == "carpk":
        return data.remap_split(sp, {"car": "car"}, CAR_ONLY), CAR_ONLY
    if key[0] == "uavdt":
        return data.remap_split(sp, {c: c for c in SHARED}, SHARED), SHARED
    return data.remap_split(sp, {"car": "car", "van": rule_tgt, "truck": "truck", "bus": "bus"}, SHARED), SHARED

def det_map(tag, shared, rule_tgt):
    # van -> car, or van detections dropped; anything outside the shared classes is dropped
    m = {c: c for c in shared}
    if "van" in MODELS[tag]["classes"] and rule_tgt == "car":
        m["van"] = "car"
    return m

rows, pcs = [], []
for tag in MODELS:
    for key in [("visdrone", "test-dev"), ("uavdt", "test"), ("carpk", "test"), ("carpk", "train")]:
        if not list((OUT / "caches").glob(f"{tag}__{key[0]}__{key[1]}*.parquet")):
            continue
        raw, meta = load_raw(tag, key)
        dets, _ = infer.apply_setup(raw, SPLITS[key], SETUP_OF[tag])
        for rule, tgt in VAN_RULES.items():
            gt, shared = target_view(key, tgt)
            d = data.remap_dets(dets, MODELS[tag]["classes"], det_map(tag, shared, tgt), shared)
            m, pc, E = score.evaluate(gt, d)
            ap = score.per_class_ap(E, shared)
            rows.append(dict(trained_on=MODELS[tag]["train"], model=MODELS[tag]["name"], tag=tag, setup=SETUP_NAME[tag], van_rule=rule, target=f"{key[0]}/{key[1]}", n_images=len(gt), n_gt=gt.n_objects,
                             AP50=m["AP50"], **{"mAP50-95": m["AP"]}, AP75=m.get("AP75"), AP50_tiny=m.get("AP50_tiny"), AP50_small=m.get("AP50_small"), **{"AP50_medium+": m.get("AP50_medium+")},
                             **{f"AP50_{c}": pc.get(c, np.nan) for c in SHARED}, **{f"AP_{c}": ap.get(c, np.nan) for c in SHARED}))
MATRIX = pd.DataFrame(rows); MATRIX.to_csv(OUT / "results" / "transfer_matrix.csv", index=False)
wide = MATRIX[MATRIX.van_rule == "van=car"].pivot(index="model", columns="target", values="AP50").round(3)
print(wide.to_string()); wide.to_csv(OUT / "results" / "transfer_matrix_ap50_van_car.csv")
''')

nb.md("## Basic metrics for the UAVDT models (Phase 0 format)")
nb.code(r'''
rows, pcs = [], []
for tag in ("tile_uv", "s_uv"):
    conf, curves = None, {}
    for key in (UV_VAL, ("uavdt", "test")):
        sp = SPLITS[key]; raw, meta = load_raw(tag, key)
        dets, mms = infer.apply_setup(raw, sp, SETUP_OF[tag])
        m, pc, E = score.evaluate(sp, dets)
        conf = conf if conf is not None else (score.best_f1_conf(E) if E is not None else 0.25)     # val first; reused unchanged on test
        prc, pr25 = score.pr_at_conf(E, conf), score.pr_at_conf(E, 0.25)
        rows.append(dict(model=MODELS[tag]["name"], split=key[1], setup=SETUP_NAME[tag], n_images=len(sp), precision=prc["precision"], recall=prc["recall"], f1=prc["f1"], conf=conf,
                         P_at_0_25=pr25["precision"], R_at_0_25=pr25["recall"], mAP50=m["AP50"], **{"mAP50-95": m["AP"]}, AP75=m.get("AP75"), AP50_tiny=m.get("AP50_tiny"),
                         AP50_small=m.get("AP50_small"), **{"AP50_medium+": m.get("AP50_medium+")}, ms_per_img=round(infer.setup_ms(meta, SETUP_OF[tag], mms), 1)))
        pcs += [dict(model=MODELS[tag]["name"], split=key[1], cls=c, AP50=v, AP=score.per_class_ap(E, sp.class_names)[c]) for c, v in pc.items()]
        cm = score.confusion_matrix(sp, dets, conf)
        pd.DataFrame(cm, index=UV + ["background"], columns=UV + ["background"]).to_csv(OUT / "results" / f"confusion_{tag}__{key[1]}.csv")
        plots.plot_confusion(score.normalize_rows(cm), UV, OUT / "figures" / f"confusion_{tag}__{key[1]}.png", f"{MODELS[tag]['name']}, {key[1]}", conf)
        if E is not None:
            curves[key[1]] = score.pr_curves(E)
    if curves:
        plots.plot_pr_curves(curves, UV, OUT / "figures" / f"pr_{tag}.png", title=f"{MODELS[tag]['name']}: PR curves at IoU 0.5")
BASIC = pd.DataFrame(rows); BASIC.to_csv(OUT / "results" / "basic_metrics_uavdt.csv", index=False)
pd.DataFrame(pcs).to_csv(OUT / "results" / "per_class_uavdt.csv", index=False)
print(BASIC[["model", "split", "setup", "precision", "recall", "conf", "mAP50", "mAP50-95", "AP50_tiny", "AP50_small", "AP50_medium+", "ms_per_img"]].round(3).to_string())
''')

nb.code(r'''
# --- Bundle what comes back; the raw caches stay in this version's output ------------------------------------------
(OUT / "results" / "run_meta.json").write_text(json.dumps(dict(dronekit=dronekit.__version__, torch=torch.__version__, python=platform.python_version(), ultralytics=UL_VERSION,
    n_gpu=N_GPU, smoke=SMOKE, val_seqs=VAL_SEQS, strides=dict(val=VAL_STRIDE, test=TEST_STRIDE), min_gain=MIN_GAIN), indent=1))
with zipfile.ZipFile(OUT / "send_back.zip", "w", zipfile.ZIP_STORED) as z:
    for sub in ("results", "figures", "merged"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT))
print(f"send_back.zip: {(OUT / 'send_back.zip').stat().st_size / 1e6:.0f} MB")
if ON_KAGGLE:
    !du -sh {OUT} {OUT}/*
''')
nb.save("notebooks/phase1_eval.ipynb")
