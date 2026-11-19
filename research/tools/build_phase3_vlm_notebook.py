"""Build notebooks/phase3_vlm.ipynb: Qwen2.5-VL (4-bit, one per GPU) on the fixed count-set queries, systems B, C and D.
The query list from countset/vlm_queries.csv is embedded, so the notebook needs only the weights Dataset.

Run:  python tools/build_phase3_vlm_notebook.py
"""
from nb_common import CARPK_FETCH_CELL, ROOT, UAVDT_FETCH_CELL, Notebook

QUERIES = (ROOT / "countset/vlm_queries.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
assert "'''" not in QUERIES

nb = Notebook()
nb.md(r"""
# Phase 3: VLM counting (systems B, C, D)

Qwen2.5-VL-7B-Instruct, 4-bit, on the fixed queries from the 300-image subset (whole image; each image's most frequent class, plus car).
- **B direct:** one integer. **C localize-then-count:** one point per object. **D grounded:** the detector's boxes drawn and listed, then one integer.
- Strict parsing: an unparsable answer is kept and scored as wrong.
- The order is shuffled once with a fixed seed and each query runs B, D, C in turn, so stopping early leaves a random sample with all systems paired.
- Results are written after every answer. The time box stops new queries at `TIME_H`.

## How to run (about 3 h at most: the time box stops new queries after 2.5 h; timing is unknown until this run)
- **Inputs:** the same weights Dataset as `phase1_eval` (`yolo26m_tile_visdrone.pt`, `yolo26m_tile_uavdt.pt`). On a second account, share the Dataset with it first.
- **Secret:** `CARPK_PASSWORD`, attached to this notebook.
- **Settings:** GPU T4 x2, Internet ON. *Save Version → Save & Run All*. It can run at the same time as `phase1_eval`, on the other account.
- The first cells print a few real answers. If they look broken, stop and send them.
- **Send back** `/kaggle/working/phase3_vlm/send_back.zip`.

**Rerun (10 Nov 2026).** The first full run lost 115 of 346 queries to GPU out-of-memory errors: the notebook's own detector run left memory
held on GPU 0, and an out-of-memory error was not freed before the next query. `ONLY_QIDS` now reruns just those queries, with the same model,
prompts, image size and detector boxes; the GPU is freed before the workers start, and a query that still runs out of memory is retried on a
smaller image (the size used is logged per answer). About 45 minutes.
""")
nb.embed_package('"ultralytics==8.4.174" "sahi==0.12.8" pycocotools ensemble-boxes pytest gdown "transformers>=4.45" accelerate bitsandbytes qwen-vl-utils')

nb.code(r'''
# --- Config ----------------------------------------------------------------
import os, sys, io, json, time, shutil, pickle, zipfile, logging, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path("dronekit_src").resolve()))
import dronekit
from dronekit import data, infer, score, vlm
logging.getLogger("sahi").setLevel(logging.WARNING)

ON_KAGGLE = Path("/kaggle").exists()
INPUT = Path(os.environ.get("DK_INPUT", "/kaggle/input"))
RAW = Path(os.environ.get("DK_UAVDT_RAW", "/tmp/UAVDT/raw"))
VIS = Path(os.environ.get("DK_DATA", "/tmp/VisDrone"))
CARPK_RAW = Path(os.environ.get("DK_CARPK_RAW", "/tmp/CARPK"))
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
OUT = WORK / "phase3_vlm"
CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count() if CUDA else 0
SMOKE = not CUDA

MODEL_ID = os.environ.get("DK_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_PIXELS = 1_500_000            # the processor resizes larger images to about this; the size used is logged per answer
TIME_H = 2.5                      # no new queries after this
MAX_TOKENS = dict(B=16, D=16, C=1024)
SEED = 0
UAV_VAL = ["M0201", "M0402", "M0604", "M0902", "M1006"]
VD_CONF = 0.41                    # YOLO26m-Tile best sliced, val best-F1 (experiments/phase0/results/basic_metrics.csv)
VD, UV = dronekit.VISDRONE_CLASSES, data.UAVDT_CLASSES
QUERIES_CSV = """''' + QUERIES + r'''"""
Q = pd.read_csv(io.StringIO(QUERIES_CSV))
assert Q[["dataset", "image", "target"]].duplicated().sum() == 0
Q = Q.sample(frac=1, random_state=SEED).reset_index(drop=True)      # the query order of every run; qid is the position in it
Q["qid"] = np.arange(len(Q))
# Rerun of the queries that ran out of GPU memory in the first full run; [] runs every query. EXPECT checks the order is the same.
EXPECT = {0: '9999976_00000_d_0000034|car', 2: '9999973_00000_d_0000147|car', 8: '0000074_11362_d_0000022|car', 14: '9999938_00000_d_0000459|car', 18: '9999938_00000_d_0000073|car', 22: '0000353_02500_d_0000193|car', 23: '0000265_03500_d_0000008|car', 26: '9999986_00000_d_0000043|car', 30: '9999973_00000_d_0000063|car', 32: '0000074_15298_d_0000030|car', 38: '9999938_00000_d_0000278|car', 39: '0000272_01500_d_0000004|car', 42: '9999938_00000_d_0000301|pedestrian', 50: '0000074_03738_d_0000007|car', 52: '0000011_01955_d_0000004|car', 60: '9999938_00000_d_0000042|pedestrian', 62: '9999938_00000_d_0000062|car', 66: '9999938_00000_d_0000202|car', 79: '0000127_00265_d_0000120|car', 84: '20160524_GF2_00030|car', 86: '9999952_00000_d_0000281|car', 88: '20161029_NTU_00039|car', 90: '9999938_00000_d_0000062|pedestrian', 91: '0000074_15298_d_0000030|pedestrian', 92: '9999938_00000_d_0000035|car', 96: '9999952_00000_d_0000050|car', 104: '20160524_GF1_00087|car', 106: '9999938_00000_d_0000086|car', 108: '9999952_00000_d_0000036|car', 109: '0000073_01275_d_0000002|pedestrian', 110: '9999938_00000_d_0000493|car', 112: '9999938_00000_d_0000197|car', 116: '9999976_00000_d_0000030|car', 117: '9999938_00000_d_0000374|car', 122: '9999952_00000_d_0000001|motor', 128: '9999986_00000_d_0000034|pedestrian', 136: '9999979_00000_d_0000016|motor', 140: '9999938_00000_d_0000098|pedestrian', 142: '20161030_GF1_00078|car', 144: '9999996_00000_d_0000010|car', 148: '9999938_00000_d_0000070|car', 150: '9999938_00000_d_0000171|pedestrian', 152: '0000353_05000_d_0000198|car', 153: '9999952_00000_d_0000293|car', 154: '9999986_00000_d_0000032|car', 155: '9999938_00000_d_0000019|car', 162: '9999952_00000_d_0000112|car', 164: '9999938_00000_d_0000209|pedestrian', 166: '9999952_00000_d_0000338|car', 168: '0000314_01000_d_0000144|car', 170: '9999979_00000_d_0000027|car', 176: '9999986_00000_d_0000050|pedestrian', 178: '9999938_00000_d_0000042|car', 184: '20161029_NTU_00009|car', 188: '9999938_00000_d_0000440|car', 196: '9999952_00000_d_0000077|car', 197: '0000087_01140_d_0000004|pedestrian', 200: '9999973_00000_d_0000057|car', 201: '9999938_00000_d_0000073|pedestrian', 202: '9999976_00000_d_0000026|car', 203: '20161225_TPZ_00290|car', 205: '20160524_GF2_00090|car', 211: '20161029_NTU_00161|car', 214: '0000074_11362_d_0000022|pedestrian', 215: '0000074_09738_d_0000019|pedestrian', 216: '9999976_00000_d_0000021|pedestrian', 218: '9999986_00000_d_0000024|car', 220: '9999952_00000_d_0000318|car', 222: '0000074_07850_d_0000015|pedestrian', 235: '20161225_TPZ_00377|car', 236: '20160524_GF2_00129|car', 240: '9999952_00000_d_0000064|car', 242: '9999976_00000_d_0000034|pedestrian', 243: '9999979_00000_d_0000023|car', 244: '9999986_00000_d_0000034|car', 246: '9999938_00000_d_0000083|car', 249: '9999947_00000_d_0000019|motor', 250: '9999952_00000_d_0000142|car', 252: '0000074_10218_d_0000020|car', 254: '9999952_00000_d_0000100|car', 255: '0000074_10218_d_0000020|pedestrian', 258: '0000272_02500_d_0000006|car', 261: '0000074_02723_d_0000005|pedestrian', 262: '9999938_00000_d_0000496|car', 264: '9999976_00000_d_0000005|car', 266: '9999963_00000_d_0000031|car', 270: '0000353_03500_d_0000195|car', 274: '0000074_03738_d_0000007|pedestrian', 276: '9999963_00000_d_0000065|car', 278: '9999947_00000_d_0000027|pedestrian', 279: '0000302_03866_d_0000117|car', 282: '9999952_00000_d_0000330|pedestrian', 286: '9999952_00000_d_0000034|car', 290: '9999963_00000_d_0000045|motor', 293: '9999952_00000_d_0000075|car', 294: '9999963_00000_d_0000041|car', 304: '9999938_00000_d_0000348|car', 305: '0000278_02351_d_0000002|car', 306: '20161225_TPZ_00036|car', 307: '20161225_TPZ_00064|car', 310: '9999976_00000_d_0000021|car', 312: '9999973_00000_d_0000131|car', 314: '9999952_00000_d_0000160|car', 316: '0000073_01275_d_0000002|car', 325: '9999979_00000_d_0000026|car', 326: '0000063_12500_d_0000014|pedestrian', 327: '9999979_00000_d_0000044|car', 330: '9999973_00000_d_0000147|motor', 334: '9999952_00000_d_0000268|car', 336: '9999952_00000_d_0000103|pedestrian', 338: '20160524_GF1_00111|car', 340: '9999938_00000_d_0000447|car', 342: '0000074_15793_d_0000031|pedestrian', 344: '20161225_TPZ_00433|car', 345: '0000063_02235_d_0000003|pedestrian'}
ONLY_QIDS = sorted(EXPECT)
FALLBACK_PIXELS = [1_000_000, 700_000]   # only after an out-of-memory error at MAX_PIXELS, then at the same size again
if ONLY_QIDS:
    got = {int(q.qid): f"{q.image}|{q.target}" for q in Q.itertuples() if q.qid in EXPECT}
    assert got == EXPECT, "query order differs from the first run: stop and send this"
    Q = Q[Q.qid.isin(ONLY_QIDS)].reset_index(drop=True)
for d in ("results", "figures", "logs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
print(len(Q), "queries", Q.dataset.value_counts().to_dict(), "| GPUs", N_GPU, "| SMOKE", SMOKE)
''')
nb.code(UAVDT_FETCH_CELL)
nb.code(CARPK_FETCH_CELL)

nb.code(r'''
# --- The query images, and a detector run for system D -------------------------------------------------------------
from ultralytics import YOLO
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)
ATTRS, OFFICIAL = data.read_uavdt_attrs(ATTR)
vis = data.load_visdrone(VIS / "VisDrone2019-DET-test-dev" if (VIS / "VisDrone2019-DET-test-dev/annotations").exists() else data.download_visdrone(VIS, "test-dev"), "test-dev")
uv_seqs = sorted(s for s, v in OFFICIAL.items() if v == "test")
uv = data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, uv_seqs, 10, "test")
cp = data.load_carpk(CARPK_ROOT, "test").images + data.load_carpk(CARPK_ROOT, "train").images
REC = {("visdrone", r.key): r for r in vis.images} | {("uavdt", r.key): r for r in uv.images} | {("carpk", r.key): r for r in cp}
if SMOKE:
    Q = Q[[k in REC for k in zip(Q.dataset, Q.image)]].reset_index(drop=True) if any(k in REC for k in zip(Q.dataset, Q.image)) else Q.iloc[:0]
    if not len(Q):
        keys = list(REC)[:4]
        Q = pd.DataFrame([dict(dataset=d, split="x", image=i, sequence=REC[(d, i)].sequence, target="car", count=len(REC[(d, i)].boxes), count_bin="x", size_bin="x", density_bin="x", qid=n) for n, (d, i) in enumerate(keys)])
missing = [k for k in zip(Q.dataset, Q.image) if k not in REC]
assert not missing, f"{len(missing)} query images are missing, e.g. {missing[:3]}: stop and send the log"

def weights(name):
    hits = sorted(INPUT.rglob(name))
    assert hits, f"attach {name} (the weights Dataset from phase1_eval)"
    return str(hits[0])

device = dict(cuda=CUDA, half=CUDA, device=0 if CUDA else "cpu")
def run(model_file, recs, passes, setup):
    sp = data.Split("x", "x", [], recs)
    tmp = OUT / "logs" / f"d_{Path(model_file).stem}_{len(recs)}"
    infer.run_and_cache(YOLO(model_file), sp, passes, tmp, **device)
    return infer.apply_setup(infer.load_cache(tmp)[0], sp, setup)[0]

W_VD, W_UV = weights("yolo26m_tile_visdrone.pt"), weights("yolo26m_tile_uavdt.pt")
keys = sorted(set(zip(Q.dataset, Q.image)))
# UAVDT threshold: best F1 on UAVDT val at the selection pass; VisDrone threshold from Phase 0
uv_val = data.load_uavdt(IMG_ROOT, GT_DIR, ATTR, UAV_VAL if ON_KAGGLE else [s for s, v in OFFICIAL.items() if v == "train"], 10, "val")
WHOLE = dict(passes=["full@1024"], method=None)
d_val = run(W_UV, uv_val.images if ON_KAGGLE else uv_val.images[:3], ["full@1024"], WHOLE)
_, _, E = score.evaluate(uv_val if ON_KAGGLE else uv_val.subset(3), d_val)
UV_CONF = score.best_f1_conf(E) if E is not None else 0.25
print("thresholds: VisDrone model", VD_CONF, "| UAVDT model", UV_CONF)
DET = {}
vd_recs = [REC[k] for k in keys if k[0] != "uavdt"]
uv_recs = [REC[k] for k in keys if k[0] == "uavdt"]
BEST = infer.SETUPS["best_sliced"]
if vd_recs:
    DET.update(run(W_VD, vd_recs, BEST["passes"], BEST))
if uv_recs:
    DET.update(run(W_UV, uv_recs, ["full@1024"], WHOLE))

def boxes_for(dataset, image, target):
    # detector candidates for one query, highest score first. CARPK has no model of its own: the VisDrone model, with van as car.
    d = DET[image]
    names, conf = (UV, UV_CONF) if dataset == "uavdt" else (VD, VD_CONF)
    ids = [names.index(target)] + ([names.index("van")] if dataset == "carpk" and target == "car" else [])
    d = d[np.isin(d[:, 5].astype(int), ids) & (d[:, 4] >= conf)]
    return d[np.argsort(-d[:, 4])][:, :5].tolist()
''')

nb.code(r'''
# --- Worker: one model per GPU, append after every answer, skip what is already written ---------------------------
import gc
gc.collect()
if CUDA:
    torch.cuda.empty_cache()                 # the detector above ran on GPU 0; give that memory back before the VLM workers start
    print("GPU 0 memory still reserved by this kernel:", round(torch.cuda.memory_reserved(0) / 2**30, 2), "GB")
JOB = []
for q in Q.itertuples():
    r = REC[(q.dataset, q.image)]
    JOB.append(dict(qid=int(q.qid), dataset=q.dataset, image=q.image, sequence=q.sequence, target=q.target, count=int(q.count), path=str(r.path), boxes=boxes_for(q.dataset, q.image, q.target), width=r.width, height=r.height))
pickle.dump(JOB, open(OUT / "logs" / "jobs.pkl", "wb"))
pd.DataFrame([dict(qid=j["qid"], dataset=j["dataset"], image=j["image"], target=j["target"], count=j["count"], detector_count=len(j["boxes"])) for j in JOB]).to_csv(OUT / "results" / "detector_counts_for_d.csv", index=False)

WORKER = r"""
import gc, json, os, pickle, re, sys, time
from pathlib import Path
gpu, n_gpu, out, model_id, max_pixels, deadline, limit = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
fallback = [int(v) for v in sys.argv[8].split(",") if v]
os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
sys.path.insert(0, "dronekit_src")
import cv2, torch
from PIL import Image
from dronekit import vlm
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
MAX_TOKENS = dict(B=16, D=16, C=1024)
kw = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16), device_map={"": 0}) if gpu != "cpu" else {}
model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
proc = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)

def ask(path, prompt, n, px):
    msgs = [dict(role="user", content=[dict(type="image", image=str(path), max_pixels=px), dict(type="text", text=prompt)])]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(msgs)
    x = proc(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        y = model.generate(**x, max_new_tokens=n, do_sample=False)
    new = y[0, x.input_ids.shape[1]:]
    grid = x.image_grid_thw[0].tolist() if "image_grid_thw" in x else [0, 0, 0]
    return proc.decode(new, skip_special_tokens=True), int(len(new)), [int(grid[2]) * 14, int(grid[1]) * 14]

def grounded_image(j):
    im = cv2.imread(j["path"])
    for i, (x1, y1, x2, y2, s) in enumerate(j["boxes"]):
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if len(j["boxes"]) <= 60:
            cv2.putText(im, str(i + 1), (int(x1), max(int(y1) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    f = out / "logs" / f"g_{gpu}.jpg"
    cv2.imwrite(str(f), im)
    return f

fn = out / "results" / f"answers_{gpu}.jsonl"
done = set()
if fn.exists():
    done = {(r["qid"], r["system"]) for r in map(json.loads, fn.read_text().splitlines())}
jobs = pickle.load(open(out / "logs" / "jobs.pkl", "rb"))
jobs = [j for k, j in enumerate(jobs) if k % n_gpu == (0 if gpu == "cpu" else int(gpu))][:limit or None]      # by position, so a rerun subset is split evenly
with open(fn, "a") as f:
    for j in jobs:
        if time.time() > deadline:
            print("time box reached at query", j["qid"]); break
        for sysname in ("B", "D", "C"):
            if (j["qid"], sysname) in done:
                continue
            prompt = dict(B=vlm.prompt_direct, C=vlm.prompt_localize)[sysname](j["target"]) if sysname != "D" else vlm.prompt_grounded(j["target"], j["boxes"])
            path = j["path"] if sysname != "D" else grounded_image(j)
            t0 = time.time()
            ans, n_new, size, err, used = "", 0, [0, 0], "oom", None
            for px in [max_pixels] + fallback:
                oom = False
                try:
                    ans, n_new, size = ask(path, prompt, MAX_TOKENS[sysname], px); err, used = None, px
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if not oom:
                    break
                gc.collect(); torch.cuda.empty_cache()     # outside the except block, so the failed step's tensors are really released
            parsed = vlm.parse_points(ans) if sysname == "C" else vlm.parse_int(ans)
            f.write(json.dumps(dict(qid=j["qid"], system=sysname, dataset=j["dataset"], image=j["image"], sequence=j["sequence"], target=j["target"], count=j["count"], parsed=parsed, truncated=n_new >= MAX_TOKENS[sysname],
                                    raw=ans[:6000], seconds=round(time.time() - t0, 2), new_tokens=n_new, input_hw=size, max_pixels_used=used, error=err)) + "\n")
            f.flush()
print("worker done")
"""
Path("dk_vlm_worker.py").write_text(WORKER)

def launch(limit, deadline):
    procs = []
    for i in range(max(N_GPU, 1)):
        gpu = str(i) if CUDA else "cpu"
        log = open(OUT / "logs" / f"vlm_{gpu}.log", "a")
        procs.append((subprocess.Popen([sys.executable, "dk_vlm_worker.py", gpu, str(max(N_GPU, 1)), str(OUT), MODEL_ID, str(MAX_PIXELS), str(deadline), str(limit), ",".join(map(str, FALLBACK_PIXELS))], stdout=log, stderr=subprocess.STDOUT), log))
    for p, log in procs:
        rc = p.wait(); log.close()
        print(Path(log.name).read_text(errors="replace")[-1200:])
        assert rc == 0, f"worker failed: see {log.name}"

def answers():
    rows = [json.loads(l) for p in sorted((OUT / "results").glob("answers_*.jsonl")) for l in p.read_text().splitlines()]
    return pd.DataFrame(rows)
''')

nb.md("## A few real answers first")
nb.code(r'''
t0 = time.time()
launch(2, time.time() + 1800)
A = answers()
print(A[["system", "target", "count", "parsed", "seconds", "new_tokens", "input_hw", "raw"]].assign(raw=lambda d: d.raw.str[:80]).to_string())
print(f"first answers took {(time.time() - t0) / 60:.1f} min, including loading the model")
''')

nb.md("## Full run (GPU only)")
nb.code(r'''
t0 = time.time()
if not SMOKE:
    launch(0, t0 + TIME_H * 3600)
A = answers()
A.to_csv(OUT / "results" / "vlm_answers.csv", index=False)
s = A.groupby("system").agg(n=("qid", "size"), parsed=("parsed", lambda v: v.notna().mean()), truncated=("truncated", "mean"), seconds=("seconds", "mean"), total_h=("seconds", lambda v: v.sum() / 3600))
s.to_csv(OUT / "results" / "vlm_run_summary.csv")
print(s.round(3).to_string())
(OUT / "results" / "run_meta.json").write_text(json.dumps(dict(model=MODEL_ID, max_pixels=MAX_PIXELS, quant="4-bit nf4", n_gpu=N_GPU, time_box_h=TIME_H, wall_hours=(time.time() - t0) / 3600, queries_planned=len(Q), queries_done=int(A.qid.nunique()), only_qids=ONLY_QIDS, fallback_pixels=FALLBACK_PIXELS,
    max_tokens=MAX_TOKENS, vd_conf=VD_CONF, uv_conf=UV_CONF, smoke=SMOKE, python=platform.python_version(), torch=torch.__version__), indent=1))
with zipfile.ZipFile(OUT / "send_back.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((OUT / "results").rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))
print("send_back.zip written;", A.qid.nunique(), "of", len(Q), "queries answered")
''')
nb.save("notebooks/phase3_vlm.ipynb")
