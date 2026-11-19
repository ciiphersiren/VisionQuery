"""Generate the Kaggle notebooks (run: python kaggle/make_notebooks.py)."""

import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = r'''# --- Setup: code, packages, data -------------------------------------------------
# 1) Upload skyquery.zip as a Kaggle Dataset named "skyquery-code" and add it to this notebook.
# 2) Settings -> Accelerator: GPU T4 x2, Internet: on.
# 3) Optional (OWLv2 / Grounding DINO): Add-ons -> Secrets -> HF_TOKEN (read-only token).
import os, subprocess, sys, glob, shutil
CODE = glob.glob("/kaggle/input/skyquery-code/**/pyproject.toml", recursive=True)
assert CODE, "Add the skyquery-code dataset to this notebook"
SRC = os.path.dirname(CODE[0])
WORK = "/kaggle/working/skyquery"
if not os.path.exists(WORK):
    shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns("node_modules", "data", "cache"))
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval]", "transformers>=4.57", "git+https://github.com/ultralytics/CLIP.git"], check=True)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception:
    print("No HF_TOKEN secret: OWLv2 / Grounding DINO will try anonymous downloads")

os.makedirs("data", exist_ok=True); os.makedirs("cache", exist_ok=True)
BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
def fetch(split):
    d = f"data/VisDrone2019-DET-{split}"
    if not os.path.exists(d):
        subprocess.run(f"curl -sSL -o /tmp/{split}.zip {BASE}VisDrone2019-DET-{split}.zip && unzip -q /tmp/{split}.zip -d data && rm /tmp/{split}.zip", shell=True, check=True)
    return d
'''

E1_CELLS = [
    ("markdown", "# SkyQuery E1: zero-shot prompt-based detectors on VisDrone\nPre-registered in the team's pre-registration notes (round 3). Runs every open-vocabulary model with two prompt sets on val, and with the class-name prompts on test-dev, then scores them against the fine-tuned YOLO26s with the same COCO harness and sequence-bootstrap intervals. Expect about 2–3 hours on 2×T4."),
    ("code", SETUP + 'VAL, TEST = fetch("val"), fetch("test-dev")\n'),
    ("code", r'''# Fine-tuned reference (FI1280) on GPU, cached in the same format
for tag, split in (("val", VAL), ("testdev", TEST)):
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.cache_preds", "--split", split, "--scale", "1280",
                    "--device", "0", "--out", f"cache/{tag}_1280.npz"], check=True)
'''),
    ("code", r'''MODELS = ["yoloe-26s", "yoloe-26m", "yoloe-26l", "yolo-world-v2-s", "yolo-world-v2-l",
          "owlv2-base", "owlv2-large", "grounding-dino-tiny", "grounding-dino-base"]
def run(model, prompts, tag, split):
    out = f"cache/{tag}_{model}_{prompts}.npz"
    dev = "0" if model.startswith(("yolo")) else "cuda"
    r = subprocess.run([sys.executable, "-m", "skyquery.calibrate.cache_ovd", "--split", split, "--model", model,
                        "--prompts", prompts, "--device", dev, "--out", out])
    return out if r.returncode == 0 else None

done = {"val": [], "testdev": []}
for m in MODELS:
    for p in ("names", "described"):
        o = run(m, p, "val", VAL)
        if o: done["val"].append(f"{m}:{p}={o}")
    o = run(m, "names", "testdev", TEST)
    if o: done["testdev"].append(f"{m}:names={o}")
print(done)
'''),
    ("code", r'''for tag, split in (("val", VAL), ("testdev", TEST)):
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.ovd_eval", "e1", "--split", split, "--runs", *done[tag]], check=True)
'''),
    ("code", r'''# Send these back: reports/ovd_e1_*.json (small) and, if you can, the caches
subprocess.run("zip -qr /kaggle/working/e1_results.zip reports cache -i 'reports/*.json' 'cache/*.npz'", shell=True)
print("Download /kaggle/working/e1_results.zip from the Output panel")
'''),
]

E2_CELLS = [
    ("markdown", "# SkyQuery E2: held-out classes\nRetrains YOLO26s with **bus, bicycle and awning-tricycle** removed from the labels (same recipe as the original model: 640 px tiles + full frames, 40 epochs, seed 42), then asks how well a zero-shot detector covers the classes the trained model has never seen. About 7 hours on 2×T4; the training cell resumes if the session restarts."),
    ("code", SETUP + 'TRAIN, VAL, TEST = fetch("train"), fetch("val"), fetch("test-dev")\nHOLD = ["bus", "bicycle", "awning-tricycle"]\n'),
    ("code", r'''DS = "/kaggle/working/vd_heldout"
if not os.path.exists(f"{DS}/data.yaml"):
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.tiles", "--src", TRAIN, "--val", VAL, "--out", DS, "--holdout", *HOLD], check=True)
print(open(f"{DS}/holdout.json").read())
'''),
    ("code", r'''from ultralytics import YOLO
RUN = "/kaggle/working/runs/heldout7"
DONE = f"{RUN}/weights/TRAINING_DONE"
if not os.path.exists(DONE):
    last = f"{RUN}/weights/last.pt"
    if os.path.exists(last):
        try:
            YOLO(last).train(resume=True)          # session restarted mid-training
        except Exception as e:
            print("resume not possible:", e)       # e.g. already finished
    else:
        YOLO("yolo26s.pt").train(data=f"{DS}/data.yaml", epochs=40, patience=15, batch=32, imgsz=640, device="0,1", workers=4,
                                 project="/kaggle/working/runs", name="heldout7", exist_ok=True, seed=42, cos_lr=True,
                                 close_mosaic=10, amp=True, max_det=902, plots=True)
    open(DONE, "w").write("ok")
'''),
    ("code", r'''BEST = f"{RUN}/weights/best.pt"
for tag, split in (("val", VAL), ("testdev", TEST)):
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.cache_preds", "--split", split, "--scale", "1280", "--device", "0",
                    "--out", f"cache/{tag}_1280.npz"], check=True)                     # 10-class oracle (your weights)
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.cache_preds", "--split", split, "--scale", "1280", "--device", "0",
                    "--weights", BEST, "--out", f"cache/{tag}_heldout7_1280.npz"], check=True)
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.cache_ovd", "--split", split, "--model", "yoloe-26l", "--prompts", "names",
                    "--classes", *HOLD, "--device", "0", "--out", f"cache/{tag}_yoloe26l_heldout.npz"], check=True)
    subprocess.run([sys.executable, "-m", "skyquery.calibrate.cache_ovd", "--split", split, "--model", "yoloe-26l", "--prompts", "names",
                    "--device", "0", "--out", f"cache/{tag}_yoloe26l_names.npz"], check=True)
'''),
    ("code", r'''subprocess.run([sys.executable, "-m", "skyquery.calibrate.ovd_eval", "e2", "--val", VAL, "--test", TEST, "--holdout", *HOLD], check=True)
shutil.copy(BEST, "/kaggle/working/yolo26s_heldout7.pt")
subprocess.run("zip -qr /kaggle/working/e2_results.zip reports -i 'reports/*.json'", shell=True)
print("Download e2_results.zip and yolo26s_heldout7.pt from the Output panel")
'''),
]


E3_SETUP = r"""# --- Settings ---------------------------------------------------------------------
QUICK = False          # True = 20-minute dry run on a few frames. Do this once interactively before the overnight run.
MODEL_SIZE = "s"       # "s": fast, what the app uses. "m": better but ~2x slower to train and to run.
TEACHER = "sam3"       # "sam3" (best; accept the licence at huggingface.co/facebook/sam3 first) or "owlv2"
TOTAL_HOURS = 11.0     # Kaggle stops a session at 12 h, keep a margin
LABEL_HOURS = 3.0      # teacher labelling on both GPUs; frames it doesn't reach still train on the human labels
EVAL_HOURS = 0.6
TAU = 0.5              # teacher score needed for a box to become a training label

import time
T0 = time.time()
def hours_left():
    return TOTAL_HOURS - (time.time() - T0) / 3600

# --- Code, packages, secrets ----------------------------------------------------------
# Needs: the skyquery-code dataset added, GPU T4 x2, Internet on, and an HF_TOKEN secret (read-only token).
import os, subprocess, sys, glob, shutil, json
CODE = glob.glob("/kaggle/input/skyquery-code/**/pyproject.toml", recursive=True)
assert CODE, "Add the skyquery-code dataset to this notebook"
WORK = "/kaggle/working/skyquery"
if not os.path.exists(WORK):
    shutil.copytree(os.path.dirname(CODE[0]), WORK, ignore=shutil.ignore_patterns("node_modules", "data", "cache", "frontend"))
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[eval]", "transformers==5.20.2", "ultralytics==8.4.182"], check=True)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception:
    print("No HF_TOKEN secret: SAM 3 is gated, so the teacher will fall back to OWLv2")

SCRATCH = "/tmp/e3"                      # big, temporary files (not saved as notebook output)
os.makedirs(f"{SCRATCH}/data", exist_ok=True); os.makedirs("cache", exist_ok=True); os.makedirs("reports", exist_ok=True)
BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
def fetch(split):
    d = f"{SCRATCH}/data/VisDrone2019-DET-{split}"
    if not os.path.exists(d):
        subprocess.run(f"curl -sSL -o /tmp/{split}.zip {BASE}VisDrone2019-DET-{split}.zip && unzip -q /tmp/{split}.zip -d {SCRATCH}/data && rm /tmp/{split}.zip", shell=True, check=True)
    return d
TRAIN, VAL = fetch("train"), fetch("val")
print(len(os.listdir(f"{TRAIN}/images")), "train frames,", len(os.listdir(f"{VAL}/images")), "val frames")
"""

E3_CELLS = [
    ("markdown", """# SkyQuery E3: teach YOLOE drone-view things, keep its open vocabulary

**What this does overnight.** Zero-shot YOLOE misses things VisDrone never labelled (trees, buildings, street
lamps...). Here SAM 3, the strongest open text-prompted detector we know of but far too slow for the app,
labels drone frames for 35 extra concepts. YOLOE-26 then learns from those labels plus VisDrone's own. It is
trained with *text prompts* (not the usual fine-tune that freezes the class list), so any word still works
afterwards. 8 concepts are held out completely to test exactly that.

| Step | Time (2×T4) |
|---|---|
| setup + downloads | ~15 min |
| SAM 3 labels all val frames, then as many train frames as fit | `LABEL_HOURS` (3 h) |
| build 640 px tiles | ~15 min |
| train YOLOE (time-boxed, both GPUs) | whatever is left, ~6 h |
| score zero-shot vs tuned on val | ~30 min |

**Before you start:** (1) accept the SAM 3 licence at huggingface.co/facebook/sam3 with your HF account;
(2) add a read-only `HF_TOKEN` under Add-ons → Secrets (never paste it in a cell); (3) GPU T4 ×2, Internet on;
(4) add the `skyquery-code` dataset. Run once with `QUICK = True` (about 20 min) to check everything works,
then set `QUICK = False` and use **Save Version → Save & Run All (Commit)** so it runs with the browser closed.

Pre-registered in the team's pre-registration notes, round 4."""),
    ("code", E3_SETUP),
    ("code", r"""# --- Models: teacher access check, YOLOE weights, YOLOE's text encoder -------------------
if TEACHER == "sam3":
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("facebook/sam3", token=os.environ.get("HF_TOKEN"), allow_patterns=["*.json", "*.safetensors", "*.txt"])
        print("SAM 3 downloaded")
    except Exception as e:
        print("SAM 3 NOT available:", type(e).__name__, str(e)[:300])
        print("-> accept the licence at huggingface.co/facebook/sam3 and add HF_TOKEN; using OWLv2 for now")
        TEACHER = "owlv2"
from ultralytics import YOLOE
_m = YOLOE(f"yoloe-26{MODEL_SIZE}-seg.pt")
_m.get_text_pe(["tree"])                 # downloads the MobileCLIP text encoder into this folder once
del _m
print("teacher:", TEACHER, "| student: YOLOE-26" + MODEL_SIZE, "| hours left:", round(hours_left(), 2))
"""),
    ("code", r"""# --- Teacher labels: all val frames first, then train frames until the deadline ---------
def run_on_gpus(jobs, deadline):
    procs = []
    for gpu, (args, log) in enumerate(jobs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, "-m", "skyquery.ovtrain.label", *args, "--device", "cuda:0", "--teacher", TEACHER,
               "--deadline", str(deadline), "--shard", str(gpu), "--nshards", str(len(jobs))]
        procs.append((subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
    while any(p.poll() is None for p, _ in procs):
        time.sleep(300 if not QUICK else 20)
        for _, log in procs:
            lines = open(log).read().strip().splitlines()
            print(lines[-1] if lines else f"{log}: starting")
    for p, log in procs:
        print(open(log).read()[-1500:])
        if p.returncode:
            raise RuntimeError(f"labelling failed, see {log}")

LIM = ["--limit", "16"] if QUICK else []
label_end = time.time() + (0.25 if QUICK else LABEL_HOURS) * 3600
run_on_gpus([(["--split", VAL, "--group", "all", "--out", f"cache/teacher_val_{g}.npz", *LIM], f"/tmp/label_val_{g}.log") for g in (0, 1)], label_end)
run_on_gpus([(["--split", TRAIN, "--group", "seen", "--out", f"cache/teacher_train_{g}.npz", *LIM], f"/tmp/label_train_{g}.log") for g in (0, 1)], label_end)
print("hours left:", round(hours_left(), 2))
"""),
    ("code", r"""# --- Build the training sets ----------------------------------------------------------
DS = f"{SCRATCH}/ds"
cmd = [sys.executable, "-m", "skyquery.ovtrain.build", "--train", TRAIN, "--val", VAL, "--out", DS, "--tau", str(TAU), "--workers", "4",
       "--teacher-train", "cache/teacher_train_*.npz", "--teacher-val", "cache/teacher_val_*.npz"] + (["--limit", "16"] if QUICK else [])
subprocess.run(cmd, check=True)
stats = json.load(open(f"{DS}/stats.json"))
print({k: v for k, v in stats.items() if k != "train_boxes_per_class"})
print("teacher boxes kept (train):", {k: v for k, v in stats["train_boxes_per_class"].items() if v})
"""),
    ("code", r"""# --- Train (time-boxed; falls back to one GPU if two-GPU training fails) ----------------
OUT_W = f"/kaggle/working/yoloe-26{MODEL_SIZE}-drone-ov.pt"
train_hours = 0 if QUICK else max(0.5, hours_left() - EVAL_HOURS - 0.25)
print(f"training for {train_hours:.2f} h")
cmd = [sys.executable, "-m", "skyquery.ovtrain.train", "--data", f"{DS}/data.json", "--size", MODEL_SIZE, "--hours", str(train_hours),
       "--device", "0" if QUICK else "0,1", "--project", "/kaggle/working/runs", "--name", "e3_ov", "--out", OUT_W, "--workers", "4"]
if QUICK:
    cmd += ["--epochs", "1"]
subprocess.run(cmd, check=True)
print("hours left:", round(hours_left(), 2))
"""),
    ("code", r"""# --- Score zero-shot vs tuned on val, write the app config -----------------------------
name = f"weights/yoloe-26{MODEL_SIZE}-drone-ov.pt"
cmd = [sys.executable, "-m", "skyquery.ovtrain.evaluate", "--val", VAL, "--teacher-val", "cache/teacher_val_0.npz", "cache/teacher_val_1.npz",
       "--model", f"zeroshot=yoloe-26{MODEL_SIZE}-seg.pt", "--model", f"tuned={OUT_W}", "--device", "0", "--tau", str(TAU),
       "--out", "reports/ovd_e3_val.json", "--write-config", "configs/ovd_drone.json", "--weights-name", name]
if QUICK:
    cmd += ["--limit", "16", "--boot", "20"]
subprocess.run(cmd, check=True)
"""),
    ("code", r"""# --- Package: download these from the Output tab --------------------------------------
shutil.copy("reports/ovd_e3_val.json", "/kaggle/working/ovd_e3_val.json")
shutil.copy("configs/ovd_drone.json", "/kaggle/working/ovd_drone.json")
shutil.copy(f"{DS}/stats.json", "reports/e3_dataset_stats.json")
subprocess.run("zip -qr /kaggle/working/e3_results.zip reports/ovd_e3_val.json reports/e3_dataset_stats.json configs/ovd_drone.json cache/teacher_val_*.npz cache/teacher_train_*.npz", shell=True)
for f in ("e3_results.zip", os.path.basename(OUT_W)):
    print(f, round(os.path.getsize(f"/kaggle/working/{f}") / 1e6, 1), "MB")
print("Put", os.path.basename(OUT_W), "in skyquery/weights/ and ovd_drone.json in skyquery/configs/; the app picks them up on restart.")
print("total hours:", round((time.time() - T0) / 3600, 2))
"""),
]


# --------------------------------------------------------------------------- E3b: E3 with frozen text alignment
E3B_INTRO = """# SkyQuery E3b: same as E3, with the text-alignment layers frozen

E3 taught YOLOE 35 drone concepts (tree 1 → 52 AP) but it stopped answering to every word it wasn't taught. The
diagnosis (`reports/ovd_e3_wise.json`): the contrastive head's BatchNorm statistics and the text adapter (RepRTA)
drifted. E3b changes **only that**: those layers stay at their zero-shot values. Same SAM 3 labels (reused from E3,
no relabelling), same vocabulary, same held-out words, same training time (6.9 h), so any difference is the freeze.

**Before you start:** upload your `e3_results.zip` as a Kaggle dataset named `skyquery-e3-labels` and add it; add the
new `skyquery-code` (it includes the E3 weights for the comparison); GPU T4 ×2; Internet on. No HF token needed.
Run once with `QUICK = True`, then Save & Run All (Commit). About 7.5 hours.

Pre-registered in the team's pre-registration notes, round 5."""

_set = E3_SETUP.replace(
    'TEACHER = "sam3"       # "sam3" (best; accept the licence at huggingface.co/facebook/sam3 first) or "owlv2"',
    'TEACHER = "sam3"       # only used if LABEL_HOURS > 0 (labels are reused from E3)')
_set = _set.replace(
    "LABEL_HOURS = 3.0      # teacher labelling on both GPUs; frames it doesn't reach still train on the human labels",
    "LABEL_HOURS = 0.0      # 0 = reuse E3's SAM 3 labels exactly (keeps the comparison clean)\nTRAIN_HOURS = 6.9      # same as E3")
_set = _set.replace("TOTAL_HOURS = 11.0", "TOTAL_HOURS = 9.0 ")

_models = """# --- Reuse E3's SAM 3 labels; YOLOE weights and text encoder ---------------------------
LAB = glob.glob("/kaggle/input/skyquery-e3-labels/**/teacher_val_0.npz", recursive=True)
assert LAB, "Add your e3_results.zip as a dataset named skyquery-e3-labels"
for f in glob.glob(os.path.join(os.path.dirname(LAB[0]), "teacher_*.npz")):
    shutil.copy(f, "cache/")
print(sorted(os.listdir("cache")))
from ultralytics import YOLOE
_m = YOLOE(f"yoloe-26{MODEL_SIZE}-seg.pt"); _m.get_text_pe(["tree"]); del _m
assert os.path.exists(f"weights/yoloe-26{MODEL_SIZE}-drone-ov.pt"), "the E3 weights should be inside skyquery-code/weights"
"""

_lab = E3_CELLS[3][1]
_lab = _lab.replace('run_on_gpus([(["--split", VAL', 'if LABEL_HOURS > 0:\n    run_on_gpus([(["--split", VAL')
_lab = _lab.replace('run_on_gpus([(["--split", TRAIN', '    run_on_gpus([(["--split", TRAIN')
_lab = _lab.replace('print("hours left:"', 'print("labelling:", "reused from E3" if LABEL_HOURS <= 0 else "done", "| hours left:"')

_tr = E3_CELLS[5][1]
_tr = _tr.replace('OUT_W = f"/kaggle/working/yoloe-26{MODEL_SIZE}-drone-ov.pt"', 'OUT_W = f"/kaggle/working/yoloe-26{MODEL_SIZE}-drone-ov-b.pt"')
_tr = _tr.replace("train_hours = 0 if QUICK else max(0.5, hours_left() - EVAL_HOURS - 0.25)",
                  "train_hours = 0 if QUICK else min(TRAIN_HOURS, max(0.5, hours_left() - EVAL_HOURS - 0.25))")
_tr = _tr.replace('"--name", "e3_ov",', '"--name", "e3b_ov", "--freeze-align",')

_ev = E3_CELLS[6][1]
_ev = _ev.replace('name = f"weights/yoloe-26{MODEL_SIZE}-drone-ov.pt"', 'name = f"weights/yoloe-26{MODEL_SIZE}-drone-ov-b.pt"')
_ev = _ev.replace('"--model", f"tuned={OUT_W}",', '"--model", f"tuned={OUT_W}", "--model", f"e3=weights/yoloe-26{MODEL_SIZE}-drone-ov.pt",')
_ev = _ev.replace("reports/ovd_e3_val.json", "reports/ovd_e3b_val.json")

_pk = E3_CELLS[7][1]
_pk = _pk.replace("ovd_e3_val.json", "ovd_e3b_val.json").replace("e3_results.zip", "e3b_results.zip")
_pk = _pk.replace("reports/e3_dataset_stats.json", "reports/e3b_dataset_stats.json").replace(" cache/teacher_val_*.npz cache/teacher_train_*.npz", "")

E3B_CELLS = [("markdown", E3B_INTRO), ("code", _set), ("code", _models), ("code", _lab), ("code", E3_CELLS[4][1]),
             ("code", _tr), ("code", _ev), ("code", _pk)]


def nb(cells):
    return {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    (HERE / "e1_zero_shot.ipynb").write_text(json.dumps(nb(E1_CELLS), indent=1))
    (HERE / "e2_heldout.ipynb").write_text(json.dumps(nb(E2_CELLS), indent=1))
    (HERE / "e3_drone_openvocab.ipynb").write_text(json.dumps(nb(E3_CELLS), indent=1))
    (HERE / "e3b_frozen_alignment.ipynb").write_text(json.dumps(nb(E3B_CELLS), indent=1))
    print("wrote", HERE / "e1_zero_shot.ipynb", HERE / "e2_heldout.ipynb", HERE / "e3_drone_openvocab.ipynb", HERE / "e3b_frozen_alignment.ipynb")
