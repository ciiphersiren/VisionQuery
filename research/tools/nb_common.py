"""Shared helpers for building self-contained Kaggle notebooks with dronekit embedded as %%writefile cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_FILES = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "dronekit").glob("*.py")) + \
            sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tests").glob("*.py"))


# Needs INPUT and RAW from the Config cell. Sets IMG_ROOT, GT_DIR, ATTR.
UAVDT_FETCH_CELL = r'''
# --- Download + extract the official release (skipped if an attached copy or a previous extract exists) ---
import subprocess, tarfile, zipfile
# Team copies of the official UAVDT-M Google Drive files (originals: https://sites.google.com/view/daweidu/projects/uavdt;
# the original IDs hit Drive's download quota: images 1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc, toolkit 19498uJd7T9w4quwnQEy62nibt3uyT9pq, attrs 1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1)
DRIVE = {"images": "1F9BN_AE5kqo157O2RaH91g4filLIYpRr",       # UAV-benchmark-M (frames); team copy of the official file
         "toolkit": "15N2S1QGlLxwYXHg2RtFyPw_b1q5rxK6t",      # UAV-benchmark-MOTD_v1.0 (GT incl. *_gt_whole / *_gt_ignore)
         "attrs": "1vMYo-s7CZ0QxQsY9DeIBlYjWgA2cv-Q9"}        # M_attr (sequence attributes; train/test folders = split)

def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        zipfile.ZipFile(archive).extractall(dest)
    elif tarfile.is_tarfile(archive):
        tarfile.open(archive).extractall(dest)
    else:                                            # e.g. .rar / .7z
        for cmd in (["7z", "x", "-y", f"-o{dest}", str(archive)], ["unrar", "x", "-o+", str(archive), str(dest)]):
            if shutil.which(cmd[0]) and subprocess.run(cmd, capture_output=True).returncode == 0:
                return
        raise RuntimeError(f"cannot extract {archive}: unknown format; send this message back")

def seq_dir_root(root, seq="M0101"):
    hits = [p for p in Path(root).rglob(seq) if p.is_dir() and any(p.glob("img*.jpg"))] if Path(root).exists() else []
    return hits[0].parent if hits else None

def drive_download(key, dest):
    # The direct link first: gdown gives up on Drive's "too many users" page even when this link still serves the file
    url = f"https://drive.usercontent.google.com/download?id={DRIVE[key]}&export=download&confirm=t"
    subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(dest), url])
    if dest.exists() and dest.stat().st_size > 0:
        with open(dest, "rb") as fh:
            if fh.read(1) != b"<":          # "<" = Drive's error page, not an archive
                return dest
    import gdown
    dest.unlink(missing_ok=True)
    try:
        return Path(gdown.download(id=DRIVE[key], output=str(dest), quiet=True))
    except Exception as e:
        raise RuntimeError(f"Google Drive refused {key}: {str(e)[:200]}. Upload it as a private Kaggle Dataset and attach it, or send this message back.")

def fetch(key, name):
    d = RAW / name
    if d.exists() and any(d.iterdir()):
        return d
    (RAW / "zips").mkdir(parents=True, exist_ok=True)
    print("downloading", name, "from Google Drive ...")
    f = drive_download(key, RAW / "zips" / f"{name}.bin")
    print("  got", f"{f.stat().st_size / 1e9:.2f} GB; extracting")
    extract(f, d); f.unlink()
    return d

def locate(pattern, key, name):
    # an attached Kaggle copy first, then a previous extract in /tmp, else download
    for root in (INPUT, RAW):
        hit = next(iter(sorted(root.rglob(pattern))), None) if root.exists() else None
        if hit:
            return hit
    return next(fetch(key, name).rglob(pattern))

IMG_ROOT = seq_dir_root(INPUT) or seq_dir_root(RAW / "images") or seq_dir_root(fetch("images", "images"))
GT_DIR = locate("M0101_gt_whole.txt", "toolkit", "toolkit").parent
ATTR = locate("M0101_attr.txt", "attrs", "attrs").parent.parent        # M_attr/ (holds train/ and test/)
print("images:", IMG_ROOT, "\nGT:", GT_DIR, "\nattributes:", ATTR)
'''


# Needs INPUT from the Config cell and CARPK_RAW. Sets CARPK_ROOT. The password never appears in code.
CARPK_FETCH_CELL = r'''
# --- CARPK: an attached copy, a previous extract, or the authors' Drive file with the password from Kaggle Secrets ---
CARPK_ID = "0BwSzgS8Mm48Ud2h2dW40Wko3a1E"; CARPK_KEY = "0-34K_uP-vYM7EWq0Q2iIVaw"
CARPK_IMAGES = 1448                  # 989 train + 459 test, read from ImageSets in the real archive

def carpk_password():
    if os.environ.get("DK_CARPK_PW"):
        return os.environ["DK_CARPK_PW"]
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("CARPK_PASSWORD")

def find_carpk():
    for root in (INPUT, CARPK_RAW):
        hit = next(iter(sorted(p for p in root.rglob("CARPK_devkit") if p.is_dir())), None) if root.exists() else None
        if hit:
            return hit
    CARPK_RAW.mkdir(parents=True, exist_ok=True)
    z = CARPK_RAW / "datasets.zip"
    url = f"https://drive.usercontent.google.com/download?id={CARPK_ID}&resourcekey={CARPK_KEY}&export=download&confirm=t"
    subprocess.run(["curl", "-sS", "-L", "-C", "-", "-o", str(z), url], check=True)
    print(f"CARPK archive: {z.stat().st_size / 1e9:.2f} GB")
    parts = [f"datasets/CARPK_devkit/data/{s}/*" for s in ("Images", "Annotations", "ImageSets")]     # the parent pattern matched nothing in a local test
    r = subprocess.run(["unzip", "-q", "-o", "-P", carpk_password(), str(z), *parts, "-d", str(CARPK_RAW)], capture_output=True, text=True)
    root = CARPK_RAW / "datasets/CARPK_devkit"
    assert r.returncode == 0 and len(list(root.rglob("*.png"))) == CARPK_IMAGES, "CARPK did not extract: check the CARPK_PASSWORD secret (" + r.stderr[-300:].replace("\n", " ") + ")"
    z.unlink()
    return root

CARPK_ROOT = find_carpk()
print("CARPK:", CARPK_ROOT)
'''


# Epoch scan, shared by phase1_epochscan and phase1_retrain. Needs from the notebook: VAL, NOIGN, SEQS, OUT, WDIR, RESULTS, PASS,
# SETUP, IGNORE_CONF, EARLY_PEAK, SMOKE, CUDA, RUN_NAME. Sets SCAN, SEQ, summary, peak_ev2, BEST_ROW, device.
VAL_TABLE_CODE = r'''
def ignore_area_share(sp):
    # share of image area inside ignore regions (union, so overlapping regions count once)
    total = 0.0
    for r in sp.images:
        m = np.zeros((r.height, r.width), bool)
        for x1, y1, x2, y2 in r.ignore.astype(int):
            m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        total += m.mean()
    return total / max(len(sp), 1)

seq_tab = pd.DataFrame({s: {**data.split_stats(sp), "ignore_regions_per_image": sum(len(r.ignore) for r in sp.images) / len(sp),
                            "ignore_area_share": ignore_area_share(sp), "altitude": sp.images[0].attrs.get("altitude"),
                            "weather": sp.images[0].attrs.get("weather")} for s, sp in SEQS.items()}).T
seq_tab.to_csv(OUT / "val_sequences.csv")
print(seq_tab[["images", "objects", "tiny", "small", "share_car", "share_truck", "share_bus", "ignore_area_share",
               "altitude", "weather"]])
'''

EPOCH_INFER_CELL = r'''
# --- Inference: every saved epoch, raw detections cached (re-running skips finished epochs) ---------
import logging
from ultralytics import YOLO, __version__ as UL_VERSION
from ultralytics.utils import LOGGER
EPOCH_FILE = re.compile(r"epoch(\d+)\.pt$")
ckpts = sorted((int(EPOCH_FILE.search(p.name)[1]), p) for p in WDIR.glob("epoch*.pt"))
assert [n for n, _ in ckpts] == list(range(len(ckpts))), "gap in the saved epochs: send the log"
assert len(ckpts) == len(RESULTS), f"{len(ckpts)} checkpoints but {len(RESULTS)} results.csv rows"
if SMOKE:
    ckpts = [ckpts[0], ckpts[-1]]

def rows_of_metrics(ck):
    # results.csv epochs whose validator mAP50-95 and mAP50 match this checkpoint's stored metrics
    tm = ck.get("train_metrics") or {}
    hit = RESULTS[np.isclose(RESULTS["metrics/mAP50-95(B)"], tm.get("metrics/mAP50-95(B)", np.nan), atol=1e-5) &
                  np.isclose(RESULTS["metrics/mAP50(B)"], tm.get("metrics/mAP50(B)", np.nan), atol=1e-5)]
    return hit["epoch"].astype(int).tolist()

LOGGER.setLevel(logging.ERROR)              # 'half' deprecation warning prints once per image otherwise
device = dict(cuda=CUDA, half=CUDA, device=0 if CUDA else "cpu")
identity = []
for n, p in ckpts:
    row = n + 1                             # epochN.pt is 0-based; results.csv "epoch" is 1-based
    model = YOLO(str(p))
    assert model.ckpt.get("epoch") == n, f"{p.name} stores epoch {model.ckpt.get('epoch')}"
    assert [model.names[i] for i in sorted(model.names)] == VAL.class_names, model.names
    identity.append(dict(file=p.name, row=row, stored_metrics_match_row=row in rows_of_metrics(model.ckpt)))
    infer.run_and_cache(model, VAL, [PASS], OUT / "cache" / f"epoch{row:02d}",
                        meta=dict(epoch=row, checkpoint=p.name, ultralytics=UL_VERSION), **device)
    del model
    if CUDA:
        torch.cuda.empty_cache()

model = YOLO(str(WDIR / "best.pt"))         # stripped checkpoint: stores epoch -1, so identify it by its metrics
BEST_ROW = rows_of_metrics(model.ckpt)
del model
LOGGER.setLevel(logging.INFO)
print("best.pt matches results.csv epoch(s):", BEST_ROW)
wrong = [i["file"] for i in identity if not i["stored_metrics_match_row"]]
print("checkpoints whose stored metrics do not match their results.csv row:", wrong or "none")
'''

EPOCH_SCORE_CELL = r'''
# --- Score every epoch: same detections under the crowd rule and with ignore regions removed (CPU) ---
gt_crowd, gt_noign = score._coco(VAL), score._coco(NOIGN)
rows, seq_rows = [], []
for n, _ in ckpts:
    e = n + 1
    raw, _meta = infer.load_cache(OUT / "cache" / f"epoch{e:02d}")
    dets, _ms = infer.apply_setup(raw, VAL, SETUP)
    m, pc, E = score.evaluate(VAL, dets, gt_crowd)
    mn, _pcn, _En = score.evaluate(NOIGN, dets, gt_noign)
    share, n_conf = score.ignore_share(VAL, dets, IGNORE_CONF)
    v = RESULTS[RESULTS["epoch"] == e].iloc[0]
    rows.append(dict(epoch=e, ultra_AP=v["metrics/mAP50-95(B)"], ultra_AP50=v["metrics/mAP50(B)"],
                     ultra_P=v["metrics/precision(B)"], ultra_R=v["metrics/recall(B)"],
                     **{f"ev2_{k}": x for k, x in m.items()}, noign_AP=mn["AP"], noign_AP50=mn["AP50"],
                     **{f"AP_{c}": x for c, x in score.per_class_ap(E, VAL.class_names).items()},
                     **{f"AP50_{c}": x for c, x in pc.items()},
                     dets_conf25=n_conf, share_conf25_in_ignore=share))
    for s, sp in SEQS.items():
        ms, _pcs, _Es = score.evaluate(sp, dets)
        seq_rows.append(dict(epoch=e, sequence=s, AP=ms["AP"], AP50=ms["AP50"]))
    print(f"epoch {e:2d}: validator {rows[-1]['ultra_AP']:.3f} | eval-v2 {m['AP']:.3f} | no-ignore {mn['AP']:.3f} | "
          + " ".join(f"{c} {rows[-1]['AP_' + c]:.3f}" for c in VAL.class_names)
          + f" | conf>={IGNORE_CONF} in ignore {share:.1%}")
SCAN = pd.DataFrame(rows); SEQ = pd.DataFrame(seq_rows)
SCAN.to_csv(OUT / "epoch_scan.csv", index=False); SEQ.to_csv(OUT / "per_sequence.csv", index=False)
'''

EPOCH_SUMMARY_CELL = r'''
# --- Summary + figure: the pre-registered readings are in the capstone report ---------------------------
def peak(col):
    return int(SCAN["epoch"][SCAN[col].idxmax()])

def corr(a, b):
    return float(SCAN[a].corr(SCAN[b])) if len(SCAN) > 2 else None

peak_ev2 = peak("ev2_AP")
summary = dict(
    smoke=SMOKE, epochs_scored=len(SCAN), n_val_frames=len(VAL), inference=PASS, ultralytics=UL_VERSION,
    best_pt_matches_results_epochs=BEST_ROW, checkpoints_with_mismatched_metrics=wrong,
    peak_epoch=dict(validator=peak("ultra_AP"), noignore=peak("noign_AP"), eval_v2=peak_ev2,
                    **{c: peak(f"AP_{c}") for c in VAL.class_names}),
    peak_value=dict(validator=float(SCAN["ultra_AP"].max()), noignore=float(SCAN["noign_AP"].max()),
                    eval_v2=float(SCAN["ev2_AP"].max())),
    last_epoch_value=dict(validator=float(SCAN["ultra_AP"].iloc[-1]), noignore=float(SCAN["noign_AP"].iloc[-1]),
                          eval_v2=float(SCAN["ev2_AP"].iloc[-1])),
    corr_with_validator=dict(noignore=corr("ultra_AP", "noign_AP"), eval_v2=corr("ultra_AP", "ev2_AP")),
    largest_adjacent_change=dict(validator=float(SCAN["ultra_AP"].diff().abs().max()),
                                 eval_v2=float(SCAN["ev2_AP"].diff().abs().max()),
                                 **{c: float(SCAN[f"AP_{c}"].diff().abs().max()) for c in VAL.class_names}),
    top5_eval_v2_epochs=SCAN.nlargest(5, "ev2_AP")["epoch"].tolist(),
    eval_v2_peaks_early=bool(peak_ev2 <= EARLY_PEAK))
(OUT / "epoch_scan_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
plots.plot_epoch_scan(SCAN, SEQ, VAL.class_names, OUT / "figures" / "epoch_scan.png",
                      f"{RUN_NAME}: UAVDT val per epoch, {PASS}")
'''


class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))

    def code(self, s):
        self.cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))

    def embed_package(self, install_line):
        self.code("# --- Install (Kaggle) ------------------------------------------------------\n"
                  "from pathlib import Path\nif Path(\"/kaggle\").exists():\n    !pip install -q " + install_line + "\n"
                  "Path(\"dronekit_src/dronekit\").mkdir(parents=True, exist_ok=True)\n"
                  "Path(\"dronekit_src/tests\").mkdir(parents=True, exist_ok=True)")
        self.md("## dronekit package (generated from the repo by `tools/`; do not edit here)")
        for f in PKG_FILES:
            self.code(f"%%writefile dronekit_src/{f}\n" + (ROOT / f).read_text(encoding="utf-8"))
        self.code(r'''
# --- Unit tests (CPU, under a minute) ----------------------------------------
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "dronekit_src/tests"],
                   capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print(r.stdout[-3000:], r.stderr[-2000:])
assert r.returncode == 0, "unit tests failed: stop here and send the log"
''')

    def save(self, rel_path):
        nb = dict(cells=[dict(c, source=c["source"].splitlines(keepends=True)) for c in self.cells],
                  metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
        out = ROOT / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", out, len(self.cells), "cells")
        return out
