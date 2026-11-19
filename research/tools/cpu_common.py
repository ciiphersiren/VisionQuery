"""Shared by the CPU analysis scripts: local label loading, merged-detection loading, paired count tables."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dronekit  # noqa: E402
from dronekit import countset as K, data  # noqa: E402

VD, UV = dronekit.VISDRONE_CLASSES, data.UAVDT_CLASSES
CLASSES = {"visdrone": VD, "uavdt": UV, "carpk": ["car"]}
DEV = {"VisDrone": ("visdrone", "val"), "UAVDT": ("uavdt", "val")}


def add_paths(ap):
    for a in ("visdrone", "uavdt-gt", "uavdt-attr", "carpk", "merged"):
        ap.add_argument("--" + a, required=True, type=Path)


def load_splits(A):
    """Labels for all splits. Frame sizes come from image_index.csv next to merged/ when present."""
    man = json.loads((ROOT / "experiments/phase1/results/retrain_tile/uavdt_split.json").read_text())
    ixp = A.merged.parent / "results/image_index.csv"
    sizes = {}
    if ixp.exists():
        ix = pd.read_csv(ixp)
        sizes = {s: (int(w), int(h)) for s, w, h in ix[ix.dataset == "uavdt"][["sequence", "width", "height"]].drop_duplicates().itertuples(index=False)}

    def uv(seqs, name):
        recs = []
        for s in seqs:
            if s not in sizes:
                rows = np.loadtxt(A.uavdt_gt / f"{s}_gt_whole.txt", delimiter=",", usecols=(2, 4))
                sizes[s] = (960 if (rows[:, 0] + rows[:, 1]).max() <= 960 else 1024, 540)
            recs += data.load_uavdt(Path("no-images"), A.uavdt_gt, A.uavdt_attr, [s], 10, name, size=sizes[s]).images
        return data.Split("uavdt", name, list(UV), recs)

    return {("visdrone", "val"): data.load_visdrone(A.visdrone / "VisDrone2019-DET-val", "val"),
            ("visdrone", "test-dev"): data.load_visdrone(A.visdrone / "VisDrone2019-DET-test-dev", "test-dev"),
            ("uavdt", "val"): uv(man["val_seqs"], "val"), ("uavdt", "test"): uv(man["test_seqs"], "test"),
            ("carpk", "test"): data.load_carpk(A.carpk, "test"), ("carpk", "train"): data.load_carpk(A.carpk, "train")}


def load_merged(merged, tag, key):
    """{image: (N,6)}. None if this model was not run on this split."""
    p = merged / f"{tag}__{key[0]}__{key[1]}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return {str(k): g[["x1", "y1", "x2", "y2", "score", "cls"]].to_numpy(np.float32) for k, g in df.groupby("image", observed=True, sort=False)}


def specs(model_classes, dataset, under_shift, carvan=True):
    """(pred_spec, gt_spec) {target: class ids}. Under shift the VisDrone model counts car + van, as prefixed in the capstone report;
    on its own dev split the labels' car + van are counted the same way, so calibration and evaluation measure one quantity."""
    cl = CLASSES[dataset]
    if under_shift:
        ids = lambda names: [model_classes.index(n) for n in names if n in model_classes]  # noqa: E731
        car = ["car", "van"] if carvan else ["car"]
        pred = {"car": ids(car)} if dataset == "carpk" else {"car": ids(car), "truck": ids(["truck"]), "bus": ids(["bus"])}
        gt = {t: [cl.index(t)] for t in pred}
        return pred, gt
    tg = K.targets(cl)
    return {t: [model_classes.index(cl[i]) for i in ids] for t, ids in tg.items()}, tg


def dev_specs_for_shift(model_classes, carvan=True):
    """The dev split of a VisDrone model when it is calibrated for shift: car means car + van in both labels and detections."""
    ids = lambda names: [model_classes.index(n) for n in names]  # noqa: E731
    car = ids(["car", "van"]) if carvan else ids(["car"])
    s = {"car": car, "truck": ids(["truck"]), "bus": ids(["bus"])}
    return s, s


def table(split, dets, thr, pred, gt, regions=("whole",)):
    return pd.DataFrame(K.paired_records(split, dets, thr, pred, gt, regions))
