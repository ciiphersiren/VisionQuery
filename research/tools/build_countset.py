"""Phase 2: write countset/ from the labels (CPU). No detections and no model output are read.

Run:  python tools/build_countset.py --visdrone DIR --uavdt-gt DIR --uavdt-attr DIR --carpk DIR [--index image_index.csv]
--index is image_index.csv from phase1_eval (real frame sizes). Without it, each UAVDT sequence's width comes from its labels (960 if no
box passes x = 960, else 1024), which only matters for sequences near that edge.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dronekit import countset as K, data  # noqa: E402

ap = argparse.ArgumentParser()
for a in ("visdrone", "uavdt-gt", "uavdt-attr", "carpk"):
    ap.add_argument("--" + a, required=True, type=Path)
ap.add_argument("--index", type=Path)
ap.add_argument("--out", type=Path, default=ROOT / "countset")
A = ap.parse_args()
A.out.mkdir(parents=True, exist_ok=True)

manifest = json.loads((ROOT / "experiments/phase1/results/retrain_tile/uavdt_split.json").read_text())
VAL_SEQS, TEST_SEQS = manifest["val_seqs"], manifest["test_seqs"]
STRIDE = 10
sizes = {}
if A.index:
    ix = pd.read_csv(A.index)
    sizes = {s: (int(w), int(h)) for s, w, h in ix[ix.dataset == "uavdt"][["sequence", "width", "height"]].drop_duplicates().itertuples(index=False)}


def uavdt(seqs, name):
    recs = []
    for s in seqs:
        if s in sizes:
            size = sizes[s]
        else:
            rows = np.loadtxt(A.uavdt_gt / f"{s}_gt_whole.txt", delimiter=",", usecols=(2, 4))
            size = (960 if (rows[:, 0] + rows[:, 1]).max() <= 960 else 1024, 540)
        recs += data.load_uavdt(Path("no-images"), A.uavdt_gt, A.uavdt_attr, [s], STRIDE, name, size=size).images
    return data.Split("uavdt", name, list(data.UAVDT_CLASSES), recs)


SPLITS = {("visdrone", "val"): data.load_visdrone(A.visdrone / "VisDrone2019-DET-val", "val"),
          ("visdrone", "test-dev"): data.load_visdrone(A.visdrone / "VisDrone2019-DET-test-dev", "test-dev"),
          ("uavdt", "val"): uavdt(VAL_SEQS, "val"),
          ("uavdt", "test"): uavdt(TEST_SEQS, "test"),
          ("carpk", "test"): data.load_carpk(A.carpk, "test"),
          ("carpk", "train"): data.load_carpk(A.carpk, "train")}   # evaluation only: no model trains on CARPK, and test is one parking lot
for k, sp in SPLITS.items():
    print(k, len(sp), "images", len(sp.sequences), "sequences", sp.n_objects, "objects")
assert len(SPLITS[("uavdt", "test")]) == 1665 and len(SPLITS[("uavdt", "val")]) == 367

DF = {k: pd.DataFrame(K.records(sp)) for k, sp in SPLITS.items()}

# Tertiles from the dev split (VisDrone val, whole-image records with objects), frozen for every split
dev = DF[("visdrone", "val")]
dw = dev[(dev.region == "whole") & (dev["count"] > 0)]
edges = dict(side=K.tertile_edges(dw["median_side"]), density=K.tertile_edges(dw["density"]))
(A.out / "tertiles.json").write_text(json.dumps(dict(source="VisDrone val, whole-image records with count > 0", **edges), indent=1))
print("tertile edges:", edges)

stats = []
for k, d in DF.items():
    d["size_bin"] = K.tertile(d["median_side"], edges["side"], ("small", "mid", "large"))
    d["density_bin"] = K.tertile(d["density"], edges["density"], ("dense", "mid", "sparse"))    # low nearest-neighbor distance = dense
    d.to_json(A.out / f"{k[0]}__{k[1]}.jsonl.gz", orient="records", lines=True, compression="gzip")
    w = d[d.region == "whole"]
    stats.append(dict(dataset=k[0], split=k[1], images=d.image.nunique(), sequences=d.sequence.nunique(), records=len(d), whole_records=len(w), whole_nonzero=int((w["count"] > 0).sum()),
                      **{f"bin_{a}": int((w.count_bin == a).sum()) for _, _, a in K.COUNT_BINS}))
pd.DataFrame(stats).to_csv(A.out / "stats.csv", index=False)
print(pd.DataFrame(stats).to_string())

# Fixed subset for the expensive systems: evaluation splits only, never dev
EVAL = [("visdrone", "test-dev"), ("uavdt", "test"), ("carpk", "test"), ("carpk", "train")]
pool = pd.concat([DF[k] for k in EVAL], ignore_index=True)
base = pool[~pool.target.isin(K.GROUPS)]                        # the subset balances on single classes; groups are not queried
# Frozen once written: phase3_vlm embeds these queries, so a rebuild with real frame sizes must not change them
if (A.out / "subset_300.csv").exists():
    sub = pd.read_csv(A.out / "subset_300.csv")
    subset = list(zip(sub.dataset, sub.image))
else:
    subset = K.stratified_subset(base, 300)
    sub = pd.DataFrame(subset, columns=["dataset", "image"])
    sub.to_csv(A.out / "subset_300.csv", index=False)
print(sub.dataset.value_counts().to_string())

# Queries: whole image; each image's most frequent class, plus car when that is a different class and car exists in the dataset
keys = set(subset)
w = base[(base.region == "whole")]
w = w[[(d, i) in keys for d, i in zip(w.dataset, w.image)]]
rows = []
for (d, i), g in w.groupby(["dataset", "image"]):
    g = g.sort_values(["count", "target"], ascending=[False, True], kind="stable")
    pick = [g.iloc[0]]
    car = g[g.target == "car"]
    if g.iloc[0].target != "car" and len(car):
        pick.append(car.iloc[0])
    rows += [r.to_dict() for r in pick]
Q = pd.DataFrame(rows)[["dataset", "split", "image", "sequence", "target", "count", "count_bin", "size_bin", "density_bin", "median_side", "density", "altitude"]]
if not (A.out / "vlm_queries.csv").exists():
    Q.to_csv(A.out / "vlm_queries.csv", index=False)
print(len(Q), "queries;", Q.groupby(["dataset", "count_bin"]).size().to_string(), sep="\n")
(A.out / "meta.json").write_text(json.dumps(dict(seed=K.SEED, max_ignore=K.MAX_IGNORE, uavdt_stride=STRIDE, sizes_from_index=bool(A.index)), indent=1))
