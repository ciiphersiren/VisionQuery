"""Phase 2: 20 count records drawn on their images for a human to check. Counted boxes orange, ignore regions gray, region outline white.
UAVDT frames are not on this PC, so the sample is VisDrone and CARPK.

Run:  python tools/phase2_overlays.py --visdrone DIR --carpk DIR
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dronekit import countset as K, data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--visdrone", type=Path, required=True)
ap.add_argument("--carpk", type=Path, required=True)
A = ap.parse_args()
OUT = ROOT / "experiments/phase2/overlays"
OUT.mkdir(parents=True, exist_ok=True)
SP = [(data.load_visdrone(A.visdrone / "VisDrone2019-DET-test-dev", "test-dev"), 14), (data.load_carpk(A.carpk, "test"), 3), (data.load_carpk(A.carpk, "train"), 3)]
rng = np.random.default_rng(K.SEED)
rows = []
for sp, n in SP:
    recs = {r.key: r for r in sp.images}
    d = pd.DataFrame(K.records(sp))
    d = d[d["count"] > 0]
    for _, q in d.iloc[rng.choice(len(d), n, replace=False)].iterrows():
        r = recs[q.image]
        im = cv2.imread(str(r.path))
        for x1, y1, x2, y2 in r.ignore.astype(int):
            im[max(y1, 0):y2, max(x1, 0):x2] = (im[max(y1, 0):y2, max(x1, 0):x2] * .35 + 110).astype(np.uint8)
        ids = K.targets(sp.class_names)[q.target]
        box = K.region_box(q.region, r.width, r.height)
        B = r.boxes[np.isin(r.boxes[:, 0], ids)]
        B = B[K.in_region(B[:, 1:5], box) & ~K.in_ignore(B[:, 1:5], r.ignore)]
        for b in B[:, 1:5].astype(int):
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (20, 90, 255), 2)
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]) - 1, int(box[3]) - 1), (255, 255, 255), 3)
        label = f"{q.target} in {q.region}: record says {q['count']}"
        cv2.putText(im, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(im, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA)
        i = len(rows) + 1
        cv2.imwrite(str(OUT / f"{i:02d}_{q.dataset}_{q.image}.jpg"), cv2.resize(im, (1000, round(1000 * r.height / r.width))), [cv2.IMWRITE_JPEG_QUALITY, 80])
        assert len(B) == q["count"], (q.image, q.target, q.region, len(B), q["count"])
        rows.append(dict(n=i, dataset=q.dataset, split=q.split, image=q.image, target=q.target, region=q.region, record_count=q["count"], human_count="", human_ok=""))
pd.DataFrame(rows).to_csv(OUT / "check_sheet.csv", index=False)
print(len(rows), "overlays in", OUT)
