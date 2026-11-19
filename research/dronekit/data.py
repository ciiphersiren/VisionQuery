"""Dataset loading into one common structure.

Every dataset becomes a `Split`: a list of `ImageRecord`s, each holding
  boxes   (N, 7) float32: class_id (0-based), x1, y1, x2, y2, occlusion, truncation
  ignore  (M, 4) float32: x1, y1, x2, y2 of ignore regions
  sequence: video-sequence id parsed from the filename (bootstrap / CV grouping unit)

VisDrone rules: category 0 or score 0 = ignore region; category 11
("others") dropped; categories 1-10 -> class ids 0-9. Boxes are kept exactly as annotated
(not clipped), because the eval-v2 scorer used the raw x, y, w, h.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import VISDRONE_CLASSES

VISDRONE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{split}.zip"


@dataclass
class ImageRecord:
    key: str                 # filename stem, unique within a split
    path: Path
    width: int
    height: int
    sequence: str
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), np.float32))
    ignore: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    attrs: dict = field(default_factory=dict)          # sequence-level attributes (UAVDT: weather, altitude, view)


@dataclass
class Split:
    dataset: str
    name: str
    class_names: list
    images: list             # list[ImageRecord], sorted by key; position = COCO image id

    def __len__(self):
        return len(self.images)

    def subset(self, n):
        """First n images (smoke tests). Image ids stay positional."""
        return Split(self.dataset, self.name, self.class_names, self.images[:n])

    @property
    def n_objects(self):
        return int(sum(len(r.boxes) for r in self.images))

    @property
    def sequences(self):
        return sorted({r.sequence for r in self.images})


def visdrone_sequence(stem: str) -> str:
    """'0000001_02999_d_0000005' -> '0000001'. VisDrone-DET frames from one video share the first field."""
    return stem.split("_")[0]


def parse_visdrone_txt(text: str, training_rules: bool = False):
    """Return (boxes (N,7), ignore (M,4)) from one VisDrone annotation file.

    The two original notebooks differ on one edge case, and both are kept on purpose:
    - scorer (evaluation v2, default): category 11 rows are dropped first, even with score 0;
    - training converter (visdrone-training, training_rules=True): score 0 makes any row an ignore
      region (painted gray), including category 11, which is checked afterwards.
    """
    boxes, ignore = [], []
    for row in text.strip().splitlines():
        v = row.strip().rstrip(",").split(",")
        if len(v) < 6:
            continue
        x, y, w, h, score, cat = map(int, v[:6])
        trunc = int(v[6]) if len(v) > 6 else -1
        occ = int(v[7]) if len(v) > 7 else -1
        if w <= 0 or h <= 0 or (cat == 11 and not training_rules):
            continue
        if score == 0 or cat == 0:
            ignore.append([x, y, x + w, y + h])
        elif 1 <= cat <= 10:
            boxes.append([cat - 1, x, y, x + w, y + h, occ, trunc])
    return (np.array(boxes, np.float32).reshape(-1, 7), np.array(ignore, np.float32).reshape(-1, 4))


def load_visdrone(split_dir, split_name: str | None = None, training_rules: bool = False) -> Split:
    """Load a VisDrone2019-DET split folder (with images/ and annotations/).
    training_rules=True reproduces the training converter (see parse_visdrone_txt); use it only for building
    training data, never for scoring."""
    split_dir = Path(split_dir)
    recs = []
    for ann in sorted((split_dir / "annotations").glob("*.txt")):
        img = split_dir / "images" / f"{ann.stem}.jpg"
        with Image.open(img) as im:
            W, H = im.size
        boxes, ignore = parse_visdrone_txt(ann.read_text(), training_rules)
        recs.append(ImageRecord(ann.stem, img, W, H, visdrone_sequence(ann.stem), boxes, ignore))
    return Split("visdrone", split_name or split_dir.name, list(VISDRONE_CLASSES), recs)


def download_visdrone(root, split: str) -> Path:
    """Download + extract one split ('train', 'val', 'test-dev') into root. Returns the split folder."""
    root = Path(root)
    d = root / f"VisDrone2019-DET-{split}"
    if not (d / "annotations").exists():
        root.mkdir(parents=True, exist_ok=True)
        print("Downloading", d.name)
        with urllib.request.urlopen(VISDRONE_URL.format(split=split)) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    return d


def split_stats(split: Split) -> dict:
    """Object counts and size shares (non-ignored objects)."""
    b = np.concatenate([r.boxes for r in split.images]) if split.images else np.zeros((0, 7))
    area = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
    n = max(len(b), 1)
    out = dict(images=len(split), sequences=len(split.sequences), objects=len(b),
               objects_per_image=len(b) / max(len(split), 1),
               tiny=float((area < 16 ** 2).sum() / n),
               small=float(((area >= 16 ** 2) & (area < 32 ** 2)).sum() / n),
               medium_plus=float((area >= 32 ** 2).sum() / n))
    for k, c in enumerate(split.class_names):
        out[f"share_{c}"] = float((b[:, 0] == k).sum() / n)
    return out


# --------------------------------------------------------------------------- UAVDT
UAVDT_CLASSES = ["car", "truck", "bus"]
UAVDT_ATTRS = ["daylight", "night", "fog", "low-alt", "medium-alt", "high-alt",
               "front-view", "side-view", "bird-view", "long-term"]
# gt_whole.txt codes -> ordinal levels (0 = none). Occlusion: 1 no, 4 small, 3 medium, 2 large.
_UAVDT_OCC = {1: 0, 4: 1, 3: 2, 2: 3}
_UAVDT_OUT = {1: 0, 3: 1, 2: 2}                         # out-of-view: 1 no, 3 small, 2 medium


def read_uavdt_attrs(attr_root):
    r"""M_attr/{train,test}/M0101_attr.txt -> ({seq: attrs dict}, {seq: 'train'|'test'}).
    One official file is named 'M0701 _attr.txt' (stray space), hence the \s*."""
    import re
    attrs, split = {}, {}
    for p in Path(attr_root).rglob("*_attr.txt"):
        m = re.match(r"(M\d+)\s*_attr\.txt$", p.name)
        if not m:
            continue
        bits = [int(b) for b in p.read_text().strip().split(",")]
        a = dict(zip(UAVDT_ATTRS, bits))
        a["weather"] = next((n for n in ("daylight", "night", "fog") if a.get(n)), "unknown")
        a["altitude"] = next((n for n in ("low-alt", "medium-alt", "high-alt") if a.get(n)), "unknown")
        attrs[m[1]], split[m[1]] = a, p.parent.name
    return attrs, split


def _uavdt_rows(path):
    rows = [list(map(int, l.split(","))) for l in Path(path).read_text().splitlines() if l.strip()]
    return np.array(rows, np.int64).reshape(-1, 9)


def find_uavdt_frames(images_root, seq):
    """{frame_index: path} for one sequence; files are img000001.jpg ... (frame 1 = img000001)."""
    import re
    d = next((p for p in Path(images_root).rglob(seq) if p.is_dir()), None)
    if d is None:
        return {}
    return {int(m[1]): p for p in d.glob("img*.jpg") if (m := re.match(r"img(\d+)\.jpg$", p.name))}


def load_uavdt(images_root, gt_dir, attr_root, seqs, stride=1, name="uavdt", size=None):
    """UAVDT-M detection frames of `seqs`, every `stride`-th annotated frame (frames 1, 1+stride, ...).

    boxes columns 5/6 = occlusion / out-of-view as ordinal levels (0 none ... 3 large).
    Boxes with category outside 1-3 are dropped; *_gt_ignore.txt boxes become ignore regions.
    `size` = (W, H) skips opening images; otherwise read from the first frame of each sequence (widths differ: M0901 is 960).
    """
    attrs, _ = read_uavdt_attrs(attr_root)
    recs = []
    for seq in seqs:
        gt = _uavdt_rows(Path(gt_dir) / f"{seq}_gt_whole.txt")
        ign_p = Path(gt_dir) / f"{seq}_gt_ignore.txt"
        ign = _uavdt_rows(ign_p) if ign_p.exists() else np.zeros((0, 9), np.int64)
        frames_img = find_uavdt_frames(images_root, seq)
        # annotated frames only: image files without a gt_whole row are unlabeled, not known to be empty
        frames = sorted(set(gt[:, 0].tolist()))
        frames = [f for f in frames if (f - frames[0]) % stride == 0] if frames else []
        W, H = size or (None, None)
        if W is None and frames_img:
            with Image.open(next(iter(frames_img.values()))) as im:
                W, H = im.size
        for f in frames:
            g = gt[(gt[:, 0] == f) & np.isin(gt[:, 8], [1, 2, 3])]
            b = np.c_[g[:, 8] - 1, g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5],
                      [_UAVDT_OCC.get(int(v), -1) for v in g[:, 7]], [_UAVDT_OUT.get(int(v), -1) for v in g[:, 6]]]
            g = ign[ign[:, 0] == f]
            ig = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
            path = frames_img.get(f, Path(images_root) / seq / f"img{f:06d}.jpg")
            recs.append(ImageRecord(f"{seq}_img{f:06d}", path, W, H, seq,
                                    np.asarray(b, np.float32).reshape(-1, 7), np.asarray(ig, np.float32).reshape(-1, 4),
                                    dict(attrs.get(seq, {}))))
    return Split("uavdt", name, list(UAVDT_CLASSES), recs)


# --------------------------------------------------------------------------- CARPK
CARPK_CLASSES = ["car"]


def load_carpk(root, split_name="test"):
    """CARPK devkit: Images/*.png, Annotations/*.txt (one 'x1 y1 x2 y2 cls' per line), ImageSets/{train,test}.txt.
    Sequence id = parking lot (filename without the trailing frame number)."""
    root = Path(root)
    img_dir = next(p for p in root.rglob("Images") if p.is_dir())
    ann_dir = next(p for p in root.rglob("Annotations") if p.is_dir())
    lst = next((p for p in root.rglob(f"{split_name}.txt") if p.parent.name == "ImageSets"), None)
    stems = [s.strip() for s in lst.read_text().split()] if lst else sorted(p.stem for p in img_dir.glob("*.png"))
    recs = []
    for s in sorted(stems):
        img = img_dir / f"{s}.png"
        with Image.open(img) as im:
            W, H = im.size
        rows = []
        a = ann_dir / f"{s}.txt"
        if a.exists():
            for line in a.read_text().splitlines():
                v = line.replace(",", " ").split()
                if len(v) >= 4:
                    x1, y1, x2, y2 = map(float, v[:4])
                    if x2 > x1 and y2 > y1:
                        rows.append([0, x1, y1, x2, y2, -1, -1])
        recs.append(ImageRecord(s, img, W, H, s.rsplit("_", 1)[0], np.array(rows, np.float32).reshape(-1, 7)))
    return Split("carpk", split_name, list(CARPK_CLASSES), recs)


# --------------------------------------------------------------------------- views of a split
def without_ignore(split):
    """Same images and boxes with no ignore regions: detections on them then count as false positives,
    as in the Ultralytics validator. Diagnostics only; never report it as an eval-v2 number."""
    import copy
    out = []
    for r in split.images:
        rr = copy.copy(r)
        rr.ignore = np.zeros((0, 4), np.float32)
        out.append(rr)
    return Split(split.dataset, split.name, list(split.class_names), out)


def by_sequence(split):
    """{sequence id: Split holding that sequence's images}."""
    return {s: Split(split.dataset, split.name, list(split.class_names), [r for r in split.images if r.sequence == s])
            for s in split.sequences}


# --------------------------------------------------------------------------- validation sequences
def sequence_table(split):
    """{sequence id: split_stats + altitude + weather}. Input for pick_val_sequences."""
    return {s: {**split_stats(sp), "altitude": sp.images[0].attrs.get("altitude"), "weather": sp.images[0].attrs.get("weather")}
            for s, sp in by_sequence(split).items()}


def pick_val_sequences(table, alloc, rare=("truck", "bus"), min_rare_seqs=3):
    """Pick validation sequences from `table` (sequence_table of the training pool; never pass test sequences).

    alloc: {(altitude, weather): n sequences}. Among all picks with that allocation in which every `rare` class appears in
    >= min_rare_seqs sequences, return the one whose tiny, truck and bus shares of objects are closest to the whole pool
    (sum of absolute differences). Ties break on the sorted sequence names, so the result is deterministic.
    Returns (sorted sequence ids, distance)."""
    from itertools import combinations, product

    def shares(seqs):
        n = max(sum(table[s]["objects"] for s in seqs), 1)
        return [sum(table[s][k] * table[s]["objects"] for s in seqs) / n for k in ("tiny", *(f"share_{c}" for c in rare))]

    target = shares(list(table))
    cells = [combinations(sorted(s for s, v in table.items() if (v["altitude"], v["weather"]) == c), n) for c, n in alloc.items()]
    best = None
    for pick in product(*cells):
        seqs = sorted(s for group in pick for s in group)
        if any(sum(table[s][f"share_{c}"] > 0 for s in seqs) < min_rare_seqs for c in rare):
            continue
        cand = (sum(abs(a - b) for a, b in zip(shares(seqs), target)), seqs)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise ValueError("no pick satisfies the allocation and the rare-class rule")
    return best[1], best[0]


# --------------------------------------------------------------------------- shared classes
def remap_split(split, mapping, new_names):
    """Re-label GT onto a shared class list. mapping: old class name -> new name, "ignore" (box becomes an
    ignore region: detections on it are neither TP nor FP), or absent/None (box dropped)."""
    import copy
    out = []
    for r in split.images:
        keep, ign = [], [r.ignore]
        for b in r.boxes:
            tgt = mapping.get(split.class_names[int(b[0])])
            if tgt == "ignore":
                ign.append(b[None, 1:5])
            elif tgt is not None:
                nb = b.copy(); nb[0] = new_names.index(tgt); keep.append(nb)
        rr = copy.copy(r)
        rr.boxes = np.array(keep, np.float32).reshape(-1, 7)
        rr.ignore = np.concatenate(ign).astype(np.float32).reshape(-1, 4)
        out.append(rr)
    return Split(split.dataset, split.name, list(new_names), out)


def remap_dets(dets, src_names, mapping, new_names):
    """Same mapping for detections ({key: (N,6)}); classes mapped to None or "ignore" are dropped."""
    lut = np.full(len(src_names), -1)
    for i, n in enumerate(src_names):
        t = mapping.get(n)
        if t not in (None, "ignore"):
            lut[i] = new_names.index(t)
    out = {}
    for k, d in dets.items():
        c = lut[d[:, 5].astype(int)]
        nd = d[c >= 0].copy(); nd[:, 5] = c[c >= 0]
        out[k] = nd
    return out
