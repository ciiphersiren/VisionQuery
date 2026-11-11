"""Turn VisDrone ground truth + teacher labels into YOLOE training sets.

Three Ultralytics datasets, each with its own class list (Ultralytics samples
the negative prompts for an image from its own dataset's names, so a frame is
never told "there is no tree here" unless the teacher actually looked):

  A  train frames the teacher labelled   names = VISDRONE + SEEN
  B  train frames it did not reach       names = VISDRONE
  V  val frames (full)                   names = VISDRONE + SEEN

Train frames become 640 px tiles (20% overlap) plus the full frame, as in the
original YOLO26s recipe; ignored regions are painted grey. A teacher box is
kept when its score >= --tau, its centre is outside ignored regions, and it
does not sit on a human-labelled box (IoU >= 0.5, e.g. "cart" on a tricycle).

    python -m skyquery.ovtrain.build --train data/VisDrone2019-DET-train --val data/VisDrone2019-DET-val \
        --teacher-train cache/teacher_train_*.npz --teacher-val cache/teacher_val_*.npz --out /tmp/e3
"""

from __future__ import annotations

import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..calibrate.tiles import tile_origins, yolo_lines
from ..eval.visdrone import in_ignore, load_split
from ..openvocab import iou_matrix
from . import vocab


def load_cache(patterns: list[str]) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]], dict]:
    preds, shapes, meta = {}, {}, {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            with np.load(f) as z:
                m = json.loads(str(z["__meta"])) if "__meta" in z.files else {}
                if meta and m.get("prompts") != meta.get("prompts"):
                    raise ValueError(f"{f} used different prompts")
                meta = meta or m
                for k in z.files:
                    if k.endswith("__shape"):
                        shapes[k[:-7]] = tuple(int(v) for v in z[k])
                    elif k != "__meta":
                        preds[k] = z[k]
    return preds, shapes, meta


def pseudo(arr: np.ndarray, tau: float, prompt_to_cls: dict[int, int], gt_boxes: np.ndarray, ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Teacher array -> (boxes, cls) after the keep rules. Prompts not in prompt_to_cls are dropped."""
    if arr is None or len(arr) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    a = arr[(arr[:, 4] >= tau) & np.isin(arr[:, 5].astype(int), list(prompt_to_cls))]
    a = a[~in_ignore(a[:, :4], ignore)]
    if len(gt_boxes) and len(a):
        a = a[iou_matrix(a[:, :4], gt_boxes).max(1) < 0.5]
    return a[:, :4], np.array([prompt_to_cls[int(c)] for c in a[:, 5]], int)


def _write_frame(job):
    src_img, out, split, stem, boxes, cls, ignore, tiles, tile, overlap = job
    from PIL import Image

    a = np.asarray(Image.open(src_img).convert("RGB")).copy()
    for x1, y1, x2, y2 in ignore.astype(int):
        a[max(y1, 0):y2, max(x1, 0):x2] = 114
    H, W = a.shape[:2]
    views = [(0, 0, W, H, "full")]
    if tiles:
        views += [(x0, y0, min(tile, W), min(tile, H), f"t{x0}_{y0}") for y0 in tile_origins(H, tile, overlap) for x0 in tile_origins(W, tile, overlap)]
    n = 0
    for x0, y0, tw, th, tag in views:
        lines = yolo_lines(boxes, cls, x0, y0, tw, th)
        if tag != "full" and not lines:
            continue
        name = f"{stem}_{tag}"
        Image.fromarray(a[y0:y0 + th, x0:x0 + tw]).save(Path(out) / "images" / split / f"{name}.jpg", quality=90)
        (Path(out) / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        n += 1
    return n


def write_yaml(path: Path, root: Path, names: list[str], val_dir: Path):
    path.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: {val_dir.resolve()}\nnames:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(names)))


def build(train: Path, val: Path, teacher_train: list[str], teacher_val: list[str], out: Path, tau: float = 0.5,
          tile: int = 640, overlap: float = 0.2, workers: int = 4, limit: int = 0) -> dict:
    seen_prompts = vocab.teacher_prompts(vocab.SEEN)
    names_full = vocab.VISDRONE + vocab.SEEN

    def mapping(meta):
        idx = {p: i for i, p in enumerate(meta.get("prompts", []))}
        return {idx[p]: len(vocab.VISDRONE) + j for j, p in enumerate(seen_prompts) if p in idx}

    tt, _, mt = load_cache(teacher_train)
    tv, _, mv = load_cache(teacher_val)
    m_train, m_val = mapping(mt), mapping(mv)
    stats = {"tau": tau, "teacher": mt.get("teacher") or mv.get("teacher"), "train_boxes_per_class": {}}

    jobs = []
    for ds in ("A", "B", "V"):
        for sub in ("images", "labels"):
            (out / ds / sub / ("val" if ds == "V" else "train")).mkdir(parents=True, exist_ok=True)

    gtr = load_split(train)
    stems = sorted(gtr, key=lambda s: (s not in tt, s))[: limit or None]  # --limit (tests) keeps labelled frames first
    counts = np.zeros(len(names_full), int)
    nA = nB = 0
    for s in stems:
        g = gtr[s]
        if s in tt:
            pb, pc = pseudo(tt[s], tau, m_train, g.boxes, g.ignore)
            boxes, cls, ds = np.concatenate([g.boxes, pb]), np.concatenate([g.cls, pc]).astype(int), "A"
            nA += 1
        else:
            boxes, cls, ds = g.boxes, g.cls, "B"
            nB += 1
        np.add.at(counts, cls, 1)
        jobs.append((str(train / "images" / f"{s}.jpg"), str(out / ds), "train", s, boxes, cls, g.ignore, True, tile, overlap))

    gva = load_split(val)
    vstems = sorted(gva, key=lambda s: (s not in tv, s))[: limit or None]
    for s in vstems:
        g = gva[s]
        pb, pc = pseudo(tv.get(s), tau, m_val, g.boxes, g.ignore)
        jobs.append((str(val / "images" / f"{s}.jpg"), str(out / "V"), "val", s, np.concatenate([g.boxes, pb]), np.concatenate([g.cls, pc]).astype(int), g.ignore, False, tile, overlap))

    with ProcessPoolExecutor(max(1, workers)) as ex:
        views = list(ex.map(_write_frame, jobs, chunksize=8))
    vdir = out / "V" / "images" / "val"
    write_yaml(out / "A.yaml", out / "A", names_full, vdir)
    write_yaml(out / "B.yaml", out / "B", vocab.VISDRONE, vdir)
    write_yaml(out / "V.yaml", out / "V", names_full, vdir)
    train_sets = [str(out / "A.yaml")] if nA else []
    train_sets += [str(out / "B.yaml")] if nB else []
    data = {"train": {"yolo_data": train_sets}, "val": {"yolo_data": [str(out / "V.yaml")]}}
    (out / "data.json").write_text(json.dumps(data, indent=2))
    stats.update({
        "frames_teacher_labelled": nA, "frames_gt_only": nB, "val_frames": len(vstems),
        "val_teacher_labelled": sum(s in tv for s in vstems),
        "train_views": int(sum(views[: len(stems)])),
        "train_boxes_per_class": {vocab.first(n): int(c) for n, c in zip(names_full, counts)},
    })
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--teacher-train", nargs="*", default=[])
    ap.add_argument("--teacher-val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    st = build(Path(a.train), Path(a.val), a.teacher_train, a.teacher_val, Path(a.out), a.tau, workers=a.workers, limit=a.limit)
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
