"""Label VisDrone frames with the teacher, time-boxed and resumable.

Frames are visited in a fixed random order (so whatever gets labelled before
the deadline is a random sample of sequences), split across processes with
--shard/--nshards (one per GPU). Results are flushed to the .npz every few
frames; re-running skips frames already done.

    python -m skyquery.ovtrain.label --split data/VisDrone2019-DET-train --group seen \
        --shard 0 --nshards 2 --device cuda:0 --deadline 1760990000 --out cache/teacher_train_0.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from . import vocab


def group_prompts(group: str) -> list[str]:
    names = {"seen": vocab.SEEN, "heldout": vocab.HELDOUT, "all": vocab.SEEN + vocab.HELDOUT}[group]
    return vocab.teacher_prompts(names)


def ordered(images: list[Path], seed: int = 0) -> list[Path]:
    images = sorted(images)
    random.Random(seed).shuffle(images)
    return images


def open_teacher(name: str, device: str, log):
    from .teacher import load_teacher

    if name == "sam3":
        try:
            return load_teacher("sam3", device=device)
        except Exception as e:  # gated repo not accepted, no token, no network...
            log(f"SAM 3 unavailable ({type(e).__name__}: {str(e)[:300]}); falling back to OWLv2-large")
            name = "owlv2"
    return load_teacher(name, device=device)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--group", choices=["seen", "heldout", "all"], default="seen")
    ap.add_argument("--teacher", default="sam3", choices=["sam3", "owlv2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=0, help="unix time to stop at (0 = no limit)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    def log(msg):
        print(f"[label {Path(a.out).stem}] {msg}", flush=True)

    from PIL import Image

    prompts = group_prompts(a.group)
    imgs = ordered(list((Path(a.split) / "images").glob("*.jpg")), a.seed)
    if a.limit:
        imgs = imgs[: a.limit]
    imgs = imgs[a.shard :: a.nshards]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    meta = {}
    if out.exists():
        with np.load(out) as z:
            done = {k: z[k] for k in z.files if k != "__meta"}
            meta = json.loads(str(z["__meta"])) if "__meta" in z.files else {}
    todo = [p for p in imgs if p.stem not in done]
    log(f"{len(imgs)} frames in shard, {len(todo)} to label, {len(prompts)} prompts")
    if not todo:
        return
    if a.deadline and time.time() > a.deadline - 60:
        log("deadline already passed")
        return

    t = open_teacher(a.teacher, a.device, log)
    if meta.get("teacher") and meta["teacher"] != t.name:
        log(f"cache was made by {meta['teacher']}, now {t.name}: starting a fresh file")
        done = {}
        todo = imgs
    meta = {"teacher": t.name, "prompts": prompts, "group": a.group, "grid": a.grid, "floor": t.floor}
    t.grid = a.grid

    def save():
        np.savez_compressed(out, __meta=np.array(json.dumps(meta)), **done)

    t0 = time.time()
    n = 0
    for p in todo:
        if a.deadline and time.time() > a.deadline:
            log("deadline reached")
            break
        im = Image.open(p).convert("RGB")
        arr = t.label(im, prompts)
        if n == 0 and getattr(t, "fp16", False) and not np.isfinite(arr[:, :5]).all():
            log("fp16 produced NaN/inf, switching to fp32")
            t.fp16 = False
            t._bank = {}
            arr = t.label(im, prompts)
        done[p.stem] = arr.astype(np.float32)
        done[f"{p.stem}__shape"] = np.array(im.size, np.int32)
        n += 1
        if n % 20 == 0:
            save()
            el = time.time() - t0
            log(f"{n}/{len(todo)}  {el / n:.2f} s/frame  {len(arr)} boxes on last frame")
    save()
    log(f"labelled {n} frames in {(time.time() - t0) / 60:.1f} min with {t.name}")


if __name__ == "__main__":
    sys.exit(main())
