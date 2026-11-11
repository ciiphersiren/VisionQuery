"""Zero-shot prompt-based (open-vocabulary) predictions, cached for the COCO harness.

Each prompt set maps one text prompt to each VisDrone class id (or, with
--classes, to a subset, as in the held-out-class experiment E2).

    python -m skyquery.calibrate.cache_ovd --split data/VisDrone2019-DET-val \
        --model yoloe-26s --prompts names --out cache/val_yoloe26s_names.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..classes import CLASSES, CLASS_TO_ID
from .ovd_models import ALL_MODELS, load_ovd

PROMPT_SETS: dict[str, list[str]] = {
    # the dataset's own class names
    "names": ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"],
    # plain-English descriptions of VisDrone's annotation conventions
    "described": [
        "person walking or standing",
        "person sitting or riding",
        "bicycle",
        "car",
        "van",
        "truck",
        "three-wheeled tricycle",
        "tricycle with a canopy",
        "bus",
        "motorcycle",
    ],
}
assert all(len(v) == len(CLASSES) for v in PROMPT_SETS.values())


def prompts_for(prompt_set: str, classes: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Prompts to send and, per prompt, the VisDrone class id it stands for."""
    cls = classes or list(CLASSES)
    ids = np.array([CLASS_TO_ID[c] for c in cls])
    return [PROMPT_SETS[prompt_set][i] for i in ids], ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="yoloe-26s", choices=ALL_MODELS)
    ap.add_argument("--prompts", choices=list(PROMPT_SETS), required=True)
    ap.add_argument("--classes", nargs="*", help="prompt only these VisDrone classes (default: all ten)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    m = load_ovd(a.model, device=a.device, imgsz=a.imgsz)
    prompts, ids = prompts_for(a.prompts, a.classes)
    imgs = sorted((Path(a.split) / "images").glob("*.jpg"))[: a.limit or None]
    out = Path(a.out)
    done: dict[str, np.ndarray] = {}
    if out.exists():
        with np.load(out) as z:
            done = {k: z[k] for k in z.files}
    todo = [p for p in imgs if p.stem not in done]
    print(f"{a.model}/{a.prompts}: {len(imgs)} images, {len(todo)} to run", flush=True)
    t0 = time.time()
    from PIL import Image

    for i, p in enumerate(todo, 1):
        arr = m.predict(str(p), prompts)
        arr[:, 5] = ids[arr[:, 5].astype(int)]  # prompt index -> VisDrone class id
        done[p.stem] = arr.astype(np.float32)
        done[f"{p.stem}__shape"] = np.array(Image.open(p).size, np.int32)
        if i % 25 == 0 or i == len(todo):
            np.savez_compressed(out, **done)
            el = time.time() - t0
            print(f"[{a.model}/{a.prompts}] {i}/{len(todo)} {el/i:.2f}s/img eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
