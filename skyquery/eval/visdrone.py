"""VisDrone2019-DET annotation loading.

Annotation line: x, y, w, h, score, category, truncation, occlusion
  * category 1..10 -> detector classes 0..9
  * category 0     -> ignored region (treated as crowd for AP, and used to
                      discard detections for ignore-aware counting)
  * category 11    -> "others", dropped
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..classes import NUM_CLASSES, VISDRONE_IGNORE_CAT, visdrone_cat_to_id


@dataclass
class ImageGT:
    stem: str
    boxes: np.ndarray  # (N, 4) xyxy
    cls: np.ndarray  # (N,)
    ignore: np.ndarray  # (M, 4) xyxy ignore regions

    @property
    def sequence(self) -> str:
        return sequence_of(self.stem)

    def counts(self) -> np.ndarray:
        return np.bincount(self.cls, minlength=NUM_CLASSES)


def sequence_of(stem: str) -> str:
    """VisDrone frames are named <seq>_<frame>_<...>; frames in a sequence are correlated."""
    return stem.split("_")[0]


def load_split(split_dir: str | Path) -> dict[str, ImageGT]:
    split_dir = Path(split_dir)
    out = {}
    for f in sorted((split_dir / "annotations").glob("*.txt")):
        rows = []
        for line in f.read_text().strip().splitlines():
            vals = [int(v) for v in line.strip().rstrip(",").split(",")[:8]]
            rows.append(vals)
        a = np.array(rows, np.int64).reshape(-1, 8)
        xyxy = np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], 1).astype(np.float32) if len(a) else np.zeros((0, 4), np.float32)
        cat = a[:, 5] if len(a) else np.zeros(0, np.int64)
        keep = (cat >= 1) & (cat <= 10)
        ign = cat == VISDRONE_IGNORE_CAT
        out[f.stem] = ImageGT(
            stem=f.stem,
            boxes=xyxy[keep],
            cls=np.array([visdrone_cat_to_id(int(c)) for c in cat[keep]], np.int64),
            ignore=xyxy[ign],
        )
    return out


def in_ignore(xyxy: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """True for boxes whose centre lies inside any ignore region."""
    if len(xyxy) == 0 or len(ignore) == 0:
        return np.zeros(len(xyxy), bool)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    inside = (
        (cx[:, None] >= ignore[None, :, 0]) & (cx[:, None] <= ignore[None, :, 2])
        & (cy[:, None] >= ignore[None, :, 1]) & (cy[:, None] <= ignore[None, :, 3])
    )
    return inside.any(1)
