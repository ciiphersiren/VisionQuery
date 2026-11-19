import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronekit.data import ImageRecord, Split  # noqa: E402


def make_split(images, class_names=("a", "b")):
    """images: list of dict(boxes=[[cls,x1,y1,x2,y2],...], ignore=[[x1,y1,x2,y2],...], wh=(W,H))."""
    recs = []
    for i, im in enumerate(images):
        b = np.array([list(r) + [0, 0] for r in im.get("boxes", [])], np.float32).reshape(-1, 7)
        g = np.array(im.get("ignore", []), np.float32).reshape(-1, 4)
        W, H = im.get("wh", (1000, 1000))
        recs.append(ImageRecord(f"img{i:03d}", Path(f"img{i:03d}.jpg"), W, H, "seq0", b, g))
    return Split("toy", "toy", list(class_names), recs)


@pytest.fixture
def split_factory():
    return make_split
