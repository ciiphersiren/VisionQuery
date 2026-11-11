import numpy as np

from skyquery.calibrate.cache_ovd import PROMPT_SETS, prompts_for
from skyquery.calibrate.ovd_models import GroundingDino
from skyquery.calibrate.tiles import tile_origins, yolo_lines
from skyquery.eval.coco_eval import ap_from_eval  # noqa: F401  (import check)


def test_prompt_subset_maps_back_to_visdrone_ids():
    prompts, ids = prompts_for("names", ["bus", "bicycle", "awning-tricycle"])
    assert prompts == ["bus", "bicycle", "awning-tricycle"] and ids.tolist() == [8, 2, 7]
    assert len(PROMPT_SETS["described"]) == 10


def test_grounding_dino_phrase_mapping():
    P = ["person walking or standing", "car", "tricycle with a canopy", "tricycle"]
    assert GroundingDino._index("car", P) == 1
    assert GroundingDino._index("tricycle with a canopy", P) == 2
    assert GroundingDino._index("tricycle", P) == 3
    assert GroundingDino._index("person", P) == 0
    assert GroundingDino._index("zebra", P) == -1


def test_tiles_cover_image_and_clip_boxes():
    xs = tile_origins(1920, 640, 0.2)
    assert xs[0] == 0 and xs[-1] == 1280 and all(b - a <= 512 for a, b in zip(xs, xs[1:]))
    lines = yolo_lines(np.array([[630, 10, 650, 30]], float), np.array([3]), 0, 0, 640, 640)
    assert lines == ["3 0.992188 0.031250 0.015625 0.031250"]  # half visible -> kept and clipped
    assert yolo_lines(np.array([[635, 10, 660, 30]], float), np.array([3]), 0, 0, 640, 640) == []  # 20% visible -> dropped
