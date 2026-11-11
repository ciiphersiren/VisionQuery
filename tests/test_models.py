import json

import numpy as np
import pytest

from skyquery.answer import Answerer
from skyquery.calibration import PipelineCalibration, load_calibration
from skyquery.classes import CLASS_TO_ID, CLASSES
from skyquery.config import ROOT
from skyquery.detector import Detections, Detector
from skyquery.models import load_models
from skyquery.planner.rules import plan_rules

REG = ROOT / "configs" / "models.json"


def test_registry_lists_every_model_and_what_is_missing():
    models, default = load_models(REG, ROOT, "", "")
    assert default == "yolo26s_visdrone" and models[default].available
    assert set(models) == {"yolo26s_visdrone", "yolo26m_tile_visdrone", "ensemble_visdrone", "yolo26m_tile_uavdt", "yolo26s_uavdt"}
    assert models["yolo26m_tile_uavdt"].classes == ["car", "truck", "bus"]
    for m in models.values():
        assert load_calibration(m.calibration)[m.default_pipeline] is not None
        if not m.available:
            assert m.missing and all(n.endswith(".pt") for n in m.missing)


def test_registry_fallback_without_file(tmp_path):
    models, default = load_models(tmp_path / "none.json", ROOT, "w.pt", "c.json")
    assert list(models) == ["default"] and models["default"].pipelines == ["fi1280", "fi1280+fi1920"]


def test_missing_classes_get_no_threshold_and_conformal_interval():
    cal = PipelineCalibration.from_dict({"count_thresholds": {"car": 0.4, "truck": 0.5, "bus": 0.9},
                                         "conformal": {"q": {"car": 1.8974}, "label": "90% range"}})
    assert np.isnan(cal.count_thr[CLASS_TO_ID["pedestrian"]]) and cal.count_thr[CLASS_TO_ID["car"]] == 0.4
    assert cal.interval("car", 20) == (12, 28)              # 20 +/- 1.8974 * sqrt(21) = 11.3 .. 28.7
    assert cal.interval("car", 0) == (0, 1)
    assert cal.interval("truck", 3) == (3, 3) and cal.interval_label == "90% range"


def _uavdt_answerer(threshold=None):
    cal = PipelineCalibration.from_dict({"count_thresholds": {"car": 0.3, "truck": 0.3, "bus": 0.3}})
    return Answerer(cal, threshold, supported=["car", "truck", "bus"], model_name="YOLO26m-Tile (UAVDT)")


def _det():
    rows = [(100, 100, 140, 120, 0.9, "car"), (600, 100, 640, 120, 0.8, "car"), (300, 300, 380, 340, 0.7, "truck")]
    return Detections.from_array(np.array([[*r[:5], CLASS_TO_ID[r[5]]] for r in rows], np.float32), 1000, 750)


def test_class_the_model_never_learned_is_not_answered_as_zero():
    a = _uavdt_answerer()(plan_rules("how many pedestrians are there?"), _det())
    assert a.value is None and "can't look for" in a.text and "UAVDT" in a.text and a.mood == "stumped"


def test_group_is_cut_to_known_classes_with_a_caveat():
    a = _uavdt_answerer()(plan_rules("how many vehicles?"), _det())
    assert a.value == 3 and a.groups[0].classes == ["car", "truck", "bus"]
    assert any("only knows" in c for c in a.caveats)


def test_mixed_question_answers_the_known_part():
    a = _uavdt_answerer()(plan_rules("are there more cars than bicycles?"), _det())
    assert "2" in a.text and "can't look for bicycles" in a.text


def test_summary_and_slider_only_touch_known_classes():
    a = _uavdt_answerer(0.75)(plan_rules("what's in the image?"), _det())
    assert a.value == {"car": 2} and a.threshold_mode == "override"
    assert all(t["class"] in {"car", "truck", "bus"} for t in a.trace)


def test_conformal_range_shows_in_the_answer():
    cal = PipelineCalibration.from_dict({"count_thresholds": {c: 0.3 for c in CLASSES}, "conformal": {"q": {"car": 1.8974}, "label": "90% range"}})
    a = Answerer(cal)(plan_rules("how many cars?"), _det())
    assert a.value == 2 and "(90% range 0–5)" in a.text        # 2 +/- 1.8974 * sqrt(3) = -1.3 .. 5.3


@pytest.fixture(scope="module")
def three_class_weights(tmp_path_factory):
    import torch
    from ultralytics.nn.tasks import DetectionModel

    m = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    m.names = {0: "car", 1: "truck", 2: "bus"}
    p = tmp_path_factory.mktemp("w") / "uavdt_like.pt"
    torch.save({"model": m, "date": "test", "version": "test", "train_args": {}}, p)
    return p


def test_uavdt_class_ids_become_visdrone_ids(three_class_weights):
    d = Detector(three_class_weights, pipeline="fi1024", conf_floor=0.001)
    assert d.luts[0].tolist() == [CLASS_TO_ID["car"], CLASS_TO_ID["truck"], CLASS_TO_ID["bus"]] and d.classes == ["car", "truck", "bus"]
    det = d(np.random.default_rng(0).integers(0, 255, (540, 1024, 3), dtype=np.uint8))
    assert set(np.unique(det.cls)) <= {CLASS_TO_ID["car"], CLASS_TO_ID["truck"], CLASS_TO_ID["bus"]}


@pytest.mark.slow
def test_sliced_and_ensemble_pipelines_run_on_a_real_frame():
    import cv2

    models, _ = load_models(REG, ROOT, "", "")
    w = models["yolo26s_visdrone"].weights[0]
    img = cv2.imread(str(sorted((ROOT / "samples").glob("*.jpg"))[0]))
    whole = Detector(w, pipeline="fi1280")(img)
    sliced = Detector(w, pipeline="sliced")(img)
    ens = Detector([w, w], pipeline="sliced")(img)
    assert len(sliced) > 0 and len(whole) > 0 and len(ens) > 0
    assert sliced.xyxy[:, 2].max() <= img.shape[1] + 1 and sliced.xyxy[:, 3].max() <= img.shape[0] + 1
    assert len(ens) <= 2 * len(sliced)
