import numpy as np
import pytest

from dronekit import data, score


def dets(rows):
    return np.array(rows, np.float32).reshape(-1, 6)


def test_box_inside_ignore_region_is_neither_tp_nor_fp(split_factory):
    sp = split_factory([dict(boxes=[[0, 100, 100, 200, 200]], ignore=[[500, 500, 700, 700]])])
    d = {"img000": dets([[100, 100, 200, 200, 0.9, 0],      # true positive
                         [550, 550, 600, 600, 0.95, 0]])}   # fully inside the ignore region, highest score
    m, pc, E = score.evaluate(sp, d)
    assert m["AP50"] == pytest.approx(1.0)                  # an FP ranked first would cut AP below 1
    pr = score.pr_at_conf(E, 0.0)
    assert (pr["tp"], pr["fp"], pr["n_gt"]) == (1, 0, 1)    # ignored det counted as neither
    cm = score.confusion_matrix(sp, d, conf=0.0)
    assert cm[-1].sum() == 0                                # no background (FP) entry either


def test_same_box_outside_ignore_region_is_fp(split_factory):
    # control: the identical detection without an ignore region does count as a false positive
    sp = split_factory([dict(boxes=[[0, 100, 100, 200, 200]])])
    d = {"img000": dets([[100, 100, 200, 200, 0.9, 0], [550, 550, 600, 600, 0.95, 0]])}
    m, _, E = score.evaluate(sp, d)
    assert m["AP50"] < 1.0
    assert score.pr_at_conf(E, 0.0)["fp"] == 1


def test_without_ignore_turns_ignored_det_into_fp(split_factory):
    sp = split_factory([dict(boxes=[[0, 100, 100, 200, 200]], ignore=[[500, 500, 700, 700]])])
    d = {"img000": dets([[100, 100, 200, 200, 0.9, 0], [550, 550, 600, 600, 0.95, 0]])}
    assert score.evaluate(sp, d)[0]["AP50"] == pytest.approx(1.0)
    assert score.pr_at_conf(score.evaluate(data.without_ignore(sp), d)[2], 0.0)["fp"] == 1
    assert len(sp.images[0].ignore) == 1                    # the original split is untouched


def test_ignore_share_and_per_class_ap(split_factory):
    sp = split_factory([dict(boxes=[[0, 100, 100, 200, 200]], ignore=[[500, 500, 700, 700]])])
    d = {"img000": dets([[100, 100, 200, 200, 0.9, 0], [550, 550, 600, 600, 0.8, 0], [550, 550, 600, 600, 0.1, 0]])}
    assert score.ignore_share(sp, d, conf=0.25) == (0.5, 2)  # the 0.1 detection is below conf
    ap = score.per_class_ap(score.run_cocoeval(sp, d), sp.class_names)
    assert ap["a"] == pytest.approx(1.0) and np.isnan(ap["b"])


def test_by_sequence(split_factory):
    sp = split_factory([dict(), dict(), dict()])
    sp.images[2].sequence = "seq1"
    parts = data.by_sequence(sp)
    assert {k: len(v) for k, v in parts.items()} == {"seq0": 2, "seq1": 1}


def test_hand_computed_ap_three_boxes(split_factory):
    # 2 GT, 3 detections ranked: TP (0.9), FP (0.8), TP (0.7). Exact boxes -> same at every IoU.
    # precision = 1, 1/2, 2/3 at recall = 1/2, 1/2, 1. COCO envelope: 1 for r <= 0.5 (51 of the
    # 101 recall points), 2/3 for r in 0.51..1.00 (50 points).  AP = (51 + 50*2/3) / 101.
    sp = split_factory([dict(boxes=[[0, 100, 100, 200, 200], [0, 400, 400, 500, 500]])])
    d = {"img000": dets([[100, 100, 200, 200, 0.9, 0],
                         [700, 700, 800, 800, 0.8, 0],
                         [400, 400, 500, 500, 0.7, 0]])}
    m, pc, E = score.evaluate(sp, d)
    expected = (51 + 50 * 2 / 3) / 101
    assert pc["a"] == pytest.approx(expected, abs=1e-6)
    assert m["AP50"] == pytest.approx(expected, abs=1e-6)   # class "b" has no GT -> excluded
    assert m["AP"] == pytest.approx(expected, abs=1e-6)
    assert np.isnan(pc["b"])
    pr = score.pr_at_conf(E, 0.75)                          # keeps TP 0.9 and FP 0.8
    assert pr["precision"] == pytest.approx(0.5) and pr["recall"] == pytest.approx(0.5)


def test_size_buckets(split_factory):
    # one tiny (10x10), one small (20x20), one medium+ (40x40) object, all detected exactly
    sp = split_factory([dict(boxes=[[0, 0, 0, 10, 10], [0, 100, 100, 120, 120], [0, 300, 300, 340, 340]])])
    d = {"img000": dets([[0, 0, 10, 10, .9, 0], [100, 100, 120, 120, .8, 0], [300, 300, 340, 340, .7, 0]])}
    m, _, _ = score.evaluate(sp, d)
    for k in ("AP50_tiny", "AP50_small", "AP50_medium+", "AP50"):
        assert m[k] == pytest.approx(1.0), k


def test_max_dets_500_is_used(split_factory):
    # 200 GT objects, all detected: with maxDets 100 recall would cap at 0.5
    boxes = [[0, 20 * (i % 20), 20 * (i // 20) * 2, 20 * (i % 20) + 15, 20 * (i // 20) * 2 + 15] for i in range(200)]
    sp = split_factory([dict(boxes=boxes, wh=(1000, 1000))])
    d = {"img000": dets([[b[1], b[2], b[3], b[4], 0.5 + i / 1000, 0] for i, b in enumerate(boxes)])}
    m, _, E = score.evaluate(sp, d)
    assert m["AP50"] == pytest.approx(1.0)
    assert score.pr_at_conf(E, 0.0)["recall"] == pytest.approx(1.0)


def test_confusion_matrix_counts(split_factory):
    sp = split_factory([dict(boxes=[[0, 0, 0, 50, 50], [1, 100, 100, 150, 150], [1, 300, 300, 350, 350]])])
    d = {"img000": dets([[0, 0, 50, 50, .9, 0],            # a -> a
                         [100, 100, 150, 150, .8, 0],      # b predicted as a
                         [600, 600, 650, 650, .7, 1]])}    # background FP as b; GT at 300 missed
    cm = score.confusion_matrix(sp, d, conf=0.5)
    assert cm.tolist() == [[1, 0, 0], [1, 0, 1], [0, 1, 0]]
    assert np.allclose(score.normalize_rows(cm)[1], [0.5, 0, 0.5])
