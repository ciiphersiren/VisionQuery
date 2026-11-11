import numpy as np

from skyquery.answer import Answerer
from skyquery.calibration import PipelineCalibration
from skyquery.classes import CLASS_TO_ID, NUM_CLASSES
from skyquery.detector import Detections
from skyquery.planner.rules import plan_rules
from skyquery.regions import in_region


def make_det():
    # 3 cars on the left (one below threshold), 1 car on the right, 1 bus right, 1 faint bicycle left
    rows = [
        (100, 100, 140, 120, 0.9, "car"),
        (200, 300, 240, 320, 0.6, "car"),
        (300, 50, 340, 70, 0.1, "car"),
        (900, 400, 940, 420, 0.8, "car"),
        (800, 100, 900, 160, 0.7, "bus"),
        (50, 600, 60, 615, 0.12, "bicycle"),
    ]
    arr = np.array([[*r[:5], CLASS_TO_ID[r[5]]] for r in rows], np.float32)
    return Detections.from_array(arr, 1000, 750)


def cal():
    return PipelineCalibration(
        count_thr=np.full(NUM_CLASSES, 0.3), exist_thr=np.full(NUM_CLASSES, 0.1), global_thr=0.3,
        exist_recall={"bicycle": 0.56},
    )


def ask(q):
    return Answerer(cal())(plan_rules(q), make_det())


def test_count_region():
    a = ask("how many cars are in the left half?")
    assert a.value == 2 and len(a.evidence) == 2


def test_count_whole():
    assert ask("how many cars?").value == 3


def test_exists_weak():
    a = ask("any bicycles in the left half?")
    assert a.value is True and a.groups[0].count == 0 and a.mood == "hedged"


def test_exists_no_with_caveat():
    a = ask("any bicycles on the right?")
    assert a.value is False and any("bicycle" in c for c in a.caveats)


def test_compare():
    a = ask("are there more cars than buses?")
    assert a.value == "cars"


def test_oov_parent():
    a = ask("how many fire trucks?")
    assert a.oov and a.oov[0]["nearest"] == ["truck"] and "fire trucks" in a.text


def test_summary():
    a = ask("what's in the image?")
    assert a.value == {"car": 3, "bus": 1}


def test_halves_partition():
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 1000, (500, 2))
    c[:5, 0] = 500  # exactly on the split line
    l = in_region(c, "left half", 1000, 1000)
    r = in_region(c, "right half", 1000, 1000)
    assert (l ^ r).all()


def test_trace_and_override():
    a = Answerer(cal())(plan_rules("how many cars on the left?"), make_det())
    row = next(r for r in a.trace if r["class"] == "car")
    assert row["detected"] == 4 and row["in_region"] == 3 and row["counted"] == 2 and row["weak"] == 1
    lo = Answerer(cal(), count_threshold=0.05)(plan_rules("how many cars on the left?"), make_det())
    assert lo.value == 3 and lo.threshold_mode == "override"


def _ovd(rows, prompts):
    arr = np.array(rows, np.float32).reshape(-1, 6)
    d = Detections.from_array(arr, 1000, 750)
    d.meta = {"model": "yoloe-test", "prompts": prompts}
    return d


def test_open_vocab_novel():
    ovd = {"dog": _ovd([(10, 10, 30, 30, 0.6, 0), (600, 10, 620, 30, 0.5, 0), (40, 40, 60, 60, 0.06, 0)], ["dog"])}
    a = Answerer(cal())(plan_rules("how many dogs on the left?"), make_det(), ovd)
    assert a.value == 1 and a.open_vocab[0]["mode"] == "novel" and a.mood == "hedged"
    assert [e.strength for e in a.evidence] == ["open"]


def test_open_vocab_subtype_only_counts_parent_boxes():
    # bus at (800,100,900,160) from the trained detector; YOLOE calls an overlapping box "school bus",
    # and a box with no trained-detector bus under it must not count
    ovd = {"school buses": _ovd([(802, 101, 899, 158, 0.07, 0), (100, 600, 160, 650, 0.8, 0)], ["school buses"])}
    a = Answerer(cal())(plan_rules("how many school buses?"), make_det(), ovd)
    assert a.value == 1 and a.open_vocab[0]["mode"] == "subtype" and a.open_vocab[0]["parent_count"] == 1


def test_open_vocab_absent_falls_back():
    a = Answerer(cal())(plan_rules("how many dogs?"), make_det())
    assert a.value is None and a.mood == "stumped"
