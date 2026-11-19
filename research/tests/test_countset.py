import numpy as np
import pandas as pd

from dronekit import countset as K
from dronekit.data import ImageRecord, Split


def _split():
    # 100 x 100. Classes: pedestrian, people, car. Object D sits in the ignore region; object E on the x = 50 border.
    b = [[0, 10, 10, 20, 20, 0, 0],    # A pedestrian, top left
         [1, 70, 10, 80, 20, 1, 1],    # B people, top right
         [2, 10, 70, 30, 90, 2, 0],    # C car, bottom left
         [2, 2, 2, 8, 8, 0, 0],        # D car, centered inside ignore
         [2, 45, 40, 55, 60, 0, 0]]    # E car, center exactly on x = 50, y = 50
    r = ImageRecord("im", "im.jpg", 100, 100, "s1", np.float32(b), np.float32([[0, 0, 10, 10]]))
    return Split("visdrone", "val", ["pedestrian", "people", "car"], [r])


def _get(rows, target, region):
    return next(x for x in rows if x["target"] == target and x["region"] == region)


def test_counts_centers_and_ignore():
    rows = K.records(_split())
    assert _get(rows, "car", "whole")["count"] == 2                # D excluded by its center
    assert _get(rows, "people", "whole")["count"] == 2             # group: pedestrian + people
    assert _get(rows, "pedestrian", "left")["count"] == 1
    assert _get(rows, "car", "left")["count"] == 1 and _get(rows, "car", "right")["count"] == 1   # E counted once, on the right
    assert sum(_get(rows, "car", q)["count"] for q in ("q1", "q2", "q3", "q4")) == 2


def test_stratification_fields():
    rows = K.records(_split())
    r = _get(rows, "people", "whole")
    assert r["count_bin"] == "1-8" and r["median_side"] == 10 and np.isclose(r["median_area_frac"], 0.01)
    assert np.isclose(r["density"], 60 / 10)                       # the two centers are 60 px apart, sides 10
    assert r["frac_occluded"] == 0.5 and r["frac_truncated"] == 0.5     # B is occluded and truncated, A is neither
    assert np.isclose(_get(rows, "car", "whole")["frac_occluded"], 1 / 2)
    e = _get(rows, "pedestrian", "right")
    assert e["count"] == 0 and e["count_bin"] == "0" and np.isnan(e["median_side"]) and np.isnan(e["density"])


def test_heavily_ignored_region_is_dropped():
    sp = _split()
    sp.images[0].ignore = np.float32([[0, 0, 50, 50]])             # all of q1 and 25% of the image
    regions = {x["region"] for x in K.records(sp)}
    assert "q1" not in regions and "left" not in regions and "whole" not in regions and "q2" in regions
    sp.images[0].ignore = np.float32([[0, 0, 10, 10]])             # 1% of the image
    assert "whole" in {x["region"] for x in K.records(sp)}


def test_groups_need_two_classes():
    assert set(K.targets(["car", "truck", "bus"])) == {"car", "truck", "bus", "vehicles"}
    assert set(K.targets(["car"])) == {"car"}


def test_bins():
    assert [K.count_bin(n) for n in (0, 1, 8, 9, 20, 21, 50, 51)] == ["0", "1-8", "1-8", "9-20", "9-20", "21-50", "21-50", ">50"]


def test_tertiles_frozen_on_dev():
    e = K.tertile_edges([1, 2, 3, 4, 5, 6, np.nan])
    assert list(K.tertile([0, 3.5, 99, np.nan], e)) == ["low", "mid", "high", "none"]


def test_count_dets_same_rules_as_gt():
    r = _split().images[0]
    # (x1, y1, x2, y2, score, cls): in ignore; left, kept; left, low score; right; other class
    d = np.float32([[2, 2, 8, 8, .9, 2], [10, 70, 30, 90, .9, 2], [10, 60, 30, 80, .1, 2], [60, 60, 70, 70, .9, 2], [10, 10, 20, 20, .9, 0]])
    assert K.count_dets(d, (100, 100), r.ignore, [2], "whole", {2: .5}) == 2
    assert K.count_dets(d, (100, 100), r.ignore, [2], "left", {2: .5}) == 1
    assert K.count_dets(d, (100, 100), r.ignore, [2], "left", {2: .05}) == 2
    assert K.count_dets(d, (100, 100), r.ignore, [0, 2], "whole", {2: .5}) == 2     # class 0 has no threshold: not counted
    assert K.count_dets(np.zeros((0, 6)), (100, 100), r.ignore, [2]) == 0


def test_subset_is_fixed_balanced_and_unique():
    rng = np.random.default_rng(0)
    rows = [dict(dataset=d, image=f"{d}{i}", region="whole", count=int(rng.integers(1, 80)), count_bin="x", size_bin=rng.choice(["low", "high"]), target="car")
            for d in ("a", "b") for i in range(200)]
    df = pd.DataFrame(rows)
    df["count_bin"] = [K.count_bin(n) for n in df["count"]]
    s = K.stratified_subset(df, 60)
    assert s == K.stratified_subset(df, 60) and len(s) == len(set(s)) == 60
    n = pd.Series([d for d, _ in s]).value_counts()
    assert abs(n["a"] - n["b"]) <= 6                               # filled cell by cell, so datasets differ by at most a few rounds


def test_tune_thresholds_picks_lowest_mae():
    sp = _split()
    # car: 2 real objects; one extra detection at 0.2 makes low thresholds overcount
    d = {"im": np.float32([[10, 70, 30, 90, .9, 2], [45, 40, 55, 60, .8, 2], [60, 60, 70, 70, .2, 2], [2, 2, 8, 8, .95, 2]])}
    assert K.tune_thresholds(sp, d, [2])[2] == 0.21                # first grid point above 0.2; the box in ignore never counts


def test_paired_records_use_same_rules_and_specs():
    sp = _split()
    d = {"im": np.float32([[10, 70, 30, 90, .9, 2], [60, 60, 70, 70, .2, 2], [2, 2, 8, 8, .95, 2], [10, 10, 20, 20, .9, 0]])}
    rows = K.paired_records(sp, d, {2: .5}, {"car": [2]}, {"car": [2]}, ("whole", "left"))
    w = next(r for r in rows if r["region"] == "whole")
    assert (w["y"], w["yhat"]) == (2, 1) and np.isclose(w["pred_side"], 20) and np.isnan(w["pred_density"])
    assert next(r for r in rows if r["region"] == "left")["yhat"] == 1
