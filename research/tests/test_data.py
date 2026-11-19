import numpy as np
import pytest

from dronekit import data


def test_parse_visdrone_rules():
    txt = "\n".join([
        "10,20,30,40,1,4,0,1",     # car (cat 4 -> class 3), trunc 0, occ 1
        "0,0,50,50,0,1,0,0",       # score 0 -> ignore region
        "5,5,10,10,1,0,0,0",       # category 0 -> ignore region
        "1,1,5,5,1,11,0,0",        # others -> dropped
        "1,1,0,5,1,3,0,0",         # zero width -> dropped
        "7,8,2,3,1,10,1,2,",       # trailing comma; motor (class 9)
    ])
    boxes, ign = data.parse_visdrone_txt(txt)
    assert boxes.tolist() == [[3, 10, 20, 40, 60, 1, 0], [9, 7, 8, 9, 11, 2, 1]]
    assert ign.tolist() == [[0, 0, 50, 50], [5, 5, 15, 15]]


def test_sequence_id():
    assert data.visdrone_sequence("0000001_02999_d_0000005") == "0000001"
    assert data.visdrone_sequence("9999938_00000_d_0000207") == "9999938"


def test_split_stats(split_factory):
    sp = split_factory([dict(boxes=[[0, 0, 0, 10, 10], [1, 0, 0, 40, 40]])])
    st = data.split_stats(sp)
    assert st["objects"] == 2 and np.isclose(st["tiny"], 0.5) and np.isclose(st["medium_plus"], 0.5)


def _row(alt, wx, objects, tiny, truck, bus):
    return dict(altitude=alt, weather=wx, objects=objects, tiny=tiny, share_truck=truck, share_bus=bus)


def test_pick_val_sequences_follows_alloc_and_rare_rule():
    table = dict(a=_row("low", "day", 100, 0.2, 0.0, 0.0), b=_row("low", "day", 100, 0.2, 0.1, 0.1),
                 c=_row("med", "fog", 100, 0.2, 0.1, 0.1), d=_row("med", "fog", 100, 0.2, 0.0, 0.0),
                 e=_row("high", "day", 100, 0.9, 0.5, 0.5))
    # e is outside the allocation; a and d carry no rare class, so only b + c meets the rule
    seqs, _ = data.pick_val_sequences(table, {("low", "day"): 1, ("med", "fog"): 1}, min_rare_seqs=2)
    assert seqs == ["b", "c"]
    with pytest.raises(ValueError):
        data.pick_val_sequences(table, {("low", "day"): 1, ("med", "fog"): 1}, min_rare_seqs=3)


def test_pick_val_sequences_closest_to_pool_and_deterministic():
    table = dict(a=_row("low", "day", 100, 0.5, 0.1, 0.1), b=_row("low", "day", 100, 0.1, 0.1, 0.1),
                 c=_row("low", "day", 100, 0.1, 0.1, 0.1))     # pool tiny share = 0.233: b and c tie, b wins by name
    assert data.pick_val_sequences(table, {("low", "day"): 1}, min_rare_seqs=1)[0] == ["b"]
