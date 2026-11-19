import numpy as np
import pytest

from dronekit import infer


def test_box_cut_by_interior_slice_edge_is_removed():
    W, H = 1000, 1000
    sb = np.float32([0, 0, 600, 600])                    # slice whose right and bottom edges are interior
    d = np.float32([[550, 100, 600, 150, .9, 0],          # touches interior right edge -> cut object
                    [100, 560, 150, 599, .9, 0],          # within 2 px of interior bottom edge -> cut
                    [100, 100, 150, 150, .9, 0],          # well inside -> kept
                    [0, 300, 40, 340, .9, 0]])            # touches the image border (x=0), not interior -> kept
    out = infer.edge_filter(d, np.tile(sb, (len(d), 1)), W, H)
    assert out[:, 0].tolist() == [100, 0]


def test_edge_on_image_border_is_not_interior():
    W, H = 1000, 1000
    sb = np.float32([400, 400, 1000, 1000])               # right/bottom edges are the image border
    d = np.float32([[900, 900, 1000, 1000, .9, 0],        # touches border -> kept
                    [400, 500, 450, 550, .9, 0]])         # touches interior left edge -> removed
    out = infer.edge_filter(d, np.tile(sb, (len(d), 1)), W, H)
    assert out[:, :4].tolist() == [[900, 900, 1000, 1000]]


def test_full_pass_is_never_edge_filtered():
    d = np.float32([[0, 0, 10, 10, .9, 0]])
    assert infer.edge_filter(d, None, 100, 100) is d


def test_classwise_nms():
    d1 = np.float32([[0, 0, 100, 100, .9, 0], [0, 0, 100, 100, .8, 1]])
    d2 = np.float32([[2, 2, 100, 100, .7, 0]])            # duplicate of the class-0 box
    out = infer.merge([d1, d2], "NMS", "IOU", 0.6)
    assert sorted(out[:, 4].tolist()) == pytest.approx([0.8, 0.9])


def test_apply_setup_and_cache_roundtrip(tmp_path, split_factory):
    sp = split_factory([dict(wh=(1000, 1000)), dict(wh=(1000, 1000))])
    raw = {"full@1536": {"img000": (np.float32([[10, 10, 50, 50, .9, 0]]), None),
                         "img001": (np.zeros((0, 6), np.float32), None)},
           "s512@1280": {"img000": (np.float32([[11, 11, 50, 50, .8, 0], [480, 10, 512, 40, .9, 1]]),
                                    np.float32([[0, 0, 512, 512], [0, 0, 512, 512]])),
                         "img001": (np.float32([[100, 100, 130, 130, .5, 1]]), np.float32([[0, 0, 512, 512]]))}}
    dets, _ = infer.apply_setup(raw, sp, "best_sliced")
    # img000: slice dup of the full box removed by NMS; box at x2=512 cut by interior edge -> dropped
    assert dets["img000"].tolist() == [[10, 10, 50, 50, pytest.approx(.9), 0]]
    assert len(dets["img001"]) == 1
    whole, _ = infer.apply_setup(raw, sp, "whole1536")
    assert len(whole["img000"]) == 1

    # cache roundtrip through parquet
    import json
    import pandas as pd
    rows = []
    for p, per in raw.items():
        for k, (d, sb) in per.items():
            sb = np.full((len(d), 4), np.nan, np.float32) if sb is None else sb
            for r in np.c_[d, sb]:
                rows.append([k, p, *r])
    df = pd.DataFrame(rows, columns=infer.COLS)
    df.to_parquet(tmp_path / "c.parquet")
    (tmp_path / "c.json").write_text(json.dumps({"ms_per_img": {"full@1536": 30.0, "s512@1280": 400.0}}))
    raw2, meta = infer.load_cache(tmp_path / "c")
    d2, _ = infer.apply_setup(raw2, sp, "best_sliced")
    assert np.allclose(d2["img000"], dets["img000"])
    assert infer.setup_ms(meta, "best_sliced", 1.0) == pytest.approx(431.0)


def test_cache_with_zero_detections_keeps_its_passes(tmp_path, split_factory):
    import json
    import pandas as pd
    sp = split_factory([dict()])
    pd.DataFrame(columns=infer.COLS).astype({c: "float32" for c in infer.COLS[2:]}).to_parquet(tmp_path / "e.parquet")
    (tmp_path / "e.json").write_text(json.dumps({"passes": ["s512@1280", "full@1536"],
                                                 "ms_per_img": {"s512@1280": 1.0, "full@1536": 1.0}}))
    raw, _ = infer.load_cache(tmp_path / "e")
    dets, _ = infer.apply_setup(raw, sp, "best_sliced")
    assert len(dets["img000"]) == 0


def test_wbf_fuses_matching_boxes():
    a = np.float32([[100, 100, 200, 200, .9, 0]])
    b = np.float32([[102, 102, 202, 202, .7, 0]])
    out = infer.wbf([a, b], 1000, 1000, thr=0.65)
    assert len(out) == 1 and 100 <= out[0, 0] <= 102
