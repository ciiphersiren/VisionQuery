import numpy as np
import pytest

from dronekit import data, tiles


def _uavdt_fixture(tmp_path):
    gt = tmp_path / "GT"; gt.mkdir()
    # frame, id, x, y, w, h, out-of-view, occlusion, category
    (gt / "M0101_gt_whole.txt").write_text("1,1,10,20,30,40,1,1,1\n1,2,100,100,50,50,3,2,3\n2,1,12,20,30,40,1,4,2\n3,1,14,20,30,40,1,1,1\n")
    (gt / "M0101_gt_ignore.txt").write_text("1,9,500,300,40,20,1,-1,-1\n")
    (tmp_path / "M_attr/train").mkdir(parents=True); (tmp_path / "M_attr/test").mkdir(parents=True)
    (tmp_path / "M_attr/train/M0101_attr.txt").write_text("0,1,0,0,0,1,0,1,0,0\n")
    (tmp_path / "M_attr/test/M0701 _attr.txt").write_text("0,0,1,0,0,1,1,0,0,0\n")   # stray space, as released
    return gt


def test_uavdt_loader(tmp_path):
    gt = _uavdt_fixture(tmp_path)
    attrs, split = data.read_uavdt_attrs(tmp_path / "M_attr")
    assert split == {"M0101": "train", "M0701": "test"}
    assert attrs["M0101"]["weather"] == "night" and attrs["M0101"]["altitude"] == "high-alt"
    sp = data.load_uavdt(tmp_path / "imgs", gt, tmp_path / "M_attr", ["M0101"], stride=1, size=(1080, 540))
    assert [r.key for r in sp.images] == ["M0101_img000001", "M0101_img000002", "M0101_img000003"]
    r = sp.images[0]
    assert r.boxes.tolist() == [[0, 10, 20, 40, 60, 0, 0], [2, 100, 100, 150, 150, 3, 1]]   # occ large=3, out small=1
    assert r.ignore.tolist() == [[500, 300, 540, 320]] and r.sequence == "M0101" and r.attrs["altitude"] == "high-alt"
    assert sp.images[1].boxes[0, 0] == 1 and sp.images[1].boxes[0, 5] == 1                   # truck, small occlusion
    sp2 = data.load_uavdt(tmp_path / "imgs", gt, tmp_path / "M_attr", ["M0101"], stride=2, size=(1080, 540))
    assert [r.key for r in sp2.images] == ["M0101_img000001", "M0101_img000003"]
    # image files with no gt_whole row (frames 4, 5) are unlabeled and must not enter the split as empty frames
    (tmp_path / "imgs/M0101").mkdir(parents=True)
    for f in range(1, 6):
        (tmp_path / f"imgs/M0101/img{f:06d}.jpg").write_bytes(b"")
    sp3 = data.load_uavdt(tmp_path / "imgs", gt, tmp_path / "M_attr", ["M0101"], stride=1, size=(1080, 540))
    assert [r.key for r in sp3.images] == ["M0101_img000001", "M0101_img000002", "M0101_img000003"]


def test_remap_van_to_car_and_van_ignored(split_factory):
    names = ["car", "van", "pedestrian"]
    sp = split_factory([dict(boxes=[[0, 0, 0, 10, 10], [1, 20, 20, 30, 30], [2, 40, 40, 50, 50]])], class_names=names)
    shared = ["car", "truck", "bus"]
    a = data.remap_split(sp, {"car": "car", "van": "car"}, shared)
    assert a.images[0].boxes[:, 0].tolist() == [0, 0] and len(a.images[0].ignore) == 0     # pedestrian dropped
    b = data.remap_split(sp, {"car": "car", "van": "ignore"}, shared)
    assert b.images[0].boxes[:, 0].tolist() == [0] and b.images[0].ignore.tolist() == [[20, 20, 30, 30]]
    d = {"img000": np.float32([[0, 0, 10, 10, .9, 0], [20, 20, 30, 30, .8, 1], [40, 40, 50, 50, .7, 2]])}
    assert data.remap_dets(d, names, {"car": "car", "van": "car"}, shared)["img000"][:, 5].tolist() == [0, 0]
    assert data.remap_dets(d, names, {"car": "car", "van": "ignore"}, shared)["img000"][:, 5].tolist() == [0]


def test_tile_starts_cover_the_image():
    assert tiles.starts(540, 640, 480) == [0]
    assert tiles.starts(1080, 640, 480) == [0, 440]
    assert tiles.starts(1080, 960, 720) == [0, 120]


def test_sf_tile_rules(split_factory):
    # 1080 x 540 -> tiles x 0-640 and x 440-1080. Box A: 10/30 of its width in the first tile (33% >= 30%: kept, clipped).
    # Box B: 5/30 in the first tile (17%: dropped there). Box C: 1 px wide after clipping (below 2 px: dropped there).
    sp = split_factory([dict(boxes=[[0, 630, 100, 660, 130], [1, 635, 200, 665, 230], [0, 639, 300, 641, 330]], wh=(1080, 540))])
    plan = {p[0]: p[4] for p in tiles.sf_tile_plan(sp.images[0], 1080, 540)}
    assert set(plan) == {0, 440}
    assert len(plan[0]) == 1 and plan[0][0].startswith("0 ")
    cx, cy, w, h = map(float, plan[0][0].split()[1:])
    assert np.isclose(w * 640, 10, atol=0.01) and np.isclose(cx * 640, 635, atol=0.01) and np.isclose(h * 540, 30, atol=0.01)
    assert len(plan[440]) == 3                                              # all three lie fully inside the second tile


def test_build_sf_tiny(tmp_path, split_factory):
    import cv2
    img = tmp_path / "img000.jpg"
    cv2.imwrite(str(img), np.full((540, 1080, 3), 50, np.uint8))
    sp = split_factory([dict(boxes=[[0, 100, 100, 140, 130], [1, 700, 200, 740, 230]], ignore=[[0, 0, 20, 20]], wh=(1080, 540))])
    sp.images[0].path = img
    assert tiles.build_sf(sp, tmp_path / "out", workers=0) == (1, 2)
    assert len((tmp_path / "out/full/labels/img000.txt").read_text().splitlines()) == 2
    painted = cv2.imread(str(tmp_path / "out/tiles/images/img000_0_0.jpg"))
    assert painted.shape[:2] == (540, 640) and abs(int(painted[5, 5, 0]) - 114) <= 2


def test_build_d2_tiny(tmp_path, split_factory):
    import cv2
    img = tmp_path / "img000.jpg"
    cv2.imwrite(str(img), np.full((540, 1080, 3), 50, np.uint8))
    sp = split_factory([dict(boxes=[[0, 100, 100, 140, 130], [1, 630, 200, 660, 230]], ignore=[[0, 0, 20, 20]], wh=(1080, 540))])
    sp.images[0].path = img
    n_full, n_tiles = tiles.build_d2(sp, tmp_path / "out", workers=0)
    assert n_full == 1 and n_tiles >= 2
    lab = (tmp_path / "out/full/labels/img000.txt").read_text().splitlines()
    assert len(lab) == 2 and lab[0].startswith("0 ")
    painted = cv2.imread(str(tmp_path / "out/full/images/img000.jpg"))
    assert abs(int(painted[5, 5, 0]) - 114) <= 2                           # ignore region painted gray
    # box at x 630-660 is cut by the 640 tile [0, 640): 10/30 visible < 60% -> grayed, not labelled in that tile
    t0 = (tmp_path / "out/tiles/labels/img000_t640_0_0.txt").read_text().splitlines()
    assert len(t0) == 1
