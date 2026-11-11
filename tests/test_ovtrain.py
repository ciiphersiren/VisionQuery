"""Fast checks for the E3 pipeline (no model weights needed)."""

import numpy as np
from PIL import Image

from skyquery.ovtrain import vocab
from skyquery.ovtrain.build import pseudo
from skyquery.ovtrain.teacher import Teacher, drop_cut, grid_views


def test_vocab_is_clean():
    vocab.check()
    words = {s.strip() for n in vocab.SEEN for s in n.split("/")}
    for h in vocab.HELDOUT:
        assert not any(h in w or w in h for w in words), f"held-out {h!r} overlaps a training word"
    assert "tree" in words and "building" in words


def test_distractors_never_collide_with_real_words():
    from skyquery.ovtrain.trainer import DISTRACTORS

    real = {s.strip() for n in vocab.VISDRONE + vocab.SEEN + vocab.HELDOUT for s in n.split("/")}
    assert not set(DISTRACTORS) & real
    assert len(set(DISTRACTORS)) == len(DISTRACTORS)


def test_grid_views_cover_the_frame():
    w, h = 1360, 765
    views = grid_views(w, h, grid=2, overlap=0.15)
    assert views[0] == (0, 0, w, h) and len(views) == 5
    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in views[1:]:
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_drop_cut_keeps_frame_edges_drops_inner_edges():
    view = (0, 0, 700, 400)  # top-left crop of a 1360x765 frame
    boxes = np.array([[0, 0, 50, 50], [650, 10, 700, 60], [100, 100, 200, 200]], np.float32)
    keep = drop_cut(boxes, view, 1360, 765)
    assert keep.tolist() == [True, False, True]


class FakeTeacher(Teacher):
    """Returns one 'tree' box per view at the same image location, so merging must dedupe it."""

    def __init__(self, w, h):
        super().__init__(grid=2)
        self.w, self.h = w, h

    def detect_views(self, crops, prompts):
        out = []
        views = grid_views(self.w, self.h, 2, 0.15)
        for (x0, y0, x1, y1) in views:
            bx = np.array([[600, 300, 680, 380]], np.float32)  # inside every view? only where it fits
            vw, vh = x1 - x0, y1 - y0
            rel = (bx - [x0, y0, x0, y0]) / [vw, vh, vw, vh]
            ok = (rel[:, 0] >= 0) & (rel[:, 1] >= 0) & (rel[:, 2] <= 1) & (rel[:, 3] <= 1)
            out.append((rel[ok].astype(np.float32), np.full(ok.sum(), 0.9, np.float32), np.zeros(ok.sum(), np.float32)))
        return out


def test_label_merges_views_back_to_pixels():
    w, h = 1360, 765
    arr = FakeTeacher(w, h).label(Image.new("RGB", (w, h)), ["tree"])
    assert arr.shape == (1, 6)
    np.testing.assert_allclose(arr[0, :4], [600, 300, 680, 380], atol=1e-3)


def test_pseudo_filters():
    # teacher: tree (prompt 0) ok, low score, inside ignore, on top of a GT car, and an unmapped prompt
    arr = np.array([
        [10, 10, 40, 40, 0.9, 0],
        [50, 50, 80, 80, 0.3, 0],
        [210, 210, 240, 240, 0.9, 0],
        [300, 300, 340, 330, 0.9, 1],
        [400, 400, 440, 440, 0.9, 7],
    ], np.float32)
    gt = np.array([[300, 300, 340, 331]], np.float32)
    ignore = np.array([[200, 200, 260, 260]], np.float32)
    b, c = pseudo(arr, 0.5, {0: 10, 1: 11}, gt, ignore)
    assert c.tolist() == [10] and b.tolist() == [[10, 10, 40, 40]]


def test_app_picks_up_drone_weights(tmp_path, monkeypatch):
    import json

    from skyquery import config

    w = tmp_path / "yoloe-26s-drone-ov.pt"
    w.write_bytes(b"x")
    f = tmp_path / "ovd_drone.json"
    f.write_text(json.dumps({"weights": str(w), "threshold": 0.2, "adopt": True}))
    monkeypatch.setattr(config, "DRONE_OV", f)
    cfg = config.load_config()
    assert cfg.ovd_weights == str(w) and cfg.ovd_threshold == 0.2
    f.write_text(json.dumps({"weights": str(w), "threshold": 0.2, "adopt": False}))
    assert config.load_config().ovd_weights != str(w)


def test_taught_word_routing(tmp_path):
    from skyquery.openvocab import OpenVocab

    w = tmp_path / "tuned.pt"
    w.write_bytes(b"x")  # never loaded: models load lazily on first detect
    ov = OpenVocab("yoloe-26s-seg.pt", taught_weights=w)
    cases = {"trees": "tree", "Tree": "tree", "buildings": "building", "bushes": "bush", "street lights": "street lamp",
             "traffic lights": "traffic light", "zebra crossing": "crosswalk", "solar panels": "solar panel",
             "umbrellas": None, "houses": None, "fire truck": None, "people": None}
    for term, want in cases.items():
        assert ov.taught_word(term) == want, term
    assert OpenVocab("yoloe-26s-seg.pt").taught_word("trees") is None  # no tuned model -> no routing
