"""Open-vocabulary branch: text-prompted detection with YOLOE-26.

The fine-tuned YOLO26s stays the authority for its 10 VisDrone classes. This
branch only handles what it cannot see:

  * novel     "how many dogs?"        -> YOLOE prompted with "dog"
  * subtype   "how many fire trucks?" -> YOLOE prompted with "fire truck" alone, at a
                                         low threshold; only fine-tuned truck boxes it
                                         overlaps (IoU >= 0.5) count. (Adding "truck" as a
                                         contrast prompt sends every box to "truck".)

Two YOLOE models can sit behind it (experiment E3, reports/ovd_e3_val.json):

  * zero-shot YOLOE-26s answers any word, weakly on drone footage;
  * the drone-tuned YOLOE-26s learned 35 drone-view concepts from SAM 3 labels
    (tree 1.1 -> 52.1 AP against SAM 3) but stopped answering to words it was
    not taught, even synonyms ("person", "house"). So it is only used for the
    words it was taught, matched through the training synonyms and a plural
    rule, and always prompted with the exact training word.

The full chain for a question: VisDrone class -> fine-tuned YOLO26s; taught word
-> drone-tuned YOLOE (and, if it finds nothing, zero-shot YOLOE as a last
resort); any other word -> zero-shot YOLOE.

Zero-shot scores are not calibrated for drone imagery, and the tuned model is
only checked against SAM 3, so every open-vocabulary answer is labelled as a lead.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np

from .detector import Detections


def singular(word: str) -> str:
    w = word.strip().lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if re.search(r"(ches|shes|sses|xes)$", w):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def taught_lookup(names: list[str]) -> dict[str, str]:
    """Every synonym (and its plural) -> the canonical training prompt."""
    out = {}
    for n in names:
        canon = n.split("/")[0].strip()
        for s in n.split("/"):
            s = s.strip().lower()
            out[s] = canon
    return out


class _YOLOE:
    def __init__(self, weights: str):
        self.weights = str(weights)
        self._model = None
        self._pe: dict[tuple[str, ...], object] = {}
        self.lock = threading.Lock()

    def detect(self, image_bgr, prompts, imgsz, conf, device):
        key = tuple(prompts)
        with self.lock:  # set_classes mutates the model
            if self._model is None:
                from ultralytics import YOLOE

                self._model = YOLOE(self.weights)
            m = self._model
            if key not in self._pe:
                self._pe[key] = m.get_text_pe(list(prompts))
            m.set_classes(list(prompts), self._pe[key])
            r = m.predict(image_bgr, imgsz=imgsz, conf=conf, max_det=300, device=device, verbose=False)[0]
        b = r.boxes
        h, w = r.orig_shape
        arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1).astype(np.float32)
        return Detections.from_array(arr, w, h)


class OpenVocab:
    def __init__(self, weights: str | Path, imgsz: int = 1280, conf_floor: float = 0.05, device: str = "cpu", threshold: float = 0.3,
                 taught_weights: str | Path | None = None, taught_threshold: float = 0.25, taught_ap: dict[str, float] | None = None):
        self.weights = str(weights)
        self.imgsz = imgsz
        self.conf_floor = conf_floor
        self.device = device
        self.threshold = threshold  # default count threshold for zero-shot boxes (uncalibrated)
        self._zs = _YOLOE(self.weights)
        self.taught_weights = str(taught_weights) if taught_weights else None
        self.taught_threshold = taught_threshold
        self.taught_ap = taught_ap or {}
        self._taught = _YOLOE(self.taught_weights) if self.taught_weights and Path(self.taught_weights).exists() else None
        if self._taught:
            from .ovtrain.vocab import SEEN

            self._lookup = taught_lookup(SEEN)
        else:
            self._lookup = {}

    @property
    def available(self) -> bool:
        return Path(self.weights).exists()

    def taught_word(self, term: str) -> str | None:
        """Canonical training prompt if the drone-tuned model was taught this term, else None."""
        t = term.strip().lower()
        return self._lookup.get(t) or self._lookup.get(singular(t)) or self._lookup.get(" ".join(t.split()[:-1] + [singular(t.split()[-1])]) if t else "")

    def detect(self, image_bgr: np.ndarray, prompts: list[str]) -> Detections:
        """Detections whose cls indexes into ``prompts``."""
        canon = self.taught_word(prompts[0]) if self._taught and len(prompts) == 1 else None
        if canon:
            det = self._taught.detect(image_bgr, [canon], self.imgsz, self.conf_floor, self.device)
            det.meta = {"model": Path(self.taught_weights).stem, "prompts": [canon], "taught": True, "threshold": self.taught_threshold,
                        "ap_vs_teacher": self.taught_ap.get(canon)}
            if not (det.conf >= self.taught_threshold).any():  # tuned model found nothing: last resort, ask zero-shot
                zs = self._zs.detect(image_bgr, prompts, self.imgsz, self.conf_floor, self.device)
                if (zs.conf >= self.threshold).any():
                    zs.meta = {"model": Path(self.weights).stem, "prompts": list(prompts), "fallback_from": canon}
                    return zs
        else:
            det = self._zs.detect(image_bgr, prompts, self.imgsz, self.conf_floor, self.device)
            det.meta = {"model": Path(self.weights).stem, "prompts": list(prompts)}
        return det


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    bb = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1e-9)
