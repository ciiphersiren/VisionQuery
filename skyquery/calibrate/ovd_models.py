"""Zero-shot, text-prompted detectors behind one interface.

Every model returns an (N, 6) float32 array  x1, y1, x2, y2, score, prompt_index
in original-image pixels, so all of them are scored by the same COCO harness.

    m = load_ovd("yoloe-26s", device="cuda")
    arr = m.predict(image_path, prompts=["car", "bus"])

Ultralytics models (YOLOE-26, YOLO-World) download from GitHub; OWLv2 and
Grounding DINO download from Hugging Face (set HF_TOKEN if your network needs it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

ULTRALYTICS = {
    "yoloe-26s": "yoloe-26s-seg.pt",
    "yoloe-26m": "yoloe-26m-seg.pt",
    "yoloe-26l": "yoloe-26l-seg.pt",
    "yolo-world-v2-s": "yolov8s-worldv2.pt",
    "yolo-world-v2-l": "yolov8l-worldv2.pt",
}
HF = {
    "owlv2-base": "google/owlv2-base-patch16-ensemble",
    "owlv2-large": "google/owlv2-large-patch14-ensemble",
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
}
ALL_MODELS = list(ULTRALYTICS) + list(HF)


@dataclass
class OVDModel:
    name: str
    device: str = "cpu"
    imgsz: int = 1280
    conf: float = 0.001
    max_det: int = 500

    def predict(self, image_path: str, prompts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class UltralyticsOVD(OVDModel):
    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        if name.startswith("yoloe"):
            from ultralytics import YOLOE

            self.m = YOLOE(ULTRALYTICS[name])
        else:
            from ultralytics import YOLOWorld

            self.m = YOLOWorld(ULTRALYTICS[name])
        self._key: tuple | None = None

    def _set(self, prompts):
        key = tuple(prompts)
        if key != self._key:
            if hasattr(self.m, "get_text_pe"):
                self.m.set_classes(list(prompts), self.m.get_text_pe(list(prompts)))
            else:
                self.m.set_classes(list(prompts))
            self._key = key

    def predict(self, image_path, prompts):
        self._set(prompts)
        r = self.m.predict(image_path, imgsz=self.imgsz, conf=self.conf, max_det=self.max_det, device=self.device, verbose=False)[0]
        b = r.boxes
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1).astype(np.float32)


class OwlV2(OVDModel):
    """OWLv2 pads to a square; boxes are post-processed against the padded side."""

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(HF[name])
        self.m = Owlv2ForObjectDetection.from_pretrained(HF[name]).to(self.device).eval()

    def predict(self, image_path, prompts):
        from PIL import Image

        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        side = max(w, h)
        inp = self.proc(text=[list(prompts)], images=im, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.m(**inp)
        res = self.proc.post_process_grounded_object_detection(out, threshold=self.conf, target_sizes=[(side, side)])[0]
        boxes = res["boxes"].cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        arr = np.concatenate([boxes, res["scores"].cpu().numpy()[:, None], res["labels"].cpu().numpy()[:, None]], 1).astype(np.float32)
        return arr[np.argsort(-arr[:, 4])][: self.max_det]


class GroundingDino(OVDModel):
    """Prompts are joined as 'a. b. c.'; matched phrases are mapped back to prompt indices."""

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(HF[name])
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(HF[name]).to(self.device).eval()

    @staticmethod
    def _index(phrase: str, prompts: list[str]) -> int:
        p = phrase.strip().lower()
        low = [q.lower() for q in prompts]
        if p in low:
            return low.index(p)
        # longest prompt contained in the phrase, else best word overlap
        hits = [(len(q), i) for i, q in enumerate(low) if q in p or p in q]
        if hits:
            return max(hits)[1]
        pw = set(re.findall(r"[a-z]+", p))
        scores = [len(pw & set(re.findall(r"[a-z]+", q))) for q in low]
        return int(np.argmax(scores)) if max(scores) > 0 else -1

    def predict(self, image_path, prompts):
        from PIL import Image

        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        text = " ".join(f"{p.lower()}." for p in prompts)
        inp = self.proc(images=im, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.m(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=max(self.conf, 0.05), text_threshold=0.05, target_sizes=[(h, w)]
        )[0]
        labels = res.get("text_labels", res.get("labels"))
        idx = np.array([self._index(str(l), prompts) for l in labels], np.float32)
        arr = np.concatenate([res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()[:, None], idx[:, None]], 1).astype(np.float32)
        arr = arr[arr[:, 5] >= 0]
        return arr[np.argsort(-arr[:, 4])][: self.max_det]


def load_ovd(name: str, **kw) -> OVDModel:
    if name in ULTRALYTICS:
        return UltralyticsOVD(name, **kw)
    if name.startswith("owlv2"):
        return OwlV2(name, **kw)
    if name.startswith("grounding-dino"):
        return GroundingDino(name, **kw)
    raise ValueError(f"unknown model {name!r}; choose from {ALL_MODELS}")
