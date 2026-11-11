"""Detector: the only component that owns geometry.

Pipelines:
  * ``fi1280``         one full-image pass at 1280 px (fast, default for YOLO26s)
  * ``fi1280+fi1920``  two full-image passes fused (slower, ~+2 AP)
  * ``fi1024``         one full-image pass at 1024 px (the UAVDT models' val-chosen setup)
  * ``sliced``         512 px slices enlarged to 1280 + whole image at 1536, boundary filter, class-wise NMS 0.6:
                       the "best sliced" setup of the paper

A model may also be an ensemble: each member runs the pipeline, then weighted box fusion (IoU 0.65) joins them.
Models trained on fewer classes (UAVDT: car, truck, bus) have their class ids mapped onto the VisDrone ids here.

Detections are returned at a low confidence floor; thresholds for counting /
existence are applied later by the answer engine, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .classes import CLASSES

PIPELINES = {
    "fi1280": ("full@1280",),
    "fi1280+fi1920": ("full@1280", "full@1920"),
    "fi1024": ("full@1024",),
    "sliced": ("s512@1280", "full@1536"),
}
SLICED_NMS_IOU = 0.6        # evaluation v2, best sliced setup
SLICE_OVERLAP = 0.25
SLICE_BATCH = 8
EDGE_MARGIN = 2.0           # boundary filter: drop slice boxes within 2 px of an interior slice edge
ENSEMBLE_WBF_IOU = 0.65


@dataclass
class Detections:
    """Boxes in absolute pixel xyxy, plus confidence and class id."""

    xyxy: np.ndarray  # (N, 4) float32
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int64
    width: int
    height: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    @classmethod
    def empty(cls, width: int, height: int) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), width, height)

    def subset(self, mask: np.ndarray) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask], self.width, self.height, dict(self.meta))

    @property
    def centers(self) -> np.ndarray:
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2], axis=1)

    def to_array(self) -> np.ndarray:
        """(N, 6) array: x1, y1, x2, y2, conf, cls."""
        return np.concatenate([self.xyxy, self.conf[:, None], self.cls[:, None].astype(np.float32)], axis=1)

    @classmethod
    def from_array(cls, arr: np.ndarray, width: int, height: int) -> "Detections":
        arr = np.asarray(arr, np.float32).reshape(-1, 6)
        return cls(arr[:, :4].copy(), arr[:, 4].copy(), arr[:, 5].astype(np.int64), width, height)

    def to_json(self, min_conf: float = 0.0) -> list[dict]:
        out = []
        for (x1, y1, x2, y2), c, k in zip(self.xyxy, self.conf, self.cls):
            if c < min_conf:
                continue
            out.append({"box": [round(float(v), 1) for v in (x1, y1, x2, y2)], "conf": round(float(c), 4), "label": CLASSES[int(k)]})
        return out


def _nms_fuse(parts: list[np.ndarray], iou: float, max_det: int) -> np.ndarray:
    import torch
    from torchvision.ops import batched_nms

    arr = np.concatenate(parts, axis=0) if parts else np.zeros((0, 6), np.float32)
    if len(arr) == 0:
        return arr
    t = torch.from_numpy(arr)
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou)[:max_det]
    return arr[keep.numpy()]


def fuse(parts: list[np.ndarray], method: str = "nms", iou: float = 0.6, max_det: int = 500) -> np.ndarray:
    """Fuse per-scale (N, 6) arrays into one set."""
    if method == "nms":
        return _nms_fuse(parts, iou, max_det)
    if method == "concat":
        arr = np.concatenate(parts, axis=0)
        return arr[np.argsort(-arr[:, 4])][:max_det]
    raise ValueError(f"unknown fusion method {method!r}")


def _edge_filter(d: np.ndarray, sb: tuple, W: int, H: int, margin: float = EDGE_MARGIN) -> np.ndarray:
    """Drop slice boxes touching an interior edge of their slice (cut objects); edges on the image border never filter."""
    if len(d) == 0:
        return d
    sx1, sy1, sx2, sy2 = sb
    cut = (((sx1 > 0) & (d[:, 0] <= sx1 + margin)) | ((sy1 > 0) & (d[:, 1] <= sy1 + margin)) |
           ((sx2 < W) & (d[:, 2] >= sx2 - margin)) | ((sy2 < H) & (d[:, 3] >= sy2 - margin)))
    return d[~cut]


def _wbf(parts: list[np.ndarray], W: int, H: int, iou: float, skip: float) -> np.ndarray:
    from ensemble_boxes import weighted_boxes_fusion

    bl = [np.clip(p[:, :4] / [W, H, W, H], 0, 1).tolist() for p in parts]
    b, s, l = weighted_boxes_fusion(bl, [p[:, 4].tolist() for p in parts], [p[:, 5].tolist() for p in parts], iou_thr=iou, skip_box_thr=skip)
    return np.c_[np.asarray(b).reshape(-1, 4) * [W, H, W, H], s, l].astype(np.float32).reshape(-1, 6)


class Detector:
    """Wrapper around one Ultralytics model, or several fused as an ensemble."""

    def __init__(
        self,
        weights: str | Path | list,
        pipeline: str = "fi1280",
        device: str = "cpu",
        conf_floor: float = 0.05,
        max_det: int = 1000,
        fusion: str = "nms",
        fusion_iou: float = 0.6,
        half: bool = False,
    ):
        from ultralytics import YOLO

        if pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {list(PIPELINES)}")
        files = weights if isinstance(weights, (list, tuple)) else [weights]
        self.models = [YOLO(str(w)) for w in files]
        self.model = self.models[0]
        # model class index -> VisDrone class id, by name (UAVDT models know only car, truck, bus)
        self.luts = []
        for m in self.models:
            names = [m.names[i] for i in sorted(m.names)]
            bad = [n for n in names if n not in CLASSES]
            if bad:
                raise ValueError(f"{files}: classes {bad} are not VisDrone classes")
            self.luts.append(np.array([CLASSES.index(n) for n in names], np.float32))
        self.classes = sorted({CLASSES[int(i)] for lut in self.luts for i in lut}, key=CLASSES.index)
        self.pipeline = pipeline
        self.device = device
        self.conf_floor = conf_floor
        self.max_det = max_det
        self.fusion = fusion
        self.fusion_iou = fusion_iou
        self.half = half

    def _predict(self, model, lut, images, imgsz: int) -> list[np.ndarray]:
        kw = {"half": True} if self.half else {}
        out = []
        for r in model.predict(images, imgsz=imgsz, conf=self.conf_floor, max_det=self.max_det, device=self.device, verbose=False, **kw):
            b = r.boxes
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)
            if len(arr):
                arr[:, 5] = lut[arr[:, 5].astype(int)]
            out.append(arr.reshape(-1, 6))
        return out

    def _one_scale(self, image, imgsz: int, model=None, lut=None) -> tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        return self._predict(model or self.model, self.luts[0] if lut is None else lut, image, imgsz)[0], w, h

    def _sliced(self, image, model, lut, tile: int, imgsz: int) -> np.ndarray:
        from sahi.slicing import get_slice_bboxes

        H, W = image.shape[:2]
        boxes = get_slice_bboxes(image_height=H, image_width=W, slice_height=tile, slice_width=tile, auto_slice_resolution=False,
                                 overlap_height_ratio=SLICE_OVERLAP, overlap_width_ratio=SLICE_OVERLAP)
        crops = [np.ascontiguousarray(image[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes]
        parts = []
        for i in range(0, len(crops), SLICE_BATCH):
            for a, (x1, y1, x2, y2) in zip(self._predict(model, lut, crops[i:i + SLICE_BATCH], imgsz), boxes[i:i + SLICE_BATCH]):
                a[:, [0, 2]] += x1
                a[:, [1, 3]] += y1
                parts.append(_edge_filter(a, (x1, y1, x2, y2), W, H))
        return np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)

    def per_scale(self, image, model=None, lut=None) -> tuple[dict[str, np.ndarray], int, int]:
        model = model or self.model
        lut = self.luts[0] if lut is None else lut
        h, w = image.shape[:2]
        out = {}
        for key in PIPELINES[self.pipeline]:
            kind, size = key.split("@")
            out[key] = self._predict(model, lut, image, int(size))[0] if kind == "full" else self._sliced(image, model, lut, int(kind[1:]), int(size))
        return out, w, h

    def _merged(self, image, model, lut) -> tuple[np.ndarray, int, int]:
        parts, w, h = self.per_scale(image, model, lut)
        if len(parts) == 1:
            return next(iter(parts.values())), w, h
        if self.pipeline == "sliced":           # SAHI's own class-wise NMS, as in evaluation v2
            from sahi.postprocess.combine import batched_nms

            P = np.concatenate(list(parts.values()))
            return (P[batched_nms(P, match_metric="IOU", match_threshold=SLICED_NMS_IOU)] if len(P) else P), w, h
        return fuse(list(parts.values()), self.fusion, self.fusion_iou, self.max_det), w, h

    def __call__(self, image) -> Detections:
        import time

        t0 = time.perf_counter()
        runs = [self._merged(image, m, lut) for m, lut in zip(self.models, self.luts)]
        w, h = runs[0][1], runs[0][2]
        arr = runs[0][0] if len(runs) == 1 else _wbf([r[0] for r in runs], w, h, ENSEMBLE_WBF_IOU, self.conf_floor)
        det = Detections.from_array(arr, w, h)
        det.meta = {"pipeline": self.pipeline, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return det

    def batch(self, images: Iterable) -> list[Detections]:
        return [self(im) for im in images]
