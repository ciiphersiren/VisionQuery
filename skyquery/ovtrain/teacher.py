"""The teacher: a big, slow open-vocabulary detector that labels drone frames.

SAM 3 (Meta, 2025; facebook/sam3 on Hugging Face, gated: accept the licence
first) is the default. It is the strongest openly released text-prompted
detector we could find (zero-shot LVIS mask AP 47.0), but it is ~850M
parameters and nowhere near real time, so it labels overnight and YOLOE learns
from it. OWLv2-large is the fallback when SAM 3 cannot be downloaded.

Small objects are SAM 3's weak spot, so every frame is seen as several views:
the full frame plus an overlapping grid of crops, each resized to the model's
1008 px input. Boxes are mapped back and merged with class-wise NMS; a box
that touches an inner crop edge is dropped (it is a cut-off piece, and the
neighbouring crop or the full frame sees the whole object).

Speed trick: the image encoder runs once per view and its features are shared
by all prompts, and the mask decoder is skipped (we only need boxes).

    t = load_teacher("sam3", device="cuda")
    arr = t.label(pil_image, prompts)   # (N, 6): x1 y1 x2 y2 score prompt_index
"""

from __future__ import annotations

import numpy as np

from ..calibrate.tiles import tile_origins


def nms_classwise(arr: np.ndarray, iou: float = 0.6) -> np.ndarray:
    if len(arr) == 0:
        return arr
    import torch
    from torchvision.ops import batched_nms

    keep = batched_nms(torch.from_numpy(arr[:, :4]).float(), torch.from_numpy(arr[:, 4]).float(), torch.from_numpy(arr[:, 5]).long(), iou)
    return arr[keep.numpy()]


def grid_views(w: int, h: int, grid: int = 2, overlap: float = 0.15, full: bool = True) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) views: the full frame plus a grid x grid set of overlapping crops."""
    views = [(0, 0, w, h)] if full else []
    if grid > 1:
        tw = int(np.ceil(w / (grid - (grid - 1) * overlap)))
        th = int(np.ceil(h / (grid - (grid - 1) * overlap)))
        for y0 in tile_origins(h, th, overlap):
            for x0 in tile_origins(w, tw, overlap):
                views.append((x0, y0, min(x0 + tw, w), min(y0 + th, h)))
    return views


def drop_cut(boxes: np.ndarray, view: tuple[int, int, int, int], w: int, h: int, tol: float = 3.0) -> np.ndarray:
    """Mask of boxes that do NOT touch an inner edge of the view (edges shared with the frame are fine)."""
    x0, y0, x1, y1 = view
    bad = np.zeros(len(boxes), bool)
    if x0 > 0:
        bad |= boxes[:, 0] <= x0 + tol
    if y0 > 0:
        bad |= boxes[:, 1] <= y0 + tol
    if x1 < w:
        bad |= boxes[:, 2] >= x1 - tol
    if y1 < h:
        bad |= boxes[:, 3] >= y1 - tol
    return ~bad


class Teacher:
    name = "teacher"
    floor = 0.25  # keep everything above this; training/eval thresholds are applied later

    def __init__(self, grid: int = 2, overlap: float = 0.15, max_area: float = 0.5, min_side: float = 4.0, nms_iou: float = 0.6):
        self.grid, self.overlap = grid, overlap
        self.max_area, self.min_side, self.nms_iou = max_area, min_side, nms_iou

    # subclasses: boxes in view-normalised xyxy [0,1], scores, prompt index
    def detect_views(self, crops: list, prompts: list[str]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def label(self, image, prompts: list[str]) -> np.ndarray:
        """All views of one PIL image -> merged (N, 6) float32 array in image pixels."""
        w, h = image.size
        views = grid_views(w, h, self.grid, self.overlap)
        crops = [image.crop(v) for v in views]
        outs = self.detect_views(crops, prompts)
        parts = []
        for v, (b, s, c) in zip(views, outs):
            if len(b) == 0:
                continue
            vw, vh = v[2] - v[0], v[3] - v[1]
            px = b * np.array([vw, vh, vw, vh], np.float32) + np.array([v[0], v[1], v[0], v[1]], np.float32)
            px[:, [0, 2]] = px[:, [0, 2]].clip(0, w)
            px[:, [1, 3]] = px[:, [1, 3]].clip(0, h)
            ok = drop_cut(px, v, w, h) if v != (0, 0, w, h) else np.ones(len(px), bool)
            parts.append(np.concatenate([px, s[:, None], c[:, None]], 1)[ok])
        if not parts:
            return np.zeros((0, 6), np.float32)
        arr = np.concatenate(parts).astype(np.float32)
        bw, bh = arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1]
        arr = arr[(bw >= self.min_side) & (bh >= self.min_side) & (bw * bh <= self.max_area * w * h) & (arr[:, 4] >= self.floor)]
        arr = nms_classwise(arr, self.nms_iou)
        return arr[np.argsort(-arr[:, 4])]


class Sam3Teacher(Teacher):
    name = "sam3"

    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda", fp16: bool = True, prompt_chunk: int = 8, view_batch: int = 1,
                 model=None, processor=None, **kw):
        super().__init__(**kw)
        import torch

        self.torch = torch
        self.device = device
        if model is None:
            from transformers import Sam3Model, Sam3Processor

            processor = Sam3Processor.from_pretrained(model_id)
            model = Sam3Model.from_pretrained(model_id)
        self.proc = processor
        self.m = model.to(device).eval()
        self.fp16 = fp16 and str(device).startswith("cuda")
        self.prompt_chunk, self.view_batch = prompt_chunk, view_batch
        self._bank: dict[tuple[str, ...], tuple] = {}

    def _amp(self):
        t = self.torch
        return t.autocast("cuda", dtype=t.float16) if self.fp16 else t.autocast("cpu", enabled=False)

    def tokenize(self, prompts: list[str]):
        tok = self.proc.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=32)
        return tok["input_ids"], tok["attention_mask"]

    def text_bank(self, prompts: list[str]):
        key = tuple(prompts)
        if key not in self._bank:
            ids, am = self.tokenize(prompts)
            with self.torch.no_grad(), self._amp():
                feats = self.m.get_text_features(input_ids=ids.to(self.device), attention_mask=am.to(self.device)).pooler_output
            self._bank[key] = (feats, am.to(self.device).bool())
        return self._bank[key]

    def _boxes_scores(self, vis, feats, mask):
        """Detector head for one view and P prompts, without the mask decoder. Mirrors Sam3Model.forward."""
        from transformers.models.sam3.modeling_sam3 import box_cxcywh_to_xyxy, inverse_sigmoid

        m, P = self.m, feats.shape[0]
        f = vis.fpn_hidden_states[:-1][-1]
        p = vis.fpn_position_encoding[:-1][-1]
        f, p = f.expand(P, *f.shape[1:]), p.expand(P, *p.shape[1:])
        enc = m.detr_encoder(vision_features=[f], text_features=feats, vision_pos_embeds=[p], text_mask=mask)
        dec = m.detr_decoder(vision_features=enc.last_hidden_state, text_features=enc.text_features, vision_pos_encoding=enc.pos_embeds_flattened,
                             text_mask=mask, spatial_shapes=enc.spatial_shapes)
        hs = dec.intermediate_hidden_states[-1:]
        boxes = box_cxcywh_to_xyxy((inverse_sigmoid(dec.reference_boxes[-1:]) + m.detr_decoder.box_head(hs)).sigmoid())[-1]
        logits = m.dot_product_scoring(decoder_hidden_states=hs, text_features=enc.text_features, text_mask=mask).squeeze(-1)[-1]
        scores = logits.sigmoid() * dec.presence_logits[-1].sigmoid()
        return boxes.float(), scores.float()

    def detect_views(self, crops, prompts):
        t = self.torch
        feats, mask = self.text_bank(prompts)
        out = []
        for i in range(0, len(crops), self.view_batch):
            px = self.proc.image_processor(crops[i:i + self.view_batch], return_tensors="pt")["pixel_values"].to(self.device)
            with t.no_grad(), self._amp():
                vis_all = self.m.get_vision_features(pixel_values=px)
                for j in range(px.shape[0]):
                    vis = type(vis_all)(fpn_hidden_states=[h[j:j + 1] for h in vis_all.fpn_hidden_states],
                                        fpn_position_encoding=[q[j:j + 1] for q in vis_all.fpn_position_encoding])
                    B, S, C = [], [], []
                    k = 0
                    while k < len(prompts):
                        n = self.prompt_chunk
                        try:
                            b, s = self._boxes_scores(vis, feats[k:k + n], mask[k:k + n])
                        except t.cuda.OutOfMemoryError:
                            if n == 1:
                                raise
                            self.prompt_chunk = max(1, n // 2)  # remember the smaller chunk for later frames
                            t.cuda.empty_cache()
                            continue
                        keep = s >= self.floor
                        B.append(b[keep])
                        S.append(s[keep])
                        C.append(keep.nonzero()[:, 0] + k)
                        k += n
                    out.append((t.cat(B).cpu().numpy(), t.cat(S).cpu().numpy(), t.cat(C).cpu().numpy().astype(np.float32)))
        return out


class Owlv2Teacher(Teacher):
    """Fallback. OWLv2 pads each crop to a square, so boxes come back normalised to the padded side."""

    name = "owlv2"
    floor = 0.15

    def __init__(self, model_id: str = "google/owlv2-large-patch14-ensemble", device: str = "cuda", fp16: bool = True, **kw):
        super().__init__(**kw)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch, self.device = torch, device
        self.proc = Owlv2Processor.from_pretrained(model_id)
        self.m = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.fp16 = fp16 and str(device).startswith("cuda")

    def detect_views(self, crops, prompts):
        t = self.torch
        out = []
        for im in crops:
            w, h = im.size
            side = max(w, h)
            inp = self.proc(text=[list(prompts)], images=im, return_tensors="pt").to(self.device)
            with t.no_grad(), t.autocast("cuda", dtype=t.float16, enabled=self.fp16):
                o = self.m(**inp)
            r = self.proc.post_process_grounded_object_detection(o, threshold=self.floor, target_sizes=[(side, side)])[0]
            b = r["boxes"].float().cpu().numpy() / np.array([w, h, w, h], np.float32)
            out.append((b.clip(0, 1), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy().astype(np.float32)))
        return out


def load_teacher(name: str = "sam3", device: str = "cuda", **kw) -> Teacher:
    if name == "sam3":
        return Sam3Teacher(device=device, **kw)
    if name.startswith("owlv2"):
        return Owlv2Teacher(device=device, **kw)
    raise ValueError(name)
