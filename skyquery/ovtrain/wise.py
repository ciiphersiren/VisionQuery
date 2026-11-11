"""Weight-space ensembling (WiSE-FT, Wortsman et al., CVPR 2022) of zero-shot and drone-tuned YOLOE.

    theta(alpha) = (1 - alpha) * theta_zeroshot + alpha * theta_tuned

E3 showed the tuned model learns the taught words (tree 1 -> 52 AP) but stops
answering to everything else, even synonyms. Interpolating the two sets of
weights is the cheapest known way to buy back zero-shot behaviour while keeping
part of the fine-tuning gain, and needs no training. Only tensors present in
both checkpoints with the same shape are mixed (the zero-shot file also carries
a segmentation head the detector does not have).

    python -m skyquery.ovtrain.wise --zeroshot yoloe-26s-seg.pt --tuned weights/yoloe-26s-drone-ov.pt \
        --alpha 0.5 --out weights/yoloe-26s-drone-wise50.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def interpolate(zeroshot: str, tuned: str, alpha: float, out: str) -> dict:
    import torch
    from ultralytics import YOLOE
    from ultralytics.utils.patches import torch_load

    z = YOLOE(zeroshot).model.float().state_dict()
    ck = torch_load(tuned, map_location="cpu")
    m = ck["model"].float()
    sd = m.state_dict()
    mixed = 0
    for k, v in sd.items():
        if k in z and z[k].shape == v.shape and v.dtype.is_floating_point:
            sd[k] = (1 - alpha) * z[k] + alpha * v
            mixed += 1
    m.load_state_dict(sd)
    ck["model"] = m.half()
    ck["ema"] = None
    ck["optimizer"] = None
    ck["train_metrics"] = {"wise_alpha": alpha}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return {"alpha": alpha, "mixed": mixed, "total": len(sd)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot", default="yoloe-26s-seg.pt")
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True, help="path; with several alphas, '{a}' is replaced by 100*alpha")
    a = ap.parse_args(argv)
    for al in a.alpha:
        out = a.out.replace("{a}", f"{round(al * 100):02d}")
        print(out, interpolate(a.zeroshot, a.tuned, al, out))


if __name__ == "__main__":
    main()
