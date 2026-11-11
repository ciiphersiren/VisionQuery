"""Fine-tune YOLOE-26 on drone frames *without* losing its open vocabulary.

Ultralytics' usual YOLOE fine-tune (YOLOEPETrainer) bakes the class names into
the head, after which ``set_classes()`` can no longer change what it detects.
We use the text-prompt trainer instead (YOLOETrainerFromScratch via
.trainer.DroneOVTrainer, started from the pretrained weights): every image is trained against text embeddings
of its positive classes plus sampled negatives, and the MobileCLIP text
encoder stays frozen, so the head keeps reading prompts and any new word still
works after training.

E3 found that this alone is not enough: the tuned model stops answering to
untaught words. The biggest single cause was the contrastive head's BatchNorm
statistics and the text adapter (RepRTA) drifting, so ``--freeze-align`` (E3b)
keeps both at their zero-shot values.

Training is time-boxed (Ultralytics ``time=``, hours): the learning-rate
schedule is fitted to however many epochs fit. If two-GPU DDP fails it falls
back to one GPU for the time that is left, starting from the last checkpoint
if there is one.

    python -m skyquery.ovtrain.train --data /tmp/e3/data.json --size s --hours 6 --device 0,1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

BATCH = {"s": 32, "m": 24, "l": 16}


FREEZE_ALIGN = ["23.reprta", "23.cv4", "23.one2one_cv4"]  # E3b: text adapter + contrastive heads (Ultralytics also freezes their BatchNorm stats)


def train_once(data: dict, start: str, size: str, hours: float, device: str, batch: int, project: str, name: str, workers: int, epochs: int,
               imgsz: int = 640, freeze: list | None = None):
    from ultralytics import YOLOE

    from .trainer import DroneOVTrainer

    model = YOLOE(f"yoloe-26{size}.yaml").load(start)
    import os
    if os.environ.get("SKYQ_DEBUG_ITERS"):
        model.add_callback("on_train_batch_end", lambda tr: print("ITER", {k: round(float(v), 3) for k, v in tr.loss_items.items()}, flush=True))
    model.train(
        data=data, trainer=DroneOVTrainer, epochs=epochs, time=hours if hours > 0 else None, imgsz=imgsz, batch=batch, device=device,
        optimizer="AdamW", lr0=2e-4, lrf=0.1, weight_decay=0.025, warmup_epochs=1, warmup_bias_lr=0.0, cos_lr=True, close_mosaic=2,
        project=project, name=name, exist_ok=True, workers=workers, seed=42, amp=True, max_det=500, plots=False, patience=100,
        val=True, freeze=freeze or None,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.json written by skyquery.ovtrain.build")
    ap.add_argument("--size", default="s", choices=list(BATCH))
    ap.add_argument("--hours", type=float, default=6.0, help="training budget; 0 = use --epochs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="0,1")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="e3_ov")
    ap.add_argument("--out", default="", help="copy best.pt here when done")
    ap.add_argument("--freeze-align", action="store_true", help="E3b: keep RepRTA and the contrastive heads (incl. BatchNorm stats) at their zero-shot values")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.data).read_text())
    run = Path(a.project) / a.name
    batch = a.batch or BATCH[a.size]
    t0 = time.time()
    freeze = FREEZE_ALIGN if a.freeze_align else None
    start = f"yoloe-26{a.size}-seg.pt"
    try:
        train_once(data, start, a.size, a.hours, a.device, batch, a.project, a.name, a.workers, a.epochs, a.imgsz, freeze)
    except Exception as e:
        if "," not in a.device:
            raise
        used = (time.time() - t0) / 3600
        left = a.hours - used if a.hours > 0 else 0
        last = run / "weights" / "last.pt"
        if last.exists():
            keep = run.parent / f"{a.name}_ddp_last.pt"
            shutil.copy(last, keep)
            start = str(keep)
        print(f"[train] multi-GPU run failed after {used:.2f} h ({type(e).__name__}: {e}); "
              f"continuing on one GPU for {left:.2f} h from {start}", flush=True)
        if a.hours > 0 and left < 0.25:
            raise
        train_once(data, start, a.size, left, a.device.split(",")[0], max(8, batch // 2), a.project, a.name, a.workers, a.epochs, a.imgsz, freeze)

    best = run / "weights" / "best.pt"
    if not best.exists():
        best = run / "weights" / "last.pt"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, a.out)
        print(f"[train] {best} -> {a.out}", flush=True)
    print(f"[train] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    sys.exit(main())
