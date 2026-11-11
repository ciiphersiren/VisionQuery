"""Calibration bundle: thresholds and count estimator per pipeline.

Loaded from ``configs/calibration.json``. The shipped file starts from the
report's frozen count thresholds (§6.2) and is extended by
``python -m skyquery.calibrate.run`` with existence thresholds and count
intervals tuned on val (sequence-grouped CV).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classes import CLASSES

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "calibration.json"


@dataclass
class PipelineCalibration:
    count_thr: np.ndarray
    exist_thr: np.ndarray
    global_thr: float
    # per class: residual (gt - pred) quantiles by predicted-count bucket, for answer ranges
    intervals: dict = field(default_factory=dict)
    # per class existence recall / precision measured on held-out data (for honest caveats)
    exist_recall: dict = field(default_factory=dict)
    exist_precision: dict = field(default_factory=dict)
    count_method: str = "threshold"
    # conformal count intervals (sequence-calibrated): per class q for the score |y - n| / sqrt(n + 1)
    conformal_q: dict = field(default_factory=dict)
    interval_label: str = "likely"
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineCalibration":
        # a class the model was not trained on has no threshold: NaN, so no box of it is ever counted
        ct = np.array([d["count_thresholds"].get(c, np.nan) for c in CLASSES], float)
        et = np.array([d.get("exist_thresholds", d["count_thresholds"]).get(c, np.nan) for c in CLASSES], float)
        conf = d.get("conformal", {})
        return cls(
            count_thr=ct, exist_thr=et, global_thr=float(d.get("global_threshold", 0.25)),
            intervals=d.get("intervals", {}), exist_recall=d.get("exist_recall", {}),
            exist_precision=d.get("exist_precision", {}), count_method=d.get("count_method", "threshold"),
            conformal_q=conf.get("q", {}), interval_label=conf.get("label", "likely") if conf else "likely", source=d.get("source", ""),
        )

    def interval(self, cls_name: str, n: int) -> tuple[int, int]:
        """Range for the true count given predicted n: a conformal interval where one was calibrated, else the 80% residual band."""
        q = self.conformal_q.get(cls_name)
        if q is not None:
            r = float(q) * np.sqrt(n + 1)
            return max(0, int(np.ceil(n - r - 1e-9))), int(np.floor(n + r + 1e-9))
        iv = self.intervals.get(cls_name)
        if not iv:
            return n, n
        # bucket containing n, else the highest bucket below n (rare large counts)
        cand = [b for b in iv if b["lo_pred"] <= n <= b["hi_pred"]] or [b for b in iv if b["lo_pred"] <= n and b.get("rel")][-1:]
        if not cand:
            return n, n
        b = cand[0]
        f = n if b.get("rel") else 1
        return max(0, int(np.floor(n + b["q10"] * f + 0.5))), max(0, int(np.floor(n + b["q90"] * f + 0.5)))


def load_calibration(path: str | Path | None = None) -> dict[str, PipelineCalibration]:
    data = json.loads(Path(path or DEFAULT_PATH).read_text())
    return {k: PipelineCalibration.from_dict(v) for k, v in data["pipelines"].items()}
