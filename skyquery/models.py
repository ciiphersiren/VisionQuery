"""Model registry: which detectors the app can switch between (``configs/models.json``).

Each model names its weights (one file, or several for an ensemble), the VisDrone classes it was trained on, the
pipelines it was evaluated with and the calibration file that holds its thresholds. A model whose weights are not on
disk is still listed, as unavailable, so the UI can say which file is missing instead of hiding the option.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .classes import CLASSES


@dataclass
class ModelSpec:
    id: str
    name: str
    weights: list[str]
    classes: list[str]
    pipelines: list[str]
    calibration: str
    trained_on: str = "VisDrone"
    notes: str = ""
    max_det: int = 1000

    @property
    def default_pipeline(self) -> str:
        return self.pipelines[0]

    @property
    def missing(self) -> list[str]:
        return [Path(w).name for w in self.weights if not Path(w).exists()]

    @property
    def available(self) -> bool:
        return not self.missing

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "available": self.available, "missing": self.missing, "classes": self.classes,
                "pipelines": self.pipelines, "default_pipeline": self.default_pipeline, "trained_on": self.trained_on, "notes": self.notes}


def load_models(path: str | Path | None, root: Path, fallback_weights: str, fallback_calibration: str) -> tuple[dict[str, ModelSpec], str]:
    """-> ({id: spec}, default id). Without a registry file, the single configured model is the only entry."""
    from .detector import PIPELINES

    p = Path(path) if path else None
    if p is None or not p.exists():
        spec = ModelSpec("default", "YOLO26s (VisDrone)", [fallback_weights], list(CLASSES), ["fi1280", "fi1280+fi1920"], fallback_calibration, max_det=500)
        return {"default": spec}, "default"
    data = json.loads(p.read_text())
    out = {}
    for mid, m in data["models"].items():
        abs_ = lambda s: str(Path(s) if Path(s).is_absolute() else root / s)  # noqa: E731
        w = m["weights"] if isinstance(m["weights"], list) else [m["weights"]]
        spec = ModelSpec(mid, m["name"], [abs_(x) for x in w], m.get("classes", list(CLASSES)), m["pipelines"], abs_(m["calibration"]),
                         m.get("trained_on", "VisDrone"), m.get("notes", ""), int(m.get("max_det", 1000)))
        bad = [c for c in spec.classes if c not in CLASSES]
        assert not bad, f"{mid}: classes {bad} are not VisDrone classes"
        assert all(pp in PIPELINES for pp in spec.pipelines), f"{mid}: unknown pipeline in {spec.pipelines}"
        out[mid] = spec
    return out, data.get("default", next(iter(out)))
