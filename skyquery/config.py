"""Runtime configuration (YAML + env overrides: SKYQUERY_<FIELD>)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    weights: str = str(ROOT / "weights" / "yolo26s_visdrone_sahi.pt")
    calibration: str = str(ROOT / "configs" / "calibration.json")
    pipeline: str = "fi1280"
    device: str = "cpu"
    half: bool = False
    conf_floor: float = 0.05
    max_det: int = 1000         # dense VisDrone frames hold up to 902 objects
    fusion: str = "nms"
    fusion_iou: float = 0.5
    planner_policy: str = "rules_first"
    ovd_enabled: bool = True
    ovd_weights: str = "yoloe-26s-seg.pt"  # auto-downloaded by Ultralytics on first use
    ovd_threshold: float = 0.1
    ovd_subtype_threshold: float = 0.05
    ovd_taught_weights: str = ""      # drone-tuned YOLOE for the words it was taught (configs/ovd_drone.json)
    ovd_taught_threshold: float = 0.25
    cache_images: int = 32
    models: str = str(ROOT / "configs" / "models.json")   # registry of switchable detectors; missing file = `weights` only
    model: str = ""                                       # default model id; empty = the registry's default
    max_upload_mb: int = 25


DRONE_OV = ROOT / "configs" / "ovd_drone.json"


def _drone_ov(cfg: Config) -> None:
    """Use the drone-tuned YOLOE (Kaggle notebook E3) when its weights and threshold file are present."""
    import json

    if not DRONE_OV.exists():
        return
    d = json.loads(DRONE_OV.read_text())
    w = Path(d.get("weights", ""))
    w = w if w.is_absolute() else ROOT / w
    if not w.exists():
        return
    if d.get("adopt", True):  # replaces zero-shot for every word (pre-registered rule passed)
        cfg.ovd_weights = str(w)
        cfg.ovd_threshold = float(d.get("threshold", cfg.ovd_threshold))
    elif d.get("route_taught"):  # only for the words it was taught; zero-shot keeps everything else
        cfg.ovd_taught_weights = str(w)
        cfg.ovd_taught_threshold = float(d.get("threshold", cfg.ovd_taught_threshold))


def load_config(path: str | Path | None = None, **overrides) -> Config:
    import yaml

    cfg = Config()
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    _drone_ov(cfg)
    for f in fields(Config):
        env = os.environ.get(f"SKYQUERY_{f.name.upper()}")
        if env is not None:
            setattr(cfg, f.name, type(getattr(cfg, f.name))(env) if not isinstance(getattr(cfg, f.name), bool) else env.lower() in {"1", "true", "yes"})
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    for k in ("weights", "calibration", "ovd_weights", "models"):
        v = Path(getattr(cfg, k))
        if not v.is_absolute() and (k != "ovd_weights" or (ROOT / v).exists() or "/" in str(v)):
            setattr(cfg, k, str(ROOT / v))
    return cfg
