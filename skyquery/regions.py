"""Named image regions in normalised [x1, y1, x2, y2] coordinates.

A box belongs to a region when its centre lies inside it (half-open on the
right/bottom edge, closed at the image border), so halves partition the image
and every box is counted exactly once across left + right.
"""

from __future__ import annotations

import re

import numpy as np

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "whole": (0.0, 0.0, 1.0, 1.0),
    "left half": (0.0, 0.0, 0.5, 1.0),
    "right half": (0.5, 0.0, 1.0, 1.0),
    "top half": (0.0, 0.0, 1.0, 0.5),
    "bottom half": (0.0, 0.5, 1.0, 1.0),
    "top-left quadrant": (0.0, 0.0, 0.5, 0.5),
    "top-right quadrant": (0.5, 0.0, 1.0, 0.5),
    "bottom-left quadrant": (0.0, 0.5, 0.5, 1.0),
    "bottom-right quadrant": (0.5, 0.5, 1.0, 1.0),
    "center": (0.25, 0.25, 0.75, 0.75),
    "left third": (0.0, 0.0, 1 / 3, 1.0),
    "middle third": (1 / 3, 0.0, 2 / 3, 1.0),
    "right third": (2 / 3, 0.0, 1.0, 1.0),
    "top third": (0.0, 0.0, 1.0, 1 / 3),
    "bottom third": (0.0, 2 / 3, 1.0, 1.0),
}

_V = r"(?:top|upper|bottom|lower)"
_H = r"(?:left|right)"

# (pattern, region) — checked in order, first match wins.
_PATTERNS: list[tuple[str, str]] = [
    # quadrants / corners: "top left", "upper-left corner", "bottom right quadrant"
    (rf"\b(top|upper)[\s-]*(left)\b", "top-left quadrant"),
    (rf"\b(top|upper)[\s-]*(right)\b", "top-right quadrant"),
    (rf"\b(bottom|lower)[\s-]*(left)\b", "bottom-left quadrant"),
    (rf"\b(bottom|lower)[\s-]*(right)\b", "bottom-right quadrant"),
    (rf"\b(left)[\s-]*(top|upper)\b", "top-left quadrant"),
    (rf"\b(right)[\s-]*(top|upper)\b", "top-right quadrant"),
    (rf"\b(left)[\s-]*(bottom|lower)\b", "bottom-left quadrant"),
    (rf"\b(right)[\s-]*(bottom|lower)\b", "bottom-right quadrant"),
    # thirds
    (r"\bleft(?:most)?\s+third\b", "left third"),
    (r"\bright(?:most)?\s+third\b", "right third"),
    (r"\b(?:middle|central|centre|center)\s+third\b", "middle third"),
    (rf"\b(?:top|upper)\s+third\b", "top third"),
    (rf"\b(?:bottom|lower)\s+third\b", "bottom third"),
    # centre
    (r"\b(?:in the |at the )?(?:center|centre|middle)\b(?! third)", "center"),
    (r"\bcentral\b", "center"),
    # halves
    (rf"\bleft\b", "left half"),
    (rf"\bright\b(?!\s+now)", "right half"),
    (rf"\b(?:top|upper)\b", "top half"),
    (rf"\b(?:bottom|lower)\b", "bottom half"),
]


def parse_region(text: str) -> tuple[str, str | None]:
    """Return (region_name, matched_span). Defaults to 'whole'."""
    t = text.lower()
    for pat, name in _PATTERNS:
        m = re.search(pat, t)
        if m:
            return name, m.group(0)
    return "whole", None


def region_px(name: str, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = REGIONS[name]
    return x1 * width, y1 * height, x2 * width, y2 * height


def in_region(centers: np.ndarray, name: str, width: int, height: int) -> np.ndarray:
    if name == "whole" or len(centers) == 0:
        return np.ones(len(centers), bool)
    x1, y1, x2, y2 = region_px(name, width, height)
    cx, cy = centers[:, 0], centers[:, 1]
    right_ok = (cx < x2) | (x2 >= width)
    bottom_ok = (cy < y2) | (y2 >= height)
    return (cx >= x1) & right_ok & (cy >= y1) & bottom_ok
