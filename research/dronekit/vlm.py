"""Prompts and strict parsing for the VLM counting systems. Model-free, so all of it is unit tested on CPU.

B direct: integer answer. C localize-then-count: one point per object, count = valid distinct points.
D grounded: the detector's boxes drawn on the image and listed; the VLM gives the final count. Unparsable answers count as wrong.
"""
from __future__ import annotations

import re

NOUNS = {"pedestrian": "pedestrians (people standing or walking)", "people": "people who are sitting or riding, not standing or walking",
         "bicycle": "bicycles", "car": "cars", "van": "vans", "truck": "trucks", "tricycle": "tricycles", "awning-tricycle": "tricycles with an awning",
         "bus": "buses", "motor": "motorcycles"}
MAX_POINT = 1000
MAX_LISTED = 150      # boxes listed in the prompt, highest scores first; all are drawn on the image


def prompt_direct(target):
    return f"Count the {NOUNS[target]} in this aerial image. Answer with one integer and nothing else."


def prompt_localize(target):
    return (f"List every one of the {NOUNS[target]} in this aerial image. Write one line per object as x,y: the center of the object, where x and y are "
            f"integers from 0 to {MAX_POINT} relative to the image width and height. Write only those lines. If there are none, write NONE.")


def prompt_grounded(target, boxes):
    lines = "\n".join(f"{i + 1}: {x1},{y1},{x2},{y2} score {s:.2f}" for i, (x1, y1, x2, y2, s) in enumerate(boxes)) or "(none)"
    return (f"An object detector marked {len(boxes)} candidate {NOUNS[target]} in this aerial image. They are drawn as numbered green boxes and listed below "
            f"as number: x1,y1,x2,y2 in pixels. The detector may have false positives and misses. Decide the true number of {NOUNS[target]}. "
            f"Answer with one integer and nothing else.\n{lines}")


def parse_int(text):
    """A bare integer, optionally with a final period. Anything else is unparsable (None)."""
    m = re.fullmatch(r"\s*(\d{1,6})\s*\.?\s*", text or "")
    return int(m[1]) if m else None


def parse_points(text):
    """-> count of distinct valid points, or None when the answer has no valid line and is not NONE.
    A final partial line (cut by the token limit) is ignored because it fails the line pattern."""
    t = (text or "").strip()
    if re.fullmatch(r"(?i)none\.?", t):
        return 0
    pts = set()
    for line in t.splitlines():
        m = re.fullmatch(r"\s*\(?\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)?\s*", line)
        if m and int(m[1]) <= MAX_POINT and int(m[2]) <= MAX_POINT:
            pts.add((int(m[1]), int(m[2])))
    return len(pts) if pts else None


def parse_points_lenient(text):
    """Exploratory, added after seeing the first run (10 Nov 2026): Qwen2.5-VL mostly answered the localize prompt with boxes in pixels,
    "x1,y1,x2,y2 label", its own grounding format, instead of the requested points. This reading also accepts such box lines, optionally in
    a code fence or with a label, and counts distinct objects. Same rules otherwise: a final partial line fails the pattern, NONE is 0."""
    t = re.sub(r"```\w*", "", text or "").strip()
    if re.fullmatch(r"(?i)none\.?", t):
        return 0
    objs = set()
    for line in t.splitlines():
        m = re.fullmatch(r"\s*\(?\s*(\d{1,5})\s*,\s*(\d{1,5})\s*(?:,\s*(\d{1,5})\s*,\s*(\d{1,5})\s*)?\)?\s*(?::?\s*[A-Za-z][A-Za-z \-]*)?\s*", line)
        if m:
            objs.add(tuple(int(g) for g in m.groups() if g is not None))
    return len(objs) if objs else None
