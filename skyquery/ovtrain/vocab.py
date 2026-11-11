"""Vocabulary for drone-domain open-vocabulary training (experiment E3).

Three groups of concepts:

* ``VISDRONE``: the 10 annotated classes, written as plain-English prompts
  (E1 showed descriptions beat VisDrone's jargon: 11.77 vs 9.52 AP zero-shot).
  Labels come from the human ground truth.
* ``SEEN``: things a drone sees that VisDrone never labelled. Labels come from
  the teacher (SAM 3). The student trains on them.
* ``HELDOUT``: also teacher-labelled, but only on *val*, never in training,
  not even as negatives. They test whether the student still generalises to
  words it was not taught, i.e. whether open vocabulary survived fine-tuning.

A "/" separates synonyms. Training samples one at random per image (Ultralytics
``RandomLoadText``); the teacher, evaluation and the app use the first.

Nothing here overlaps a VisDrone class (no "taxi", "cyclist", "fire truck"):
two classes on one box would fight in the label assigner.
"""

from __future__ import annotations

VISDRONE: list[str] = [
    "person walking or standing/pedestrian",
    "person sitting or riding",
    "bicycle",
    "car",
    "van",
    "truck",
    "three-wheeled tricycle",
    "tricycle with a canopy",
    "bus",
    "motorcycle/motorbike/scooter",
]

SEEN: list[str] = [
    "tree",
    "building",
    "bush/shrub",
    "traffic light",
    "street lamp/street light/lamp post",
    "utility pole/telephone pole",
    "traffic sign/road sign",
    "road barrier/traffic barrier",
    "crosswalk/zebra crossing",
    "billboard",
    "tent",
    "market stall/street stall",
    "construction crane",
    "excavator",
    "scaffolding",
    "shipping container",
    "manhole cover",
    "solar panel",
    "swimming pool",
    "basketball court",
    "football field/soccer field",
    "gate",
    "bridge",
    "table",
    "handcart/cart",
    "baby stroller",
    "dog",
    "cow",
    "horse",
    "bird",
    "flag",
    "statue",
    "fountain",
    "stairs/staircase",
    "water tank",
]

HELDOUT: list[str] = [
    "umbrella",
    "trash can",
    "bench",
    "potted plant",
    "traffic cone",
    "chair",
    "boat",
    "fire hydrant",
]


def first(name: str) -> str:
    """The canonical prompt: the first synonym."""
    return name.split("/")[0].strip()


def teacher_prompts(names: list[str]) -> list[str]:
    return [first(n) for n in names]


def check() -> None:
    allnames = [first(n) for n in VISDRONE + SEEN + HELDOUT]
    assert len(allnames) == len(set(allnames)), "duplicate concept"
    syn = [s.strip() for n in VISDRONE + SEEN + HELDOUT for s in n.split("/")]
    assert len(syn) == len(set(syn)), "a synonym appears twice"
    assert not set(first(n) for n in HELDOUT) & set(s.strip() for n in SEEN for s in n.split("/")), "held-out word leaks into training"


check()
