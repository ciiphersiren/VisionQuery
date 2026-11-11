"""VisDrone2019-DET class vocabulary.

Detector ids are 0..9. VisDrone annotation category ids are 1..10, with 0 =
ignored region and 11 = "others" (neither is a detector class).
"""

CLASSES: tuple[str, ...] = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
NUM_CLASSES = len(CLASSES)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Friendly emoji per class, used by the API / UI. Purely cosmetic.
CLASS_EMOJI = {
    "pedestrian": "🚶",
    "people": "🧍",
    "bicycle": "🚲",
    "car": "🚗",
    "van": "🚐",
    "truck": "🚚",
    "tricycle": "🛺",
    "awning-tricycle": "🛺",
    "bus": "🚌",
    "motor": "🏍️",
}

VISDRONE_IGNORE_CAT = 0
VISDRONE_OTHERS_CAT = 11


def visdrone_cat_to_id(cat: int) -> int | None:
    """Map a VisDrone annotation category (1..10) to a detector id (0..9)."""
    if 1 <= cat <= 10:
        return cat - 1
    return None
