"""Deterministic rule planner: query text -> Plan.

Design notes
* Every rule maps to closed-set classes only. Anything the detector cannot
  see (a sub-type like "fire truck", a colour, an animal) is flagged as
  out-of-vocabulary (OOV) instead of being silently mapped to a parent class.
* "people" in everyday English means humans, so it maps to both VisDrone
  person classes. VisDrone's own split is reachable with qualifiers:
  walking/standing -> pedestrian, sitting/riding -> people.
"""

from __future__ import annotations

import re

from ..classes import CLASSES
from ..regions import parse_region
from .schema import OOVTerm, Plan, Target

HUMANS = ["pedestrian", "people"]
VEHICLES = ["car", "van", "truck", "bus", "motor", "tricycle", "awning-tricycle"]
FOUR_WHEELERS = ["car", "van", "truck", "bus"]
TWO_WHEELERS = ["bicycle", "motor"]
THREE_WHEELERS = ["tricycle", "awning-tricycle"]

# (regex, label, classes). Longest / most specific first.
LEXICON: list[tuple[str, str, list[str]]] = [
    # --- awning-tricycle before tricycle
    (r"awning[\s-]*tricycles?|tricycles? with (?:an )?awnings?|covered tricycles?|tricycles? with (?:a )?(?:roof|canopy|canopies)", "awning-tricycles", ["awning-tricycle"]),
    (r"tricycles?(?: without (?:an )?awnings?)?|trikes?", "tricycles", ["tricycle"]),
    (r"three[\s-]*wheelers?|tuk[\s-]*tuks?|auto[\s-]*rickshaws?|rickshaws?", "three-wheelers", THREE_WHEELERS),
    # --- humans: qualified forms first
    (r"(?:people|persons?|humans?|anyone|someone|folks|men|women|kids|children) (?:who (?:are|is) )?(?:sitting|seated|riding|lying|squatting|crouching)(?: (?:on )?(?:a |the )?(?:bikes?|bicycles?|motor[\\s-]*bikes?|motor[\\s-]*cycles?|scooters?|mopeds?|tricycles?))?|(?:sitting|seated|riding|lying|squatting|crouching) (?:people|persons?|humans?)|anyone (?:sitting|seated|riding)|riders?|cyclists?|motorcyclists?|bikers?", "people (sitting/riding)", ["people"]),
    (r"(?:people|persons?|humans?|anyone|someone|folks|men|women|kids|children) (?:who (?:are|is) )?(?:walking|standing)|(?:walking|standing) (?:people|persons?|humans?)|pedestrians?|walkers?", "pedestrians", ["pedestrian"]),
    (r"people|persons?|humans?|human beings|individuals|folks|anyone|anybody|someone|somebody|men|women|kids|children|crowds?", "people", HUMANS),
    # --- two-wheelers
    (r"motor[\s-]*bikes?|motor[\s-]*cycles?|scooters?|mopeds?|motors?(?![\s-]*vehicles?)", "motorbikes", ["motor"]),
    (r"two[\s-]*wheelers?", "two-wheelers", TWO_WHEELERS),
    (r"bicycles?|bikes?|push[\s-]*bikes?", "bicycles", ["bicycle"]),
    # --- vehicles
    (r"four[\s-]*wheelers?", "four-wheelers", FOUR_WHEELERS),
    (r"mini[\s-]*vans?|vans?|mini[\s-]*bus(?:es)?", "vans", ["van"]),
    (r"buses|busses|bus|coach(?:es)?", "buses", ["bus"]),
    (r"trucks?|lorr(?:y|ies)|pick[\s-]*ups?", "trucks", ["truck"]),
    (r"cars?|sedans?|suvs?|automobiles?|hatchbacks?", "cars", ["car"]),
    (r"(?:motor[\s-]*)?vehicles?|traffic|automobiles", "vehicles", VEHICLES),
    # --- everything
    (r"objects?|things|everything|anything|items?", "objects", list(CLASSES)),
]
for _ in LEXICON:  # validate at import
    assert all(c in CLASSES for c in _[2]), _

_LEX = [(re.compile(rf"\b(?:{p})\b"), lab, cls) for p, lab, cls in LEXICON]

# Modifiers that turn a closed-set noun into something the detector cannot verify.
SUBTYPE_MODIFIERS = (
    r"fire|police|school|garbage|delivery|cement|tanker|dump|tow|tour|double[\s-]*decker|army|military|"
    r"electric|sports?|race|racing|vintage|luxury|taxi|cab|ice[\s-]*cream|food|mail|postal|cargo|passenger|"
    r"red|blue|green|yellow|white|black|silver|grey|gray|orange|pink|purple|brown|"
    r"parked|moving|stopped|speeding|turning|broken[\s-]*down|"
    r"big|large|small|tiny|huge"
)
_SUBTYPE_RE = re.compile(rf"\b((?:(?:{SUBTYPE_MODIFIERS})[\s-]+)+)$")

# Out-of-vocabulary nouns with an optional nearest closed-set parent.
OOV_NOUNS: dict[str, list[str]] = {
    "ambulance": ["van", "truck"], "taxi": ["car"], "cab": ["car"], "police car": ["car"], "jeep": ["car"],
    "limousine": ["car"], "tractor": ["truck"], "trailer": ["truck"], "fire engine": ["truck"],
    "tram": ["bus"], "trolleybus": ["bus"], "train": [], "boat": [], "ship": [], "plane": [], "airplane": [],
    "helicopter": [], "drone": [], "dog": [], "cat": [], "cow": [], "horse": [], "animal": [], "bird": [],
    "tree": [], "building": [], "house": [], "road": [], "lane": [], "crosswalk": [], "zebra crossing": [],
    "traffic light": [], "streetlight": [], "pole": [], "sign": [], "umbrella": [], "tent": [], "stall": [],
    "cart": ["tricycle"], "wheelchair": [], "stroller": [], "skateboard": [], "bench": [], "pool": [],
}

_CUES = (
    r"how many|number of|count(?:\s+(?:the|all|up))?|tally(?:\s+the)?|total(?:\s+number of)?|"
    r"are there(?:\s+any)?|is there(?:\s+an?|\s+any)?|any|do you see(?:\s+any)?|can you see(?:\s+any)?|"
    r"show(?:\s+me)?(?:\s+the|\s+all)?|find(?:\s+the|\s+all|\s+any)?|where (?:are|is)(?:\s+the)?|"
    r"locate(?:\s+the|\s+all)?|detect(?:\s+the|\s+all|\s+any)?|highlight(?:\s+the|\s+all)?|mark(?:\s+the|\s+all)?|"
    r"point out(?:\s+the|\s+all)?|does (?:the|this) (?:image|photo|picture|scene) (?:contain|have|show)(?:\s+any|\s+an?)?"
)
_PHRASE_RE = re.compile(
    rf"(?:{_CUES})\s+(?P<phrase>.+?)(?=\s+(?:in|on|at|near|within|across|around|along|inside|visible|are there|is there|present|here|there|do you)\b|[?.!,]|$)"
)
_STOP = {"in", "on", "at", "near", "within", "inside", "half", "side", "corner", "quadrant", "part", "area", "region", "a", "an", "the", "any", "all", "some", "of", "there", "are", "is", "visible", "you", "can", "see", "me", "i", "image", "photo", "picture", "scene", "this", "that"}


def _norm(q: str) -> str:
    q = q.lower().strip()
    q = q.replace("’", "'").replace("centre", "center")
    q = re.sub(r"[“”\"]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def _intent(q: str) -> str | None:
    if re.search(r"\b(more|fewer|less|greater)\b.*\bthan\b|\bcompare\b|\bvs\.?\b|\bversus\b|\bwhich (?:is|are) (?:more|most)\b|\bmore\b.+\bor\b", q):
        return "compare"
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btally\b|\btotal\b|\bhow much traffic\b", q):
        return "count"
    if re.search(r"^(?:is|are) there\b|\b(?:is|are) there (?:any|a|an)\b|^any\b|\bdo you see\b|\bcan you see\b|\bdoes (?:the|this) (?:image|photo|picture|scene) (?:contain|have|show)\b|^(?:is|are) (?:any|anyone|anybody|someone|somebody)\b|\bexists?\b|\bpresent\b", q):
        return "exists"
    if re.search(r"\b(?:show|where|find|locate|detect|highlight|mark|point out|draw|box)\b", q):
        return "locate"
    if re.search(r"\bwhat(?:'s| is| do you see| can you see)\b|\bdescribe\b|\bsummar(?:y|ise|ize)\b|\boverview\b|\bbreakdown\b", q):
        return "summary"
    return None


def _match_targets(q: str) -> tuple[list[Target], list[OOVTerm], list[tuple[int, int]]]:
    spans: list[tuple[int, int, str, list[str]]] = []
    taken = [False] * (len(q) + 1)
    # OOV multi-word nouns first so "fire engine"/"police car" are not eaten by the lexicon
    oov: list[OOVTerm] = []
    for noun, parents in sorted(OOV_NOUNS.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"\b{re.escape(noun)}(?:e?s)?\b", q):
            if not any(taken[m.start():m.end()]):
                oov.append(OOVTerm(term=noun, nearest=parents))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    for rx, label, classes in _LEX:
        for m in rx.finditer(q):
            if any(taken[m.start():m.end()]):
                continue
            spans.append((m.start(), m.end(), label, classes))
            for i in range(m.start(), m.end()):
                taken[i] = True
    spans.sort()
    targets: list[Target] = []
    for s, e, label, classes in spans:
        mod = _SUBTYPE_RE.search(q[:s])
        if mod and label != "objects":
            term = (mod.group(1).strip() + " " + q[s:e]).strip()
            oov.append(OOVTerm(term=term, nearest=classes))
            continue
        if any(t.classes == classes for t in targets):
            continue
        targets.append(Target(label=label, classes=classes))
    return targets, oov, [(s, e) for s, e, *_ in spans]


def _unknown_phrase(q: str, region_span: str | None) -> str | None:
    m = _PHRASE_RE.search(q)
    if not m:
        return None
    words = [w for w in re.findall(r"[a-z][a-z'-]*", m.group("phrase")) if w not in _STOP]
    if region_span:
        rs = set(region_span.split())
        words = [w for w in words if w not in rs and w not in {"half", "side", "corner", "quadrant", "part", "area", "third"}]
    return " ".join(words) or None


def plan_rules(query: str) -> Plan:
    q = _norm(query)
    region, region_span = parse_region(q)
    # remove the region phrase before matching targets ("left" etc. are not objects)
    q_targets = q.replace(region_span, " ") if region_span else q
    targets, oov, _ = _match_targets(q_targets)
    intent = _intent(q)
    notes: list[str] = []

    if not targets and not oov:
        phrase = _unknown_phrase(q_targets, region_span)
        if phrase and intent in {"count", "exists", "locate", "compare"}:
            oov.append(OOVTerm(term=phrase, nearest=[]))

    if intent is None:
        intent = "count" if (targets or oov) else "summary" if q_targets.strip() == "" else None
    if intent == "compare" and len(targets) < 2:
        notes.append("compare needs two targets; answering as count")
        intent = "count"
    if intent == "summary" and targets and not re.search(r"\bdescribe\b|\bsummar|\boverview\b|\bbreakdown\b", q):
        intent = "count"

    if intent is None:
        return Plan(intent="summary", targets=[], region=region, oov=[], source="none", query=query, notes=["no intent or target recognised"])
    plan = Plan(intent=intent, targets=targets, region=region, oov=oov, source="rules", query=query, notes=notes)
    if not plan.recognized:
        plan.source = "none"
    return plan
