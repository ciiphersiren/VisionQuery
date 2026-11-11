"""Answer engine: executes a Plan against detector boxes.

Every number in an answer is computed from boxes above a calibrated
threshold, and every answer carries those boxes as evidence. Free text is
only ever a rendering of those numbers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .calibration import PipelineCalibration
from .classes import CLASS_EMOJI, CLASSES, CLASS_TO_ID
from .detector import Detections
from .planner.schema import Plan, Target
from .openvocab import iou_matrix
from .regions import REGIONS, in_region, region_px

WEAK_RECALL = 0.80  # below this held-out existence recall we warn on "no" answers


class Evidence(BaseModel):
    box: list[float]
    label: str
    conf: float
    target: str
    strength: Literal["counted", "weak", "open"] = "counted"


class GroupResult(BaseModel):
    label: str
    classes: list[str]
    count: int
    count_range: list[int]
    exists: bool
    per_class: dict[str, int]


class Answer(BaseModel):
    text: str
    intent: str
    value: int | bool | str | dict | None
    region: str
    region_box: list[float]
    groups: list[GroupResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    oov: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list, description="per class: thresholds and how many boxes were kept / dropped")
    threshold_mode: Literal["calibrated", "override"] = "calibrated"
    open_vocab: list[dict] = Field(default_factory=list, description="open-vocabulary lookups for out-of-vocabulary terms (zero-shot, uncalibrated)")
    mood: Literal["confident", "hedged", "stumped", "curious"] = "confident"


def _plural(n: int, label: str) -> str:
    if n == 1:
        sing = {"buses": "bus", "people": "person", "people (sitting/riding)": "person sitting/riding", "pedestrians": "pedestrian", "awning-tricycles": "awning-tricycle", "three-wheelers": "three-wheeler", "two-wheelers": "two-wheeler", "four-wheelers": "four-wheeler", "vehicles": "vehicle", "objects": "object"}
        return sing.get(label, label[:-1] if label.endswith("s") else label)
    return label


def _term_n(n: int, term: str) -> str:
    """'1 tree', '3 trees', '2 buses' for a free-text term the user typed (singular or plural)."""
    from .openvocab import singular

    words = term.split()
    one = " ".join(words[:-1] + [singular(words[-1])]) if words else term
    if n == 1:
        return one
    last = one.split()[-1]
    plural = last + ("es" if last.endswith(("s", "x", "ch", "sh")) else "s") if not last.endswith("y") or last.endswith(("ay", "ey", "oy")) else last[:-1] + "ies"
    return " ".join(one.split()[:-1] + [plural])


def _class_plural(c: str) -> str:
    return {"bus": "buses", "people": "people", "motor": "motorbikes"}.get(c, c + "s")


def _region_phrase(region: str) -> str:
    return "" if region == "whole" else f" in the {region}"


def _emoji(classes: list[str]) -> str:
    return CLASS_EMOJI.get(classes[0], "🔎") if len(classes) == 1 else ("🧑‍🤝‍🧑" if set(classes) == {"pedestrian", "people"} else "🚦")


class Answerer:
    def __init__(self, calib: PipelineCalibration, count_threshold: float | None = None, ovd_threshold: float = 0.1, ovd_subtype_threshold: float = 0.05,
                 supported: list[str] | None = None, model_name: str = ""):
        """``count_threshold`` overrides every per-class count threshold (UI slider) and the open-vocabulary one.
        ``supported``: the classes the chosen model was trained on (None = all ten)."""
        self.supported = list(supported) if supported is not None else list(CLASSES)
        self.model_name = model_name or "This model"
        self.ovd_thr = float(count_threshold) if count_threshold is not None else ovd_threshold
        self.ovd_sub_thr = ovd_subtype_threshold
        self.count_thr = calib.count_thr.copy()
        self.exist_thr = calib.exist_thr.copy()
        self.override = count_threshold is not None
        if self.override:
            known = ~np.isnan(self.count_thr)
            self.count_thr[known] = float(count_threshold)
            self.exist_thr = np.where(known, np.fmin(self.exist_thr, self.count_thr), np.nan)
        self.cal = calib
        self._trace: dict[str, dict] = {}

    # ------------------------------------------------------------------ core numbers
    def _group(self, det: Detections, mask_region: np.ndarray, t: Target):
        ids = [CLASS_TO_ID[c] for c in t.classes]
        per_class, lo_sum, hi_sum = {}, 0, 0
        counted, weak = [], []
        exists = False
        for k in ids:
            m = mask_region & (det.cls == k)
            cm = m & (det.conf >= self.count_thr[k])
            em = m & (det.conf >= self.exist_thr[k])
            n = int(cm.sum())
            self._trace[CLASSES[k]] = {
                "class": CLASSES[k], "count_threshold": round(float(self.count_thr[k]), 3), "exist_threshold": round(float(self.exist_thr[k]), 3),
                "detected": int((det.cls == k).sum()), "in_region": int(m.sum()), "counted": n,
                "weak": int((em & ~cm).sum()), "below_threshold": int((m & ~em).sum()),
            }
            per_class[CLASSES[k]] = n
            lo, hi = self.cal.interval(CLASSES[k], n)
            lo_sum += lo
            hi_sum += hi
            exists |= bool(em.any())
            counted += list(np.where(cm)[0])
            weak += list(np.where(em & ~cm)[0])
        total = sum(per_class.values())
        g = GroupResult(label=t.label, classes=t.classes, count=total, count_range=[min(lo_sum, total), max(hi_sum, total)], exists=exists or total > 0, per_class=per_class)
        return g, counted, weak

    def _evidence(self, det, idx, target, strength):
        return [Evidence(box=[round(float(v), 1) for v in det.xyxy[i]], label=CLASSES[int(det.cls[i])], conf=round(float(det.conf[i]), 3), target=target, strength=strength) for i in idx]

    def _weak_recall_caveat(self, classes: list[str]) -> str | None:
        weak = [c for c in classes if self.cal.exist_recall.get(c, 1.0) < WEAK_RECALL]
        if not weak:
            return None
        names = {"people": "people sitting or riding", "motor": "motorbikes", "bus": "buses"}
        parts = " and ".join(f"{names.get(c, c + 's')} in about {self.cal.exist_recall[c]:.0%}" for c in weak)
        return f"I find {parts} of the images that really have them, so small or hidden ones may have slipped past me."

    # ------------------------------------------------------------------ main
    def __call__(self, plan: Plan, det: Detections, ovd: dict[str, Detections] | None = None) -> Answer:
        """``ovd`` maps each OOV term to YOLOE detections prompted with [term, *parent prompts]."""
        self._trace = {}
        self._ovd = ovd or {}
        self._open: list[dict] = []
        ans = self._answer(plan, det)
        ans.open_vocab = self._open
        ans.trace = list(self._trace.values())
        ans.threshold_mode = "override" if self.override else "calibrated"
        return ans

    def _restrict(self, plan: Plan) -> tuple[Plan, list[str], list[str]]:
        """Keep only the classes the chosen model knows. -> (plan, lines for targets it cannot see at all, caveats)."""
        if len(self.supported) == len(CLASSES):
            return plan, [], []
        sup = set(self.supported)
        known = ", ".join(_class_plural(c) for c in self.supported)
        lines, caveats, targets = [], [], []
        for t in plan.targets:
            ok = [c for c in t.classes if c in sup]
            if not ok:
                lines.append(f"🚫 {self.model_name} was only trained on {known}, so it can't look for {t.label}. Pick a VisDrone model in the menu to ask about them.")
                continue
            if len(ok) < len(t.classes):
                caveats.append(f"{self.model_name} only knows {known}, so other kinds of {t.label} are not in this count.")
            targets.append(Target(label=t.label, classes=ok))
        oov = [o.model_copy(update={"nearest": [c for c in o.nearest if c in sup]}) for o in plan.oov]
        return plan.model_copy(update={"targets": targets, "oov": oov}), lines, caveats

    def _answer(self, plan: Plan, det: Detections) -> Answer:
        W, H = det.width, det.height
        mask = in_region(det.centers, plan.region, W, H) if len(det) else np.zeros(0, bool)
        rbox = [round(v, 1) for v in region_px(plan.region, W, H)]
        where = _region_phrase(plan.region)
        had_targets = bool(plan.targets)
        plan, unsupported, caveats = self._restrict(plan)
        if had_targets and not plan.targets and not plan.oov:
            return Answer(text=" ".join(unsupported), intent=plan.intent, value=None, region=plan.region, region_box=rbox, caveats=caveats, mood="stumped")
        oov = [o.model_dump() for o in plan.oov]

        targets = list(plan.targets)
        # OOV with a closed-set parent: answer about the parent, clearly labelled
        for o in plan.oov:
            if o.nearest and not any(t.classes == o.nearest for t in targets):
                targets.append(Target(label=" or ".join(_class_plural(c) for c in o.nearest), classes=o.nearest))

        if plan.source == "none" or (not targets and plan.intent != "summary" and not plan.oov):
            return Answer(
                text="🤔 I didn't catch what to look for. Try something like *“how many cars are in the left half?”* or *“any buses?”*",
                intent=plan.intent, value=None, region=plan.region, region_box=rbox, mood="curious",
            )

        if plan.intent == "summary" or (not plan.targets and not plan.oov):
            return self._summary(plan, det, mask, rbox, where)

        results, evidence = [], []
        for t in targets:
            g, counted, weak = self._group(det, mask, t)
            results.append(g)
            evidence += self._evidence(det, counted, t.label, "counted")
            if plan.intent == "exists":
                evidence += self._evidence(det, weak, t.label, "weak")

        oov_lines = []
        found_open = 0
        for o in plan.oov:
            od = self._ovd.get(o.term)
            thr = self.ovd_sub_thr if o.nearest else float((od.meta or {}).get("threshold", self.ovd_thr) if od is not None else self.ovd_thr)
            if od is not None and len(od):
                om = in_region(od.centers, plan.region, W, H) & (od.cls == 0)
                keep = np.where(om & (od.conf >= thr))[0]
            else:
                om, keep = np.zeros(0, bool), np.zeros(0, int)
            if o.nearest:
                g = next(r for r in results if r.classes == o.nearest)
                if od is None:
                    oov_lines.append(
                        f"🧩 *{o.term}* isn't something I can tell apart — I only know VisDrone's 10 classes. "
                        f"I see **{g.count} {_plural(g.count, g.label)}**{where}; any {o.term} would be among those."
                    )
                    continue
                # subtype: fine-tuned parent boxes that YOLOE (contrastively) calls `term`
                parent_idx = [i for i in np.where(mask & np.isin(det.cls, [CLASS_TO_ID[c] for c in o.nearest]))[0] if det.conf[i] >= self.count_thr[det.cls[i]]]
                ious = iou_matrix(det.xyxy[parent_idx], od.xyxy[keep]) if len(parent_idx) and len(keep) else np.zeros((len(parent_idx), 0))
                matched = [parent_idx[i] for i in range(len(parent_idx)) if ious.shape[1] and ious[i].max() >= 0.5]
                found_open += len(matched)
                evidence += [Evidence(box=[round(float(v), 1) for v in det.xyxy[i]], label=o.term, conf=round(float(od.conf[keep[int(ious[j].argmax())]]), 3), target=o.term, strength="open") for j, i in enumerate(parent_idx) if i in matched]
                self._open.append({"term": o.term, "mode": "subtype", "parent": o.nearest, "parent_count": g.count, "count": len(matched), "threshold": round(self.ovd_sub_thr, 3), "model": od.meta.get("model"), "prompts": od.meta.get("prompts")})
                oov_lines.append(
                    f"🧩 My trained detector can't tell *{o.term}* apart, so I asked an open-vocabulary detector: of the "
                    f"{g.count} {_plural(g.count, g.label)}{where}, it thinks **{len(matched)}** look like {o.term}. Treat that part as a lead, not a verified count."
                )
            elif od is not None:
                n = len(keep)
                found_open += n
                evidence += [Evidence(box=[round(float(v), 1) for v in od.xyxy[i]], label=o.term, conf=round(float(od.conf[i]), 3), target=o.term, strength="open") for i in keep]
                taught = bool(od.meta.get("taught"))
                self._open.append({"term": o.term, "mode": "taught" if taught else "novel", "count": n, "threshold": round(thr, 3), "model": od.meta.get("model"),
                                   "prompts": od.meta.get("prompts"), "seen": int(om.sum()), "ap_vs_teacher": od.meta.get("ap_vs_teacher")})
                self._trace[f"open:{o.term}"] = {
                    "class": o.term, "count_threshold": round(thr, 3), "exist_threshold": round(thr, 3),
                    "detected": int((od.cls == 0).sum()), "in_region": int(om.sum()), "counted": n, "weak": 0,
                    "below_threshold": int(om.sum()) - n, "open_vocab": True,
                }
                if plan.intent == "exists":
                    head = f"Possibly — an open-vocabulary detector spots {n} {_term_n(n, o.term)}{where}" if n else f"I don't see any {o.term}{where}, even with an open-vocabulary detector"
                else:
                    head = f"*{o.term}* is beyond my 10 trained classes, so I asked an open-vocabulary detector: it finds **{n}**{where}"
                if taught:
                    ap = od.meta.get("ap_vs_teacher")
                    score = f" (on validation frames it agrees with SAM 3 at {ap:.0f} AP for this)" if isinstance(ap, (int, float)) else ""
                    head = head.replace("an open-vocabulary detector", "my drone-tuned open-vocabulary detector")
                    oov_lines.append(f"🌳 {head}. It learned *{od.meta['prompts'][0]}* from SAM 3's labels{score}, not from people, so treat it as a strong lead.")
                elif od.meta.get("fallback_from"):
                    oov_lines.append(f"🔭 {head}. My drone-tuned detector found none, so this comes from the general one, which isn't tuned for drone footage: treat it as a lead.")
                else:
                    oov_lines.append(f"🔭 {head}. It isn't tuned for drone footage, so treat this as a lead, not a count.")
            else:
                oov_lines.append(f"🛸 *{o.term}* is outside my vocabulary (I know: {', '.join(CLASSES)}), so I can't count or find it.")

        in_vocab = [r for r in results if any(t.classes == r.classes for t in plan.targets)]
        mood = ("hedged" if found_open else "stumped") if plan.oov and not in_vocab else "confident"

        if plan.intent == "compare" and len(in_vocab) >= 2:
            a, b = in_vocab[0], in_vocab[1]
            if a.count == b.count:
                head = f"⚖️ It's a tie{where}: {a.count} {_plural(a.count, a.label)} and {b.count} {_plural(b.count, b.label)}."
                value = "tie"
            else:
                w, l = (a, b) if a.count > b.count else (b, a)
                head = f"🏆 **{w.label.capitalize()}** win{where}: {w.count} vs {l.count} {_plural(l.count, l.label)}."
                value = w.label
            close = abs(a.count - b.count) <= max(1, 0.15 * max(a.count, b.count))
            if close:
                caveats.append("That's close enough that counting noise could flip it.")
                mood = "hedged"
            text = head
        elif plan.intent == "exists":
            lines = []
            for r in in_vocab:
                e = _emoji(r.classes)
                if r.exists and r.count > 0:
                    lines.append(f"{e} Yes — {r.count} {_plural(r.count, r.label)}{where}.")
                elif r.exists:
                    lines.append(f"{e} Probably — I see a faint {_plural(1, r.label)}{where}, but it's below my counting threshold.")
                    mood = "hedged"
                else:
                    lines.append(f"{e} I don't see any {r.label}{where}.")
                    c = self._weak_recall_caveat(r.classes)
                    if c:
                        caveats.append(c)
                        mood = "hedged"
            text = " ".join(lines)
            value = all(r.exists for r in in_vocab) if in_vocab else None
        elif plan.intent == "locate":
            lines = [f"{_emoji(r.classes)} Highlighted {r.count} {_plural(r.count, r.label)}{where}." if r.count else f"{_emoji(r.classes)} Couldn't find any {r.label}{where}." for r in in_vocab]
            text = " ".join(lines)
            value = sum(r.count for r in in_vocab)
        else:  # count
            lines = []
            for r in in_vocab:
                rng = ""
                lo, hi = r.count_range
                if hi - lo >= 1:
                    rng = f" ({self.cal.interval_label} {lo}–{hi})"
                    if hi - lo >= max(3, 0.5 * max(r.count, 1)):
                        mood = "hedged"
                lines.append(f"{_emoji(r.classes)} I count **{r.count} {_plural(r.count, r.label)}**{where}{rng}.")
            text = " ".join(lines)
            value = in_vocab[0].count if len(in_vocab) == 1 else {r.label: r.count for r in in_vocab}

        if not in_vocab:
            text = ""
            value = self._open[0]["count"] if len(self._open) == 1 else ({o["term"]: o["count"] for o in self._open} if self._open else None)
        text = " ".join([text] + oov_lines + unsupported).strip()
        tiny = sum(1 for ev in evidence if (ev.box[2] - ev.box[0]) < 16 or (ev.box[3] - ev.box[1]) < 16)
        if tiny and tiny >= 0.3 * max(len(evidence), 1) and len(evidence) >= 5:
            caveats.append(f"{tiny} of these are under 16 px — the hardest size for me.")
        return Answer(text=text, intent=plan.intent, value=value, region=plan.region, region_box=rbox, groups=results, evidence=evidence, caveats=caveats, oov=oov, mood=mood)

    def _summary(self, plan, det, mask, rbox, where) -> Answer:
        results, evidence = [], []
        for c in self.supported:
            g, counted, _ = self._group(det, mask, Target(label=c, classes=[c]))
            if g.count:
                results.append(g)
                evidence += self._evidence(det, counted, c, "counted")
        results.sort(key=lambda r: -r.count)
        total = sum(r.count for r in results)
        if not results:
            what = "my 10 classes" if len(self.supported) == len(CLASSES) else "the classes this model knows (" + ", ".join(self.supported) + ")"
            return Answer(text=f"🌫️ I don't see any of {what}{where}.", intent="summary", value={}, region=plan.region, region_box=rbox, mood="hedged")
        parts = ", ".join(f"{CLASS_EMOJI[r.label]} {r.count} {r.label if r.count == 1 else _class_plural(r.label)}" for r in results[:6])
        rest = f" and {len(results) - 6} more classes" if len(results) > 6 else ""
        return Answer(
            text=f"📋 {total} objects{where}: {parts}{rest}.", intent="summary", value={r.label: r.count for r in results},
            region=plan.region, region_box=rbox, groups=results, evidence=evidence,
        )
