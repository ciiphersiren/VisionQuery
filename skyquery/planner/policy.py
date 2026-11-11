"""Planner policies. Default is rules_first, as proposed in the report (§9)."""

from __future__ import annotations

from typing import Literal

from .llm import LLMPlanner
from .rules import plan_rules
from .schema import Plan

Policy = Literal["rules", "llm", "rules_first", "governed"]


class Planner:
    def __init__(self, policy: Policy = "rules_first", llm: LLMPlanner | None = None):
        self.policy = policy
        self.llm = llm if llm is not None else LLMPlanner()

    def __call__(self, query: str) -> Plan:
        rules = plan_rules(query)
        if self.policy == "rules":
            return rules
        if self.policy == "llm":
            return self.llm.plan(query) or rules
        if self.policy == "rules_first":
            # rules when they recognise a target; LLM only otherwise
            if rules.source == "rules" and rules.recognized and not _only_unknown_oov(rules):
                return rules
            llm = self.llm.plan(query)
            return llm or rules
        if self.policy == "governed":
            llm = self.llm.plan(query)
            if llm is None:
                return rules
            # LLM wins unless it dropped an OOV term the rules flagged
            rule_terms = {o.term for o in rules.oov}
            llm_terms = {o.term for o in llm.oov}
            if rule_terms and not rule_terms <= llm_terms:
                return rules
            return llm
        raise ValueError(self.policy)


def _only_unknown_oov(p: Plan) -> bool:
    """Rules found nothing in-vocabulary and only a free-text OOV guess: worth asking the LLM."""
    return not p.targets and bool(p.oov) and all(not o.nearest for o in p.oov)
