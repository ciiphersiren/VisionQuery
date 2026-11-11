"""The query plan: the only thing the planner is allowed to produce.

The answer engine executes a plan against detector boxes. Planners (rules or
LLM) never see the image and never produce numbers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..classes import CLASSES
from ..regions import REGIONS

Intent = Literal["count", "exists", "locate", "compare", "summary"]


class Target(BaseModel):
    label: str = Field(description="How the user referred to it, e.g. 'vehicles'")
    classes: list[str] = Field(description="Closed-set VisDrone classes this target maps to")

    @field_validator("classes")
    @classmethod
    def _known(cls, v):
        bad = [c for c in v if c not in CLASSES]
        if bad:
            raise ValueError(f"unknown classes {bad}")
        return list(dict.fromkeys(v))


class OOVTerm(BaseModel):
    term: str = Field(description="Out-of-vocabulary phrase, e.g. 'fire truck'")
    nearest: list[str] = Field(default_factory=list, description="Closed-set parent classes, if any")

    @field_validator("nearest")
    @classmethod
    def _known(cls, v):
        return [c for c in v if c in CLASSES]


class Plan(BaseModel):
    intent: Intent
    targets: list[Target] = Field(default_factory=list)
    region: str = "whole"
    oov: list[OOVTerm] = Field(default_factory=list)
    source: Literal["rules", "llm", "none"] = "rules"
    query: str = ""
    notes: list[str] = Field(default_factory=list)

    @field_validator("region")
    @classmethod
    def _region(cls, v):
        if v not in REGIONS:
            raise ValueError(f"unknown region {v!r}")
        return v

    @property
    def is_oov(self) -> bool:
        return bool(self.oov)

    @property
    def recognized(self) -> bool:
        return bool(self.targets) or self.intent == "summary" or bool(self.oov)

    def key(self) -> tuple:
        """Fields compared when scoring planners."""
        return (
            self.intent,
            tuple(tuple(sorted(t.classes)) for t in self.targets),
            self.region,
            tuple(sorted(o.term for o in self.oov)),
        )
