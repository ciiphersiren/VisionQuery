"""Optional LLM planner (OpenAI-compatible chat API, e.g. the Hugging Face router).

Only used by the rules_first policy when the rules do not recognise a
target. The LLM returns JSON that must validate against ``Plan``; anything
else is discarded. It never sees the image and never produces numbers.

Configure with env vars:
    SKYQUERY_LLM_BASE_URL  (default https://router.huggingface.co/v1)
    SKYQUERY_LLM_MODEL     (default Qwen/Qwen3.6-35B-A3B)
    HF_TOKEN or SKYQUERY_LLM_API_KEY
"""

from __future__ import annotations

import json
import os
import re

from ..classes import CLASSES
from ..regions import REGIONS
from .schema import Plan

SYSTEM = f"""You convert a question about an aerial drone image into a JSON plan.
You never answer the question yourself.

Closed-set detector classes (VisDrone): {", ".join(CLASSES)}.
VisDrone conventions: "pedestrian" = a person standing or walking; "people" = a person
sitting, lying or riding. Plain "people/person/humans" means both classes.
Regions: {", ".join(REGIONS)}.

Rules:
- intent is one of count, exists, locate, compare, summary.
- targets: list of {{"label": str, "classes": [closed-set class names]}}.
- If the user asks about something the detector cannot see or verify (an object type not in
  the list, a sub-type such as "fire truck" or "taxi", a colour or other attribute), add it to
  "oov" as {{"term": str, "nearest": [closest closed-set classes or empty]}} and DO NOT put it in targets.
- Return only the JSON object, no prose.

Example: "how many fire trucks are on the left?" ->
{{"intent":"count","targets":[],"region":"left half","oov":[{{"term":"fire truck","nearest":["truck"]}}]}}
"""


class LLMPlanner:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None, timeout: float = 20.0):
        self.base_url = (base_url or os.environ.get("SKYQUERY_LLM_BASE_URL") or "https://router.huggingface.co/v1").rstrip("/")
        self.model = model or os.environ.get("SKYQUERY_LLM_MODEL") or "Qwen/Qwen3.6-35B-A3B"
        self.api_key = api_key or os.environ.get("SKYQUERY_LLM_API_KEY") or os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self._no_thinking_flag = False  # remembered per model

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _call(self, query: str) -> str:
        import httpx

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}],
            "temperature": 0,
            "max_tokens": 300,
        }
        if not self._no_thinking_flag:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        if r.status_code == 400 and not self._no_thinking_flag:
            self._no_thinking_flag = True
            body.pop("chat_template_kwargs", None)
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def plan(self, query: str) -> Plan | None:
        if not self.available:
            return None
        try:
            text = self._call(query)
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return None
            data = json.loads(m.group(0))
            data.update(source="llm", query=query)
            return Plan.model_validate(data)
        except Exception:
            return None
