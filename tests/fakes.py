"""Deterministic stand-in for a chat model so the agent graph can be tested offline (no API key)."""

from __future__ import annotations

import json
import re
from typing import Any, List

from langchain_core.messages import AIMessage


class ScriptedLLM:
    """Mimics the parts of a LangChain chat model the agents use (`.invoke`)."""

    def __init__(self, reasoner_answers: List[str] | None = None, judge_supported: bool = True):
        self.reasoner_answers = list(reasoner_answers or [])
        self.judge_supported = judge_supported
        self.calls: List[str] = []

    def _text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        return "\n".join(getattr(m, "content", str(m)) for m in prompt)

    def invoke(self, prompt: Any) -> AIMessage:
        text = self._text(prompt)
        if "You plan document searches" in text:
            self.calls.append("planner")
            q = re.search(r"Question: (.*)", text).group(1).strip()
            return AIMessage(content=json.dumps({"category": "lookup", "queries": [q], "k": 5}))
        if "strict fact-checker" in text:
            self.calls.append("validator")
            return AIMessage(content=json.dumps({"supported": self.judge_supported, "addresses_question": True, "issues": ""}))
        self.calls.append("reasoner")
        if self.reasoner_answers:
            return AIMessage(content=self.reasoner_answers.pop(0))
        # default: quote the first sentence of the first source, cited
        m = re.search(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', text, re.DOTALL)
        if not m:
            return AIMessage(content="I could not find this information in the uploaded documents.")
        first = re.split(r"(?<=[.!?])\s", " ".join(m.group(2).split()))[0]
        return AIMessage(content=f"{first} [{m.group(1)}]\nCONFIDENCE: high")
