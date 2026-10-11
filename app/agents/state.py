"""Shared state and runtime context for the LangGraph agent workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict

from app.config import Settings


class AgentState(TypedDict, total=False):
    question: str
    attempt: int                 # 1-based pass number through Planner -> Validator
    blocked: bool                # input guard rejected the question
    plan: Dict[str, Any]         # {category, queries, k, source}
    sources: List[Dict[str, Any]]  # retrieved, labelled S1..Sn
    candidates_found: int        # how many passages retrieval saw before filtering
    draft: Dict[str, Any]        # {answer, self_confidence, not_found}
    validation: Dict[str, Any]   # {ok, issues, warnings, safety_failure}
    feedback: str                # validator -> planner/reasoner on retry
    final: Dict[str, Any]        # what the UI shows
    error: str
    trace: List[Dict[str, Any]]


@dataclass
class AgentContext:
    settings: Settings
    store: Any                               # app.store.VectorStore
    llm: Optional[Any] = None                # LangChain chat model, or None for retrieval-only mode
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None   # UI progress callback


def add_trace(state: AgentState, ctx: AgentContext, agent: str, summary: str, t0: float, **details) -> List[Dict]:
    entry = {
        "agent": agent,
        "attempt": state.get("attempt", 1),
        "summary": summary,
        "ms": int((time.time() - t0) * 1000),
        "details": details,
    }
    if ctx.on_step:
        try:
            ctx.on_step(entry)
        except Exception:  # noqa: BLE001 - UI callbacks must never break the pipeline
            pass
    return list(state.get("trace", [])) + [entry]
