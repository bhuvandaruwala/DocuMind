"""
Planner agent: classifies the question and writes 1-3 focused search queries.

Uses the LLM when available (one short JSON call). If the LLM is unavailable or
returns something unparseable, a rule-based planner takes over so the pipeline
never stops here. On a retry it receives the Validator's feedback and broadens
the search (different queries, more passages).
"""

from __future__ import annotations

import re
import time
from typing import Dict, List

from app.agents.state import AgentContext, AgentState, add_trace
from app.llm import LLMError, invoke_llm
from app.rag import PLANNER_FEEDBACK, PLANNER_PROMPT, parse_json_object

CATEGORIES = {"lookup", "comparison", "aggregate", "summary", "out_of_scope"}

_AGG = re.compile(r"\b(total|sum|average|avg|mean|count|number of|highest|lowest|most|least|max(imum)?|min(imum)?|top \d+|largest|smallest|how many (laptops|rows|items|units|orders|records|licen[cs]es|seats|stores|devices))\b", re.I)
_CMP = re.compile(r"\b(compare|comparison|versus|vs\.?|difference|differ|relative to|against)\b", re.I)
_SUM = re.compile(r"\b(summari[sz]e|summary|overview|key points|main points|what is .* about)\b", re.I)


def heuristic_plan(question: str, top_k: int) -> Dict:
    if _CMP.search(question):
        cat = "comparison"
    elif _AGG.search(question):
        cat = "aggregate"
    elif _SUM.search(question):
        cat = "summary"
    else:
        cat = "lookup"
    return {"category": cat, "queries": [question], "k": top_k, "source": "rules"}


def _sanitize_plan(obj: Dict, question: str, top_k: int) -> Dict:
    cat = str(obj.get("category", "lookup")).strip().lower()
    if cat not in CATEGORIES:
        cat = "lookup"
    queries = obj.get("queries") or []
    if isinstance(queries, str):
        queries = [queries]
    queries = [str(q).strip()[:200] for q in queries if str(q).strip()][:3]
    if not queries:
        queries = [question]
    try:
        k = int(obj.get("k", top_k))
    except (TypeError, ValueError):
        k = top_k
    return {"category": cat, "queries": queries, "k": max(3, min(k, 10)), "source": "llm"}


def plan_node(state: AgentState, ctx: AgentContext) -> AgentState:
    t0 = time.time()
    question = state["question"]
    attempt = state.get("attempt", 1)
    prev = state.get("plan") or {}
    s = ctx.settings
    note = ""

    plan = heuristic_plan(question, s.top_k)
    if ctx.llm is not None:
        files = ", ".join(ctx.store.list_sources().keys()) or "(none)"
        feedback = ""
        if attempt > 1:
            feedback = PLANNER_FEEDBACK.format(reason=state.get("feedback", ""), queries=prev.get("queries", []))
        prompt = PLANNER_PROMPT.format(files=files, question=question, feedback=feedback)
        try:
            raw = invoke_llm(ctx.llm, prompt, s, retries=2)
            obj = parse_json_object(raw)
            if obj:
                plan = _sanitize_plan(obj, question, s.top_k)
            else:
                note = "LLM plan was not valid JSON; used rule-based plan"
        except LLMError as e:
            note = f"LLM unavailable for planning ({e}); used rule-based plan"

    # Always keep the user's own wording as one query: it is the safest fallback.
    if question not in plan["queries"]:
        plan["queries"] = (plan["queries"] + [question])[:4]

    if attempt > 1:  # broaden on retry
        plan["k"] = min(plan["k"] + 3, 12)
        old: List[str] = prev.get("queries", [])
        plan["queries"] = [q for q in plan["queries"] if q not in old] or plan["queries"]
        if question not in plan["queries"]:
            plan["queries"].append(question)
        plan["broadened"] = True

    summary = f"{plan['category']} question → {len(plan['queries'])} search quer{'y' if len(plan['queries']) == 1 else 'ies'}, k={plan['k']} ({plan['source']})"
    trace = add_trace(state, ctx, "Planner", summary, t0, queries=plan["queries"], note=note)
    return {"plan": plan, "trace": trace}
