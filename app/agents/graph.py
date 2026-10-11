"""
LangGraph orchestration.

    START → guard → planner → retriever → reasoner → validator ─┬─ done → finalize → END
                      ▲                                          │
                      └──────────── retry (max N passes) ────────┘

* guard     - input validation / jailbreak check on the question
* planner   - classify + rewrite into search queries        (app/agents/planner.py)
* retriever - hybrid search + MMR, labelled passages         (app/agents/retriever.py)
* reasoner  - grounded, cited answer from the LLM            (app/agents/reasoner.py)
* validator - citation / number / safety / LLM fact checks   (app/agents/validator.py)
* finalize  - confidence score, citations list, safe fallbacks
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.planner import plan_node
from app.agents.reasoner import reason_node
from app.agents.retriever import retrieve_node
from app.agents.state import AgentContext, AgentState, add_trace
from app.agents.validator import route_after_validation, validate_node
from app.config import NOT_FOUND_MESSAGE
from app.guardrails import extract_citations, validate_question

SAFE_REFUSAL = ("I can't provide that answer safely. The retrieved content contained instructions or "
                "material that failed the safety checks. Please rephrase your question.")


def guard_node(state: AgentState, ctx: AgentContext) -> AgentState:
    t0 = time.time()
    check = validate_question(state["question"], ctx.settings.max_question_chars)
    if not check.ok:
        trace = add_trace(state, ctx, "Guard", f"blocked: {check.message}", t0, flags=check.flags)
        return {"blocked": True, "final": {"answer": check.message, "status": "blocked"}, "trace": trace}
    trace = add_trace(state, ctx, "Guard", "question accepted", t0)
    return {"question": check.cleaned, "blocked": False, "attempt": 1, "trace": trace}


def next_attempt_node(state: AgentState, ctx: AgentContext) -> AgentState:
    return {"attempt": state.get("attempt", 1) + 1}


def _confidence(state: AgentState) -> float:
    draft = state.get("draft") or {}
    v = state.get("validation") or {}
    sources = state.get("sources") or []
    if draft.get("not_found") or not sources:
        return 0.0
    base = {"high": 0.9, "medium": 0.7, "low": 0.45}.get(draft.get("self_confidence", "medium"), 0.6)
    cited = set(extract_citations(draft.get("answer", "")))
    cited_src = [s for s in sources if s["label"] in cited] or sources[:1]
    retrieval = max(min(1.0, max(s["similarity"] * 1.6, s["keyword_score"]) ) for s in cited_src)
    score = 0.65 * base + 0.35 * retrieval
    score -= 0.07 * len(v.get("warnings", []))
    if draft.get("retrieval_only"):
        score = min(score, 0.4)
    if not v.get("ok"):
        score = min(score, 0.3)
    return round(max(0.0, min(score, 1.0)), 2)


def finalize_node(state: AgentState, ctx: AgentContext) -> AgentState:
    t0 = time.time()
    if state.get("blocked"):
        return {}
    draft = state.get("draft") or {}
    v = state.get("validation") or {}
    sources = state.get("sources") or []

    if state.get("error"):
        final = {"answer": f"⚠️ {state['error']}", "status": "error", "confidence": 0.0, "citations": []}
    elif v.get("safety_failure") and not v.get("ok"):
        final = {"answer": SAFE_REFUSAL, "status": "refused", "confidence": 0.0, "citations": []}
    else:
        answer = draft.get("answer", NOT_FOUND_MESSAGE)
        status = "not_found" if draft.get("not_found") else ("verified" if v.get("ok") else "unverified")
        if draft.get("retrieval_only"):
            status = "retrieval_only"
        cited = extract_citations(answer)
        citations = [s for s in sources if s["label"] in cited]
        final = {
            "answer": answer,
            "status": status,
            "confidence": _confidence(state),
            "citations": citations,
            "issues": v.get("issues", []) if not v.get("ok") else [],
            "warnings": v.get("warnings", []),
        }
    final["attempts"] = state.get("attempt", 1)
    final["category"] = (state.get("plan") or {}).get("category")
    trace = add_trace(state, ctx, "Finalize", f"status={final['status']}, confidence={final.get('confidence')}", t0)
    return {"final": final, "trace": trace}


def build_graph(ctx: AgentContext):
    g = StateGraph(AgentState)
    g.add_node("guard", lambda s: guard_node(s, ctx))
    g.add_node("planner", lambda s: plan_node(s, ctx))
    g.add_node("retriever", lambda s: retrieve_node(s, ctx))
    g.add_node("reasoner", lambda s: reason_node(s, ctx))
    g.add_node("validator", lambda s: validate_node(s, ctx))
    g.add_node("next_attempt", lambda s: next_attempt_node(s, ctx))
    g.add_node("finalize", lambda s: finalize_node(s, ctx))

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", lambda s: "stop" if s.get("blocked") else "go",
                            {"stop": "finalize", "go": "planner"})
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "reasoner")
    g.add_conditional_edges("reasoner", lambda s: "error" if s.get("error") else "ok",
                            {"error": "finalize", "ok": "validator"})
    g.add_conditional_edges(
        "validator",
        lambda s: route_after_validation(s, ctx.settings.max_agent_loops),
        {"retry": "next_attempt", "done": "finalize"},
    )
    g.add_edge("next_attempt", "planner")
    g.add_edge("finalize", END)
    return g.compile()


def answer_question(question: str, ctx: AgentContext) -> Dict[str, Any]:
    """Run the full agent workflow and return {final, trace, sources, plan}."""
    t0 = time.time()
    graph = build_graph(ctx)
    state = graph.invoke({"question": question, "trace": [], "attempt": 1}, config={"recursion_limit": 40})
    final = dict(state.get("final") or {})
    final.setdefault("status", "error")
    final["elapsed_s"] = round(time.time() - t0, 2)
    return {
        "final": final,
        "trace": state.get("trace", []),
        "sources": state.get("sources", []),
        "plan": state.get("plan", {}),
    }


def mermaid_diagram() -> Optional[str]:
    """Mermaid source of the compiled graph (shown in the UI 'How it works' tab)."""
    try:
        from app.config import Settings
        dummy = AgentContext(settings=Settings(), store=None)
        return build_graph(dummy).get_graph().draw_mermaid()
    except Exception:  # noqa: BLE001
        return None
