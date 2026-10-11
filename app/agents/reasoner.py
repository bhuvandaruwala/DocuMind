"""
Reasoner agent: drafts the grounded, cited answer.

Sends the question plus the labelled passages to the LLM with a strict grounding
prompt (answer only from sources, cite [S#], say "not found" otherwise).
Without an LLM (no API key yet) it runs in *retrieval-only mode* and returns the
most relevant passages verbatim, so the app is still useful for a first look.
"""

from __future__ import annotations

import re
import time

from app.agents.state import AgentContext, AgentState, add_trace
from app.config import NOT_FOUND_MESSAGE
from app.embeddings import tokenize
from app.llm import LLMError, invoke_llm
from app.rag import build_reasoner_messages, parse_answer


def _best_sentences(text: str, question: str, n: int = 2) -> str:
    """Extractive fallback: the sentences of a passage that share the most words with the question."""
    q = set(tokenize(question))
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # undo hard line-wraps inside paragraphs
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n{2,}", joined) if len(x.strip()) > 15]
    if not sentences:
        return " ".join(text.split())[:300]
    ranked = sorted(range(len(sentences)), key=lambda i: -len(q & set(tokenize(sentences[i]))))[:n]
    return " … ".join(sentences[i] for i in sorted(ranked))[:450]


def _retrieval_only_answer(sources, question: str) -> dict:
    lines = ["**Retrieval-only mode** (add a Gemini API key in ⚙️ Settings to get a written answer). "
             "Most relevant passages:"]
    for s in sources[:3]:
        lines.append(f"- {_best_sentences(s['text'], question)} [{s['label']}]")
    return {"answer": "\n".join(lines), "self_confidence": "low", "not_found": False, "retrieval_only": True}


def reason_node(state: AgentState, ctx: AgentContext) -> AgentState:
    t0 = time.time()
    sources = state.get("sources", [])

    if not sources:
        draft = {"answer": NOT_FOUND_MESSAGE, "self_confidence": "low", "not_found": True}
        trace = add_trace(state, ctx, "Reasoner", "no relevant passages → 'not found' (LLM not called)", t0)
        return {"draft": draft, "trace": trace}

    if ctx.llm is None:
        draft = _retrieval_only_answer(sources, state["question"])
        trace = add_trace(state, ctx, "Reasoner", "retrieval-only mode (no LLM configured)", t0)
        return {"draft": draft, "trace": trace}

    messages = build_reasoner_messages(state["question"], sources, state.get("feedback", "") if state.get("attempt", 1) > 1 else "")
    try:
        raw = invoke_llm(ctx.llm, messages, ctx.settings)
    except LLMError as e:
        trace = add_trace(state, ctx, "Reasoner", f"LLM error: {e}", t0)
        return {"error": str(e), "trace": trace}

    draft = parse_answer(raw)
    summary = "answer: not found in sources" if draft["not_found"] else f"drafted answer ({len(draft['answer'])} chars, self-confidence {draft['self_confidence']})"
    trace = add_trace(state, ctx, "Reasoner", summary, t0, model=ctx.settings.llm_model_name)
    return {"draft": draft, "trace": trace}
