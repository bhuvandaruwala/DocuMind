"""
Validator agent: decides whether the drafted answer can be shown.

Deterministic checks (always run, no LLM cost):
  * citations exist and point to real retrieved passages
  * numbers in the answer appear in the retrieved passages (numeric grounding)
  * output moderation: unsafe content, leaked secrets, values that only appeared
    in a removed prompt-injection
Optional LLM check (Settings → "LLM fact-check"): a second model call judges
whether every claim is supported by the cited passages.

If a check fails and passes remain, the Validator sends feedback to the Planner
(retry with broader retrieval). Otherwise it finalises: grounding failures are
shown with an "unverified" warning and low confidence; safety failures are
replaced with a safe refusal.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List

from app.agents.state import AgentContext, AgentState, add_trace
from app.guardrails import check_citations, check_numbers, moderate_output
from app.llm import LLMError, invoke_llm
from app.rag import VALIDATOR_PROMPT, format_sources, parse_json_object


def _injection_terms(sources: List[Dict]) -> List[str]:
    """Values from removed injections that do NOT legitimately appear in any clean passage."""
    terms = set()
    for s in sources:
        for t in (s.get("injection_terms") or "").split(","):
            if t.strip():
                terms.add(t.strip())
    clean_text = " ".join(s["text"] for s in sources)
    return [t for t in terms if not re.search(rf"(?<![\d.,]){re.escape(t)}(?![\d])", clean_text)]


def validate_node(state: AgentState, ctx: AgentContext) -> AgentState:
    t0 = time.time()
    draft = state.get("draft") or {}
    sources = state.get("sources", [])
    answer = draft.get("answer", "")
    issues: List[str] = []
    warnings: List[str] = []
    safety_failure = False
    checks: Dict[str, str] = {}

    if state.get("error"):
        v = {"ok": False, "issues": [state["error"]], "warnings": [], "safety_failure": False, "fatal": True}
        trace = add_trace(state, ctx, "Validator", "pipeline error → stop", t0)
        return {"validation": v, "trace": trace}

    if draft.get("not_found"):
        checks["not_found"] = "answer says the information is not in the documents"
        if state.get("candidates_found", 0) > 0 and state.get("attempt", 1) < ctx.settings.max_agent_loops:
            issues.append("the answer was 'not found' although related passages exist; search more broadly")
    else:
        labels = [s["label"] for s in sources]
        valid, invalid = check_citations(answer, labels)
        checks["citations"] = f"{len(valid)} valid, {len(invalid)} invalid"
        if invalid:
            issues.append(f"cites passages that do not exist: {', '.join(invalid)}")
        if not valid:
            issues.append("the answer has no [S#] citations")

        if not draft.get("retrieval_only"):
            grounded, ungrounded = check_numbers(answer, [s["text"] for s in sources])
            checks["numbers"] = f"{len(grounded)} grounded, {len(ungrounded)} not found in sources"
            if ungrounded:
                msg = f"numbers not found in the sources: {', '.join(ungrounded[:6])}"
                if len(ungrounded) > len(grounded):
                    issues.append(msg)
                else:
                    warnings.append(msg + " (may be calculated)")

        ok_mod, reason, redacted = moderate_output(answer, _injection_terms(sources))
        checks["moderation"] = "passed" if ok_mod else reason
        if redacted != answer:
            draft = {**draft, "answer": redacted}
            warnings.append("a secret-looking string was redacted")
        if not ok_mod:
            issues.append(reason)
            safety_failure = True

        # Optional LLM fact-check, only if the cheap checks passed
        if not issues and ctx.llm is not None and ctx.settings.llm_validation and not draft.get("retrieval_only"):
            prompt = VALIDATOR_PROMPT.format(question=state["question"], sources=format_sources(sources), answer=answer)
            try:
                verdict = parse_json_object(invoke_llm(ctx.llm, prompt, ctx.settings, retries=2)) or {}
                supported = verdict.get("supported", True) is not False
                addresses = verdict.get("addresses_question", True) is not False
                checks["llm_fact_check"] = "supported" if supported and addresses else f"failed: {verdict.get('issues', '')}"
                if not supported:
                    issues.append(f"fact-check: {verdict.get('issues') or 'claims not supported by sources'}")
                elif not addresses:
                    warnings.append(f"fact-check: answer may not fully address the question ({verdict.get('issues', '')})")
            except LLMError as e:
                checks["llm_fact_check"] = f"skipped ({e})"
                warnings.append("LLM fact-check skipped (LLM unavailable)")

    ok = not issues
    v = {"ok": ok, "issues": issues, "warnings": warnings, "safety_failure": safety_failure, "checks": checks}
    summary = "passed ✔" if ok else f"failed: {'; '.join(issues)}"
    trace = add_trace(state, ctx, "Validator", summary, t0, checks=checks, warnings=warnings)
    out: AgentState = {"validation": v, "trace": trace, "draft": draft}
    if not ok:
        out["feedback"] = "; ".join(issues)
    return out


def route_after_validation(state: AgentState, max_loops: int) -> str:
    v = state.get("validation") or {}
    if v.get("ok") or v.get("fatal"):
        return "done"
    if state.get("attempt", 1) < max_loops:
        return "retry"
    return "done"
