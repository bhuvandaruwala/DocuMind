"""
Golden-set evaluation.

Two modes:
  retrieval - no LLM needed. Measures whether the Retriever brings back the right
              file (hit rate, MRR) and whether the expected facts are present in the
              retrieved passages (context recall).
  full      - runs the whole agent workflow with the configured LLM (Gemini by default)
              and scores answer correctness, citation validity, correct "not found"
              replies and blocked prompt-injection attempts.

Usage (from the project folder, with the virtual environment active):
    python -m evaluation.run_eval --mode retrieval
    python -m evaluation.run_eval --mode full          # needs a Gemini API key in .env
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.graph import answer_question  # noqa: E402
from app.agents.planner import heuristic_plan  # noqa: E402
from app.agents.retriever import retrieve  # noqa: E402
from app.agents.state import AgentContext  # noqa: E402
from app.config import EVAL_DIR, SAMPLE_DATA_DIR, load_settings  # noqa: E402

GOLDEN = EVAL_DIR / "golden_set.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_golden() -> List[Dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("’", "'").split())


def contains(haystack: str, needle: str) -> bool:
    h, n = _norm(haystack), _norm(needle)
    return n in h or n.replace(",", "") in h.replace(",", "")


def ensure_sample_data_indexed(ctx: AgentContext) -> None:
    from app.ingest.pipeline import ingest_folder

    have = set(ctx.store.list_sources().keys())
    needed = {p.name for p in SAMPLE_DATA_DIR.iterdir() if p.is_file()}
    if not needed.issubset(have):
        ingest_folder(SAMPLE_DATA_DIR, ctx.store, ctx.settings)


# retrieval
def eval_retrieval(ctx: AgentContext, progress: Optional[Callable] = None) -> Dict:
    rows = []
    items = [g for g in load_golden() if g.get("expected_source")]
    for i, g in enumerate(items, 1):
        plan = heuristic_plan(g["question"], ctx.settings.top_k)
        res = retrieve(ctx, plan["queries"], plan["k"], plan["category"])
        srcs = res["sources"]
        files = [s["source"] for s in srcs]
        rank = next((r for r, f in enumerate(files, 1) if f == g["expected_source"]), None)
        context = " ".join(s["text"] for s in srcs)
        facts = g.get("must_include", [])
        found = [f for f in facts if contains(context, f)]
        rows.append({
            "id": g["id"], "type": g["type"], "question": g["question"],
            "hit": rank is not None, "rank": rank, "rr": (1 / rank) if rank else 0.0,
            "context_recall": len(found) / len(facts) if facts else 1.0,
            "retrieved_files": sorted(set(files)),
        })
        if progress:
            progress(i, len(items), g["id"])
    n = len(rows) or 1
    metrics = {
        "questions": len(rows),
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
        "context_recall": round(sum(r["context_recall"] for r in rows) / n, 3),
        "embedding_backend": ctx.store.embedder.name,
    }
    return {"mode": "retrieval", "metrics": metrics, "rows": rows}


# full
def score_answer(g: Dict, final: Dict) -> Dict:
    answer = final.get("answer", "")
    status = final.get("status")
    if g.get("expect_blocked"):
        correct = status == "blocked"
        kind = "blocked" if correct else "not blocked"
    elif g.get("expect_not_found"):
        correct = status in ("not_found", "blocked", "refused")
        kind = "correctly declined" if correct else "answered an unanswerable question"
    else:
        must = g.get("must_include", [])
        must_not = g.get("must_not_include", [])
        ok_in = all(contains(answer, m) for m in must)
        ok_out = not any(contains(answer, m) for m in must_not)
        correct = ok_in and ok_out and status not in ("not_found", "error", "blocked", "refused")
        kind = "correct" if correct else ("missing facts" if not ok_in else "contains forbidden value")
    cited_ok = bool(final.get("citations")) if not (g.get("expect_not_found") or g.get("expect_blocked")) else True
    return {"correct": correct, "verdict": kind, "citations_ok": cited_ok}


def eval_full(ctx: AgentContext, progress: Optional[Callable] = None, pause_s: float = 0.0) -> Dict:
    rows = []
    items = load_golden()
    for i, g in enumerate(items, 1):
        t0 = time.time()
        out = answer_question(g["question"], ctx)
        final = out["final"]
        sc = score_answer(g, final)
        rows.append({
            "id": g["id"], "type": g["type"], "question": g["question"],
            "answer": final.get("answer", ""), "status": final.get("status"),
            "confidence": final.get("confidence"), "attempts": final.get("attempts", 1),
            "latency_s": round(time.time() - t0, 2), **sc,
            "cited_files": sorted({c["source"] for c in final.get("citations", [])}),
        })
        if progress:
            progress(i, len(items), g["id"])
        if pause_s:
            time.sleep(pause_s)  # stay under free-tier requests-per-minute limits
    n = len(rows) or 1
    answerable = [r for r, g in zip(rows, items) if not (g.get("expect_not_found") or g.get("expect_blocked"))]
    declines = [r for r, g in zip(rows, items) if g.get("expect_not_found") or g.get("expect_blocked")]
    metrics = {
        "questions": len(rows),
        "overall_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "answer_accuracy": round(sum(r["correct"] for r in answerable) / (len(answerable) or 1), 3),
        "citation_rate": round(sum(r["citations_ok"] for r in answerable) / (len(answerable) or 1), 3),
        "refusal_accuracy": round(sum(r["correct"] for r in declines) / (len(declines) or 1), 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "retries_used": sum(1 for r in rows if (r["attempts"] or 1) > 1),
        "model": f"gemini/{ctx.settings.llm_model_name}",
        "embedding_backend": ctx.store.embedder.name,
    }
    return {"mode": "full", "metrics": metrics, "rows": rows}


def save_report(result: Dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = RESULTS_DIR / f"{result['mode']}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Evaluation ({result['mode']}) - {stamp}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in result["metrics"].items()]
    lines += ["", "| ID | Type | Result | Detail |", "|---|---|---|---|"]
    for r in result["rows"]:
        if result["mode"] == "retrieval":
            lines.append(f"| {r['id']} | {r['type']} | {'hit' if r['hit'] else 'MISS'} (rank {r['rank']}) | recall {r['context_recall']:.2f} |")
        else:
            ans = r["answer"].replace("\n", " ").replace("|", "/")[:140]
            lines.append(f"| {r['id']} | {r['type']} | {'✅' if r['correct'] else '❌'} {r['verdict']} | {ans} |")
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return base.with_suffix(".md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between questions in full mode")
    args = ap.parse_args()

    from app.llm import LLMError, get_llm
    from app.store import get_store

    settings = load_settings()
    store = get_store(settings.embedding_backend)
    llm = None
    if args.mode == "full":
        try:
            llm = get_llm(settings)
        except LLMError as e:
            print(f"Cannot run full evaluation: {e}")
            return 1
    ctx = AgentContext(settings=settings, store=store, llm=llm)
    ensure_sample_data_indexed(ctx)

    def prog(i, n, qid):
        print(f"  [{i}/{n}] {qid}", flush=True)

    result = eval_retrieval(ctx, prog) if args.mode == "retrieval" else eval_full(ctx, prog, args.pause)
    path = save_report(result)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
