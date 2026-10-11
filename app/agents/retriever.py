"""
Retriever agent: hybrid search, fusion and MMR.

For every query from the Planner:
  1. vector (semantic) search in Chroma
  2. BM25 keyword search
  3. Reciprocal Rank Fusion merges both rankings (robust to different score scales)
For aggregate questions, pre-computed table summaries are searched as well.
Then:
  4. a relevance floor drops passages that neither search considers relevant
  5. Maximal Marginal Relevance picks a relevant *and diverse* final set
     (avoids five near-identical chunks from the same page)
  6. passages get citation labels S1..Sn and their source metadata
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from app.agents.state import AgentContext, AgentState, add_trace
from app.guardrails import neutralize_injections
from app.store import mmr

RRF_K = 60
# minimum cosine similarity for a passage to count as relevant by meaning (per embedding backend)
SIM_FLOOR = {"local": 0.22, "hash": 0.06}
BM25_FLOOR = 0.25   # fraction of the best keyword score


def retrieve(ctx: AgentContext, queries: List[str], k: int, category: str) -> Dict:
    store = ctx.store
    fused: Dict[str, Dict] = {}

    def add(hits: List[Dict], weight: float = 1.0):
        for rank, h in enumerate(hits):
            e = fused.setdefault(h["id"], {**h, "rrf": 0.0, "similarity": h.get("similarity", 0.0), "bm25_norm": 0.0})
            e["rrf"] += weight / (RRF_K + rank + 1)
            e["similarity"] = max(e.get("similarity", 0.0), h.get("similarity", 0.0))
            e["bm25_norm"] = max(e.get("bm25_norm", 0.0), h.get("bm25_norm", 0.0))
            if "embedding" in h and "embedding" not in e:
                e["embedding"] = h["embedding"]

    pool = max(k * 3, 12)
    # keyword search is more trustworthy than the offline hashing embeddings, so weight it up there
    kw_weight = 1.5 if store.embedder.name == "hash" else 1.0
    for q in queries:
        add(store.vector_search(q, pool))
        add(store.keyword_search(q, pool), weight=kw_weight)
        if category == "aggregate":
            add(store.vector_search(q, 3, where={"kind": "table_summary"}))

    candidates = list(fused.values())
    floor = SIM_FLOOR.get(store.embedder.name, 0.1)
    relevant = [c for c in candidates if c["similarity"] >= floor or c["bm25_norm"] >= BM25_FLOOR]
    relevant.sort(key=lambda c: c["rrf"], reverse=True)
    relevant = relevant[: max(k * 3, 15)]

    n_final = min(len(relevant), k + 2 * (len(queries) - 1), 12)
    if relevant and n_final:
        missing = [c for c in relevant if "embedding" not in c]
        if missing:
            vecs = store.embed_texts([c["text"] for c in missing])
            for c, v in zip(missing, vecs):
                c["embedding"] = v
        qvec = store.embed_texts([" ".join(queries)])[0]
        # MMR on fused relevance: blend semantic similarity with the fused rank score
        cand_vecs = np.stack([c["embedding"] for c in relevant])
        order = mmr(qvec, cand_vecs, len(relevant), lam=0.75)
        # the strongest fused hits are always kept; MMR fills the remaining slots with diverse passages
        chosen = relevant[: max(1, min(n_final, n_final // 2 + 1))]
        chosen_ids = {c["id"] for c in chosen}
        for i in order:
            if len(chosen) >= n_final:
                break
            if relevant[i]["id"] not in chosen_ids:
                chosen.append(relevant[i])
                chosen_ids.add(relevant[i]["id"])
        chosen.sort(key=lambda c: c["rrf"], reverse=True)
    else:
        chosen = []

    sources = []
    for i, c in enumerate(chosen, start=1):
        text, removed = neutralize_injections(c["text"])  # defence in depth (also done at ingestion)
        m = c["metadata"]
        sources.append(
            {
                "label": f"S{i}",
                "id": c["id"],
                "text": text,
                "location": m.get("location", m.get("source", "?")),
                "source": m.get("source", "?"),
                "page": m.get("page"),
                "sheet": m.get("sheet"),
                "rows": m.get("rows"),
                "kind": m.get("kind", "text"),
                "similarity": round(float(c["similarity"]), 3),
                "keyword_score": round(float(c["bm25_norm"]), 3),
                "fused_score": round(float(c["rrf"]), 4),
                "injection_removed": bool(m.get("injection_removed")) or bool(removed),
                "injection_terms": m.get("injection_terms", ""),
            }
        )
    return {"sources": sources, "candidates": len(candidates), "relevant": len(relevant)}


def retrieve_node(state: AgentState, ctx: AgentContext) -> AgentState:
    t0 = time.time()
    plan = state["plan"]
    res = retrieve(ctx, plan["queries"], plan["k"], plan["category"])
    srcs = res["sources"]
    files = sorted({s["source"] for s in srcs})
    summary = (f"{len(srcs)} passages from {len(files)} file(s) "
               f"(from {res['candidates']} candidates, {res['relevant']} above relevance floor)")
    trace = add_trace(
        state, ctx, "Retriever", summary, t0,
        passages=[f"{s['label']}: {s['location']} (sim {s['similarity']}, kw {s['keyword_score']})" for s in srcs],
        injection_flags=[s["label"] for s in srcs if s["injection_removed"]],
    )
    return {"sources": srcs, "candidates_found": res["relevant"], "trace": trace}
