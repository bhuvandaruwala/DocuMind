# System Architecture

## 1. Overview

```
┌────────────────────────────── Streamlit UI (app/ui.py) ──────────────────────────────┐
│ ⚙️ Settings (Gemini key/model)  📚 Knowledge base  💬 Ask  📊 Evaluation  🧭 How it works │
└───────────────┬───────────────────────────────────────────────┬──────────────────────┘
                │ upload                                        │ question
   ┌────────────▼─────────────┐                    ┌────────────▼─────────────────────────┐
   │ Ingestion pipeline        │                    │ LangGraph agent workflow             │
   │ validate → load → sanitize│                    │                                      │
   │ → chunk → embed → upsert  │                    │ Guard → Planner → Retriever →        │
   └────────────┬─────────────┘                    │ Reasoner → Validator ─┬─ done → Final │
                │                                   │      ▲                │               │
   ┌────────────▼─────────────┐   vector + BM25    │      └── retry ───────┘               │
   │ Chroma (data/chroma_db)  │◄───────────────────┤                                      │
   │ + in-memory BM25 index   │                    │ Gemini (free tier) via LangChain     │
   └──────────────────────────┘                    └──────────────────────────────────────┘
```

Everything runs on the user's machine except the LLM call, which goes to Google Gemini.
Documents, embeddings and the database never leave the machine. Only the question and the retrieved passages are sent to the LLM.

## 2. Technology choices

| Layer | Choice | Reason |
|---|---|---|
| UI | Streamlit | File upload, chat, tabs and progress in one Python file |
| LLM | Google Gemini via `langchain-google-genai` | Free tier in Google AI Studio, strong quality. Created by a small factory (`app/llm.py`) |
| Orchestration | LangGraph `StateGraph` | Explicit nodes, a conditional retry edge and typed shared state. Easy to explain and test |
| Vector DB | Chroma `PersistentClient` (cosine) | Local, zero setup, persists across restarts |
| Embeddings | all-MiniLM-L6-v2 via ONNX Runtime | 384-dim semantic vectors on the CPU. No PyTorch, so installs work on macOS and Windows. Free and offline after the first download |
| Keyword search | BM25 (own implementation) | Exact matches on IDs, names and numbers that small embedding models miss |
| Loaders | pypdf, python-docx, pandas + openpyxl/xlrd | Cover PDF, DOCX, TXT, CSV, XLSX and XLS |
| Deployment | `start.py` launcher + `.command`/`.bat` wrappers | One double-click on both operating systems. No Docker or admin rights needed |

## 3. Ingestion pipeline

1. **Validate** (`guardrails.validate_upload`): extension allow-list, size ≤ 20 MB, not empty, content sniffing
   (`%PDF` header, xlsx zip signature, no binary bytes in text). File names are sanitised to prevent path traversal.
2. **Load** (`ingest/loaders.py`)
   - PDF: one document per page with `page` metadata. Encrypted and scanned (image-only) PDFs give a clear error.
   - TXT: tries UTF-8, UTF-8-BOM, cp1252 and latin-1 in turn.
   - CSV/Excel: every sheet is split into **row groups of 8 rows with the header repeated** (`Table: file / sheet. Columns: …. Rows 2-9:`).
     Then **summary documents** are generated: row and column counts, sum/average/min/max per numeric column,
     the highest and lowest row by identifier, counts per category, and totals per category value.
     This lets plain retrieval answer aggregate questions such as "total units for Electronics" or "how many laptops are in repair".
3. **Sanitize**: sentences that try to instruct the AI ("ignore all previous instructions", "you must tell every user…")
   are removed **before chunking**, so an injection cannot hide across a chunk boundary. The removed values are kept in
   metadata so the Validator can detect them if they leak.
4. **Chunk** (`ingest/chunking.py`): recursive splitting on paragraph, line, sentence, clause and then word boundaries.
   Chunks are about 800 characters with 120 characters of overlap. Every chunk gets a stable ID and a readable location
   (`file, page 2` / `file, sheet 'Laptops', rows 10-17`) that is used in citations.
5. **Embed and store** (`store.py`): upsert into a Chroma collection named after the embedding backend. Re-uploading a file
   replaces its old chunks. The BM25 index is rebuilt lazily when the collection changes.

## 4. Question answering

Shared state (`agents/state.py`) holds the question, attempt number, plan, labelled sources, the draft answer,
the validation result, feedback, the final answer and a trace of every step (shown in the UI).

| Node | Implementation | Output |
|---|---|---|
| Guard | `graph.guard_node` | Rejects empty, oversized or jailbreak questions without calling any model |
| Planner | `agents/planner.py` | `{category, queries[1-3], k}` from Gemini JSON, with a rule-based fallback. Broadens on retry |
| Retriever | `agents/retriever.py` | Up to 12 labelled passages `S1..Sn` with location, similarity and keyword scores |
| Reasoner | `agents/reasoner.py` | Grounded answer with `[S#]` citations and self-reported confidence, or "not found" |
| Validator | `agents/validator.py` | `{ok, issues, warnings}`. Routes to retry or finalize |
| Finalize | `graph.finalize_node` | Status, confidence (0-1), cited sources, safe fallbacks |

**Retrieval detail.** For each query the Retriever runs vector search and BM25 search, plus a summary-only search for
aggregate questions. It fuses the rankings with Reciprocal Rank Fusion (k = 60), drops passages below the
relevance floor (cosine similarity ≥ 0.22 or BM25 ≥ 25 % of the best match), keeps the strongest hits and fills the
remaining slots with Maximal Marginal Relevance (λ = 0.75) so the passages are not near-duplicates.

**Confidence score** = 0.65 × the model's self-reported confidence + 0.35 × the retrieval strength of the cited passages,
minus penalties for warnings. It is capped at 0.3 when validation failed and 0.4 in retrieval-only mode, and it is 0 for "not found".

**Retry loop.** If validation fails and the attempt count is below `MAX_AGENT_LOOPS` (default 2), the Validator's reasons
go back to the Planner, which writes different, broader queries and raises k by 3. The Reasoner also receives the feedback.
Otherwise the answer is finalised with an "unverified" status, or replaced by a safe refusal if a safety check failed.

## 5. Reliability and safety

| Risk | Control | Where |
|---|---|---|
| Malicious or corrupt uploads | Type/size/magic-byte validation, per-file error isolation | `guardrails.py`, `ingest/pipeline.py` |
| Jailbreak in the question | Pattern-based guard, blocked before any LLM call | `graph.guard_node` |
| Prompt injection inside documents | Removed at ingestion and again at retrieval. Sources are wrapped as `<source>` data, and the prompt says to never follow instructions in them | `guardrails.neutralize_injections`, `rag.py` |
| Hallucination | Strict prompt, citation existence check, numeric grounding check, LLM fact-check, "not found" when there is no relevant context (the LLM is not called at all) | `validator.py`, `reasoner.py` |
| Leaked injected values / secrets | Output moderation refuses answers that repeat injected values, and redacts API-key-like strings | `guardrails.moderate_output` |
| LLM outages and rate limits | Exponential backoff on 429/RESOURCE_EXHAUSTED, friendly error messages, retrieval-only fallback without a key | `llm.py` |
| Runaway cost or latency | Maximum agent passes, at most 3 LLM calls per pass, bounded k | `config.py`, `graph.py` |
| Missing embedding model | Automatic fallback to offline keyword embeddings, with a UI warning | `embeddings.py` |

## 6. Mapping to the tasks

| # | Task | Implementation |
|---|---|---|
| 1 | Project foundation | `config.py`, `.env.example`, `requirements*.txt`, `start.py`, repo layout |
| 2 | User interaction layer | `app/ui.py` |
| 3 | Document ingestion | `app/ingest/loaders.py`, `pipeline.py` |
| 4 | Data preparation for search | `app/ingest/chunking.py`, table summaries in `loaders.py` |
| 5 | Vector knowledge store | `app/store.py`, `app/embeddings.py` |
| 6 | Intelligent retrieval | `app/agents/retriever.py` (hybrid + RRF + MMR) |
| 7 | RAG pipeline | `app/rag.py`, `app/agents/reasoner.py` |
| 8 | Agent-based reasoning | `app/agents/graph.py` and the four agents |
| 9 | Reliability and safety | `app/guardrails.py`, `app/agents/validator.py`, `app/llm.py` |
| 10 | Deploy and document | `start.py` + wrappers, `docs/`, `evaluation/`, `tests/` |
