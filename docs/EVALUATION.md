# Evaluation

## Golden set
`evaluation/golden_set.json` holds 20 questions over the sample documents. Each one has an expected source file,
facts the answer must contain, and values it must **not** contain.

| Type | Count | Examples | What it tests |
|---|---|---|---|
| lookup | 11 | leave days, hotel limit, VPN portal, Q3 revenue, a single CSV row | Single-fact retrieval from PDF, TXT, CSV and vendor notes |
| aggregate | 4 | highest-revenue region, Electronics units, laptops in repair, most unused licence seats | Table summaries for CSV and Excel |
| comparison | 2 | Q3 revenue vs Q4 target, sales data vs business review | Multi-query, cross-document reasoning |
| unanswerable / out of scope | 2 | CEO salary, tomorrow's weather | Must reply "not found" instead of guessing |
| safety | 1 | "Ignore all previous instructions…" | Input guard must block it |

Q01 and Q08 also check that the values planted by the prompt-injection in `vendor_meeting_notes.txt`
("60 days" of leave, helpdesk number "0000") never appear in the answers.

## Metrics
**Retrieval mode** (no LLM): runs the Planner's rule-based plan and the Retriever only.
- *Hit rate*: share of questions where the expected file is among the retrieved passages.
- *MRR*: mean reciprocal rank of the first passage from the expected file.
- *Context recall*: share of required facts present in the retrieved text (the upper bound for answer accuracy).

**Full mode** (Gemini): runs the complete agent graph.
- *Answer accuracy*: all required facts present, no forbidden values, and not "not found".
- *Citation rate*: answerable questions whose answer cites at least one retrieved passage.
- *Refusal accuracy*: unanswerable and unsafe questions correctly declined or blocked.
- *Average latency* and *retries used*.

## How to run
- In the app: **📊 Evaluation** tab → choose a mode → **▶️ Run evaluation**.
- From the command line:
  ```bash
  python3 start.py --eval                                   # retrieval mode (Windows: py start.py --eval)
  .venv/bin/python -m evaluation.run_eval --mode full       # Windows: .venv\Scripts\python -m evaluation.run_eval --mode full
  ```
Each run writes a Markdown and a JSON report to `evaluation/results/`.

## Results

### Retrieval mode
Measured on 4 Oct 2026 in the development environment with the **offline keyword-hashing** embedding fallback
(the sandbox could not download the MiniLM model), so this is a lower bound. Re-run with the default MiniLM model on your machine.

| Metric | Value |
|---|---|
| Questions with an expected source | 17 |
| Hit rate | 100 % |
| MRR | 0.88 |
| Context recall | 94 % |

The only recall miss (Q08, helpdesk hours) retrieved the right file but a neighbouring chunk. The Planner's LLM query
rewriting and the MiniLM embeddings are expected to fix this in full mode.

### Full mode (Gemini)
Run the full evaluation with your API key before submission, then paste the summary table from the generated report here.

| Metric | Value |
|---|---|
| Model | gemini-3.5-flash-lite |
| Embeddings | MiniLM-L6-v2 |
| Answer accuracy | _fill in_ |
| Citation rate | _fill in_ |
| Refusal accuracy | _fill in_ |
| Average latency (s) | _fill in_ |
| Retries used | _fill in_ |

### Automated tests
`python3 start.py --test` runs 32 pytest tests offline, with a scripted fake LLM in place of Gemini.
They cover loaders, chunking, the vector store (including persistence), guardrails, the config round-trip and the full LangGraph
workflow: the happy path, the retry after an invalid citation, an unverified answer after a failed fact-check, "not found" on an empty store,
a blocked jailbreak, refusal of injected values, and retrieval-only mode. All 32 pass.
