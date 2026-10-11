"""
Streamlit user interface.

Run with:  streamlit run app/ui.py      (the start scripts do this for you)

Sidebar : ⚙️ Settings (Gemini API key, model, test connection)
          📚 Knowledge base (upload / sample documents / indexed files)
Main    : 💬 Ask questions · 📊 Evaluation · 🧭 How it works
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.agents.graph import answer_question  # noqa: E402
from app.agents.state import AgentContext  # noqa: E402
from app.config import (  # noqa: E402
    GEMINI_MODELS,
    SAMPLE_DATA_DIR,
    SUPPORTED_EXTENSIONS,
    load_settings,
    save_settings,
)
from app.embeddings import embedding_warning  # noqa: E402
from app.ingest.pipeline import ingest_folder, ingest_uploads  # noqa: E402
from app.llm import LLMError, get_llm, test_connection  # noqa: E402
from app.store import get_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

STATUS_BADGES = {
    "verified": ("✅ Verified", "green"),
    "unverified": ("⚠️ Could not fully verify", "orange"),
    "not_found": ("🔍 Not found in documents", "gray"),
    "blocked": ("⛔ Blocked by input guard", "red"),
    "refused": ("🛡️ Refused by safety check", "red"),
    "retrieval_only": ("📄 Retrieval only (no LLM)", "blue"),
    "error": ("❌ Error", "red"),
}

EXAMPLE_QUESTIONS = [
    "How many days of paid annual leave do employees get?",
    "Which region had the highest total revenue in Q3?",
    "How many laptops are currently in repair?",
    "Compare the Q3 2026 revenue with the Q4 target.",
    "When does the BrightLogix contract renew?",
    "What is the CEO's annual salary?",
]


# state helpers
def settings():
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    return st.session_state.settings


@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~80 MB)…")
def cached_store(backend: str):
    return get_store(backend)


def store():
    return cached_store(settings().embedding_backend)


def make_context(on_step=None) -> AgentContext:
    s = settings()
    llm = None
    if s.llm_ready:
        try:
            llm = get_llm(s)
        except LLMError as e:
            st.warning(str(e))
    return AgentContext(settings=s, store=store(), llm=llm, on_step=on_step)


# sidebar: settings
# Fixed pipeline options (no longer user-editable)
FIXED_OPTIONS = dict(top_k=6, max_agent_loops=2, llm_validation=True, embedding_backend="local")

# hide Streamlit's built-in "Press Enter to submit form" hint inside the key field
_HIDE_FORM_HINT = "<style>[data-testid='InputInstructions']{display:none}</style>"


def settings_panel():
    s = settings()
    with st.sidebar.expander("⚙️ Settings", expanded=not s.llm_ready):
        st.markdown(_HIDE_FORM_HINT, unsafe_allow_html=True)
        # A form commits the key and the button click together, so Enter or one click on Save is enough.
        with st.form("gemini_key_form", border=False):
            key = st.text_input(
                "Gemini API key", value=s.gemini_api_key, type="password",
                help="Create a free key at https://aistudio.google.com/app/apikey — it is stored only in the local .env file.",
            )
            st.caption("No key yet? Get one free at [Google AI Studio](https://aistudio.google.com/app/apikey).")
            c1, c2 = st.columns(2)
            save_clicked = c1.form_submit_button("💾 Save", type="primary", width="stretch")
            test_clicked = c2.form_submit_button("🔌 Test", width="stretch")
        options = GEMINI_MODELS + ["Custom…"]
        idx = options.index(s.gemini_model) if s.gemini_model in GEMINI_MODELS else len(options) - 1
        choice = st.selectbox("Gemini model", options, index=idx,
                              help="Flash-Lite models have the most generous free-tier limits.")
        model = st.text_input("Custom model ID", value=s.gemini_model) if choice == "Custom…" else choice
        new = s.copy(gemini_api_key=key.strip(), gemini_model=model.strip(), **FIXED_OPTIONS)

        if save_clicked:
            st.session_state.settings = new
            path = save_settings(new)
            st.success(f"Saved to {path.name}")
            st.rerun()
        if test_clicked:
            with st.spinner("Calling the model…"):
                ok, msg = test_connection(new)
            (st.success if ok else st.error)(msg)
        # apply unsaved edits for this session too
        st.session_state.settings = new

    s = settings()
    if s.llm_ready:
        st.sidebar.success(f"LLM: Gemini · {s.gemini_model} · key {s.masked_key()}", icon="🤖")
    else:
        st.sidebar.warning("No Gemini API key yet → retrieval-only mode. Open ⚙️ Settings.", icon="🔑")


# sidebar: knowledge base
def knowledge_base_panel():
    s = settings()
    st.sidebar.subheader("📚 Knowledge base")
    exts = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS]
    files = st.sidebar.file_uploader(
        f"Upload documents (PDF, Word, TXT, CSV, Excel · max {s.max_file_mb} MB each)",
        type=exts, accept_multiple_files=True, key=f"uploader_{st.session_state.get('uploader_n', 0)}",
    )
    c1, c2 = st.sidebar.columns(2)
    if c1.button("📥 Ingest", type="primary", disabled=not files, width="stretch"):
        with st.sidebar.status("Ingesting…", expanded=True) as status:
            results = ingest_uploads(((f.name, f.getvalue()) for f in files), store(), s)
            _show_ingest_results(results, status)
        st.session_state.uploader_n = st.session_state.get("uploader_n", 0) + 1
        st.rerun()
    if c2.button("🧪 Samples", width="stretch", help="Load the 6 demo files from sample_data/"):
        with st.sidebar.status("Loading sample documents…", expanded=True) as status:
            results = ingest_folder(SAMPLE_DATA_DIR, store(), s)
            _show_ingest_results(results, status)
        st.rerun()

    if st.session_state.get("ingest_log"):
        with st.sidebar.expander("Last ingestion report", expanded=False):
            for line in st.session_state.ingest_log:
                st.write(line)

    sources = store().list_sources()
    if not sources:
        st.sidebar.caption("No documents indexed yet.")
        return
    st.sidebar.caption(f"{len(sources)} file(s) · {sum(v['chunks'] for v in sources.values())} chunks indexed")
    for name, info in sources.items():
        a, b = st.sidebar.columns([5, 1])
        a.markdown(f"<small>📄 {name} · {info['chunks']} chunks</small>", unsafe_allow_html=True)
        if b.button("🗑", key=f"del_{name}", help=f"Remove {name}"):
            store().delete_source(name)
            st.rerun()
    if st.sidebar.button("Clear knowledge base", width="stretch"):
        store().reset()
        st.rerun()
    warn = embedding_warning()
    if warn:
        st.sidebar.warning(warn, icon="⚠️")


def _show_ingest_results(results, status):
    log = []
    ok = sum(r.ok for r in results)
    for r in results:
        line = f"{'✅' if r.ok else '❌'} {r.file}: {r.message}" + (f" ({r.seconds}s)" if r.ok else "")
        st.write(line)
        log.append(line)
    st.session_state.ingest_log = log
    status.update(label=f"Ingested {ok}/{len(results)} file(s)", state="complete" if ok == len(results) else "error")


# chat
def render_answer(result: dict):
    final = result["final"]
    label, color = STATUS_BADGES.get(final.get("status"), ("?", "gray"))
    conf = final.get("confidence")
    meta = f":{color}-badge[{label}]"
    if conf is not None:
        meta += f" &nbsp; Confidence **{int(conf * 100)}%**"
    if final.get("attempts", 1) > 1:
        meta += f" &nbsp; · {final['attempts']} agent passes"
    if final.get("elapsed_s") is not None:
        meta += f" &nbsp; · {final['elapsed_s']}s"
    # the answer stands alone; everything about how it was produced sits below a divider
    st.markdown(final.get("answer", ""))
    st.divider()
    st.caption("ANSWER DETAILS")
    st.markdown(meta, unsafe_allow_html=True)
    for w in final.get("issues", []):
        st.caption(f"⚠️ Validator: {w}")
    for w in final.get("warnings", []):
        st.caption(f"ℹ️ {w}")

    cites = final.get("citations") or []
    if cites:
        with st.expander(f"📎 Sources cited ({len(cites)})"):
            for c in cites:
                st.markdown(f"**[{c['label']}] {c['location']}** · similarity {c['similarity']} · keyword {c['keyword_score']}"
                            + (" · 🛡️ injection text removed" if c.get("injection_removed") else ""))
                st.text(c["text"][:1500])
    others = [s for s in result.get("sources", []) if s not in cites]
    if others:
        with st.expander(f"🔎 Other retrieved passages ({len(others)})"):
            for c in others:
                st.markdown(f"**[{c['label']}] {c['location']}** · similarity {c['similarity']}")
                st.text(c["text"][:600])
    with st.expander("🧠 Agent trace"):
        for t in result.get("trace", []):
            st.markdown(f"**{t['agent']}** (pass {t['attempt']}, {t['ms']} ms) — {t['summary']}")
            details = {k: v for k, v in (t.get("details") or {}).items() if v}
            if details:
                st.json(details, expanded=False)


def ask_tab():
    if "history" not in st.session_state:
        st.session_state.history = []
    kb_empty = store().count() == 0
    if kb_empty:
        st.info("👈 Upload documents or click **🧪 Samples** in the sidebar to start.", icon="📚")

    top = st.columns([6, 1])
    if top[1].button("Clear chat", disabled=not st.session_state.history, width="stretch"):
        st.session_state.history = []
        st.rerun()

    conversation = st.container()   # messages render here, so the input box stays below them
    with conversation:
        for turn in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                render_answer(turn["result"])

        pending = None
        if not st.session_state.history and not kb_empty:
            st.caption("Try an example:")
            cols = st.columns(3)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                if cols[i % 3].button(q, key=f"ex_{i}", width="stretch"):
                    pending = q

    typed = st.chat_input("Ask a question about your documents…", disabled=kb_empty,
                          max_chars=settings().max_question_chars)
    question = typed or pending
    if not question:
        return

    with conversation:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Agents working…", expanded=True) as status:
                def on_step(entry):
                    status.write(f"**{entry['agent']}** (pass {entry['attempt']}): {entry['summary']}")
                    status.update(label=f"{entry['agent']}…")
                try:
                    result = answer_question(question, make_context(on_step))
                except Exception as e:  # noqa: BLE001 - never show a raw stack trace to the user
                    logging.exception("Pipeline failure")
                    result = {"final": {"answer": f"Something went wrong: {e}", "status": "error"}, "trace": [], "sources": []}
                status.update(label="Done", state="complete", expanded=False)
            render_answer(result)
    st.session_state.history.append({"question": question, "result": result})
    if pending:
        st.rerun()  # hide the example buttons once the conversation has started


# evaluation tab
def evaluation_tab():
    from evaluation.run_eval import ensure_sample_data_indexed, eval_full, eval_retrieval, load_golden, save_report

    st.markdown(
        "Runs the **golden set** of "
        f"{len(load_golden())} questions over the sample documents (lookups, table aggregates, cross-document "
        "comparisons, unanswerable questions and a prompt-injection attempt)."
    )
    mode = st.radio(
        "Mode", ["retrieval", "full"], horizontal=True,
        format_func=lambda m: "Retrieval only (fast, no LLM)" if m == "retrieval" else "Full agent pipeline (uses Gemini)",
    )
    if mode == "full":
        st.caption("Full mode makes about 3 LLM calls per question (~60 calls) and pauses between questions "
                   "to respect free-tier rate limits. It takes a few minutes.")
        if not settings().llm_ready:
            st.warning("Add a Gemini API key in ⚙️ Settings first.")
            return
    if not st.button("▶️ Run evaluation", type="primary"):
        if "eval_result" in st.session_state:
            _render_eval(st.session_state.eval_result)
        return

    ctx = make_context()
    with st.spinner("Making sure the sample documents are indexed…"):
        ensure_sample_data_indexed(ctx)
    bar = st.progress(0.0, text="Starting…")

    def prog(i, n, qid):
        bar.progress(i / n, text=f"{qid} ({i}/{n})")

    result = eval_retrieval(ctx, prog) if mode == "retrieval" else eval_full(ctx, prog, pause_s=4.0)
    result["report"] = str(save_report(result))
    st.session_state.eval_result = result
    _render_eval(result)


def _render_eval(result):
    m = result["metrics"]
    keys = [k for k, v in m.items() if isinstance(v, (int, float))]
    cols = st.columns(min(len(keys), 6))
    for c, k in zip(cols, keys[:6]):
        v = m[k]
        c.metric(k.replace("_", " ").title(), f"{v:.0%}" if isinstance(v, float) and v <= 1 and "latency" not in k else v)
    st.caption(" · ".join(f"{k}: {v}" for k, v in m.items() if k not in keys))
    df = pd.DataFrame(result["rows"])
    st.dataframe(df, hide_index=True, width="stretch")
    if result.get("report"):
        st.caption(f"Report saved to `{Path(result['report']).relative_to(ROOT)}`")


# how it works
def how_tab():
    st.markdown(
        """
**Pipeline.** Documents are validated, loaded (PDF pages, text, CSV/Excel rows + pre-computed table summaries),
cleaned of prompt-injection text, split into ~800-character chunks, embedded with a local MiniLM model and stored
in a persistent **Chroma** vector database. Each question then flows through a **LangGraph** team of agents:
"""
    )
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR; graph [nodesep=0.35, ranksep=0.45]; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica", fontsize=16]; edge [fontsize=13];
  Q [label="Question", shape=oval, fillcolor="#ffffff"];
  Guard [label="Input guard\\n(length, jailbreak)"];
  Planner [label="Planner\\nclassify + rewrite\\ninto 1-3 queries"];
  Retriever [label="Retriever\\nvector + BM25 → RRF\\nrelevance floor + MMR"];
  Reasoner [label="Reasoner (Gemini)\\nanswer only from\\nsources, cite [S#]"];
  Validator [label="Validator\\ncitations · numbers ·\\nsafety · LLM fact-check"];
  A [label="Answer + citations\\n+ confidence", shape=oval, fillcolor="#e9f9ee"];
  DB [label="Chroma\\nvector store", shape=cylinder, fillcolor="#fff7e6"];
  Q -> Guard -> Planner -> Retriever -> Reasoner -> Validator -> A [label=""];
  Retriever -> DB [dir=both, style=dashed];
  Validator -> Planner [label="retry with feedback\\n(max passes)", style=dashed, color="#d9534f", fontcolor="#d9534f"];
  Guard -> A [label="blocked", style=dotted];
}
"""
    )
    st.markdown(
        """
| Agent | Responsibility |
|---|---|
| **Planner** | Classifies the question (lookup, comparison, aggregate, summary, out-of-scope) and writes 1-3 focused search queries; broadens them on retry. |
| **Retriever** | Hybrid search (semantic vectors + BM25 keywords, fused with Reciprocal Rank Fusion), table summaries for numeric questions, relevance floor, MMR for diversity, citation labels. |
| **Reasoner** | Gemini answers strictly from the labelled passages, cites `[S#]`, says *not found* when the context is insufficient. |
| **Validator** | Checks citations exist, numbers appear in sources, no unsafe/leaked/injected content, optional LLM fact-check. Fails → feedback to the Planner for one more pass. |
"""
    )
    with st.expander("Current settings (secrets masked)"):
        st.json(settings().as_dict())


# main
def main():
    st.title("🧠 DocuMind")
    st.caption("DocuMind reads your documents, keeps them in mind, and answers your questions "
               "from that knowledge base, with citations.")
    settings_panel()
    knowledge_base_panel()
    tab_ask, tab_eval, tab_how = st.tabs(["💬 Ask", "📊 Evaluation", "🧭 How it works"])
    with tab_ask:
        ask_tab()
    with tab_eval:
        evaluation_tab()
    with tab_how:
        how_tab()


main()
