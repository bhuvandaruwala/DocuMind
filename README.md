# DocuMind

**Your documents. Your knowledge. Your answers.**

DocuMind is an AI-powered document knowledge assistant. It reads your documents, builds a searchable
knowledge base from them, and lets you ask questions in natural language.

Instead of searching through several PDFs, spreadsheets or text files by hand, you add your documents to
DocuMind and ask about the information inside them. It retrieves the most relevant sections and gives them to
an LLM as context, so the answer stays grounded in your documents and not only in the model's general knowledge.
Every answer comes with citations, a confidence score and a validation status.

In short: **DocuMind reads your documents, keeps their knowledge in mind, and lets you have a conversation with your information.**

Supported files: PDF, Word (DOCX), plain text (TXT), CSV and Excel (XLSX/XLS).

### Example

Add an employee handbook (PDF), a laptop inventory (Excel) and an IT helpdesk FAQ (TXT), then ask:

> How many days of paid annual leave do employees get?

> How many laptops are currently in repair?

> How do I reset my password?

DocuMind searches the knowledge base, finds the relevant passages and answers from them. If the documents
don't contain the answer, it says so.

Under the hood, a team of AI agents (Planner, Retriever, Reasoner and Validator, built with LangGraph) finds
the passages in a local Chroma vector database and writes the answer with Google Gemini.

It runs **locally on macOS and Windows with one command**. No Docker, no paid API: Gemini has a free tier.

---

## Quick start (about 5 minutes)

**1. Install Python 3.11 or 3.12** (3.10-3.13 work) from <https://www.python.org/downloads/>.
On Windows, tick **"Add python.exe to PATH"** in the installer.

**2. Start the app.** Clone the Repo, then:

| macOS | Windows |
|---|---|
| Double-click **`start_mac.command`**. If macOS blocks it, right-click it, choose **Open**, or run `python3 start.py` in Terminal from the project folder | Double-click **`start_windows.bat`**, or run `py start.py` in Command Prompt from the project folder |

The first run creates a private virtual environment (`.venv`) and installs the dependencies (2-5 minutes).
The app then opens in your browser at <http://localhost:8501>. Later runs start in a few seconds.

**3. Add your free Gemini API key.**
Get a key at <https://aistudio.google.com/app/apikey>. In the app's sidebar, open **⚙️ Settings**,
paste the key, click **🔌 Test**, then **💾 Save**. The key is saved only to the local `.env` file.

**4. Load documents and ask questions.**
Upload your own or Click **🧪 Samples** to load the 6 demo files, then click an example question or type one.

> Without a key the app still works in **retrieval-only mode**. It shows the most relevant passages instead of a written answer.

---

## What's inside

| Requirement | How it is met |
|---|---|
| Upload PDF, Word, TXT, CSV, Excel | `app/ingest/loaders.py`: page-level PDF text, Word paragraphs and tables, encoding-safe TXT, CSV/Excel row groups with header repeated, plus **pre-computed table summaries** (totals, counts, max/min) |
| Semantic search | Chunking (~800 chars, 120 overlap), local **all-MiniLM-L6-v2** embeddings (ONNX, CPU), persistent **Chroma** DB |
| Intelligent retrieval | Hybrid **vector + BM25** search, Reciprocal Rank Fusion, relevance floor, **MMR** diversity |
| RAG answers | Gemini answers **only from retrieved sources**, cites `[S1]`, says *not found* when unsure |
| Agents | **LangGraph**: Guard → Planner → Retriever → Reasoner → Validator → (retry) → Finalize |
| Reliability & safety | File/question validation, prompt-injection removal, citation and number grounding checks, LLM fact-check, output moderation, rate-limit backoff, friendly errors |
| Evaluation | 20-question golden set (`evaluation/`), retrieval and full-pipeline metrics, in the app's **📊 Evaluation** tab |

## Sample documents (`sample_data/`)

These are fictional documents for "Nimbus Retail Pvt. Ltd.". The facts agree across files, so cross-document questions have a single correct answer.

| File | Type | Content |
|---|---|---|
| `Nimbus_Employee_Handbook_2026.pdf` | PDF, 3 pages | Leave, hybrid work, travel and expense, security policy |
| `Nimbus_Q3_2026_Business_Review.pdf` | PDF | Q3 revenue, region table, Q4 targets |
| `nimbus_sales_q3_2026.csv` | CSV, 60 rows | Sales by region, month and category |
| `nimbus_it_asset_inventory.xlsx` | Excel, 2 sheets | Laptops, software licences |
| `nimbus_it_helpdesk_faq.txt` | Text | VPN, Wi-Fi, password reset, helpdesk hours |
| `vendor_meeting_notes.txt` | Text | Vendor contract notes, plus a **planted prompt-injection** to test the guardrails |

Try these questions: *How many days of paid annual leave do employees get?* · *Which region had the highest total revenue in Q3?* ·
*How many laptops are currently in repair?* · *Compare the Q3 2026 revenue with the Q4 target.* · *What is the CEO's annual salary?* (should be "not found")

## Commands

```bash
python3 start.py            # macOS: set up if needed and run the app   (Windows: py start.py)
python3 start.py --test     # run the 32 automated tests (offline, no API key needed)
python3 start.py --eval     # retrieval evaluation on the golden set
python3 start.py --package    # build dist/capstone.zip for submission (no .venv, data or .env)
python3 start.py --port 8502 --reinstall
```

Full-pipeline evaluation with Gemini: use the **📊 Evaluation** tab in the app, or run
`.venv/bin/python -m evaluation.run_eval --mode full` (Windows: `.venv\Scripts\python -m evaluation.run_eval --mode full`).

## Project structure

```
app/
  ui.py                 Streamlit UI (Settings, Knowledge base, Ask, Evaluation, How it works)
  config.py             Settings from .env and the UI; saved to .env
  llm.py                LLM factory (Google Gemini) + retry/backoff
  embeddings.py         MiniLM ONNX embeddings, with an offline hashing fallback
  store.py              Chroma vector store + BM25 keyword index + MMR
  rag.py                Prompt templates and answer parsing
  guardrails.py         Validation, prompt-injection, grounding, moderation
  ingest/               loaders.py · chunking.py · pipeline.py
  agents/               state.py · planner.py · retriever.py · reasoner.py · validator.py · graph.py
evaluation/             golden_set.json · run_eval.py
sample_data/            6 demo documents (generated by scripts/generate_sample_data.py)
tests/                  pytest suite with a scripted fake LLM
docs/                   SETUP · ARCHITECTURE · AGENT_ROLES · EVALUATION · LIMITATIONS · Capstone_Documentation.docx
start.py                one-command launcher; start_mac.command / start_windows.bat wrappers
```

## Documentation

- [docs/SETUP.md](docs/SETUP.md): installation, configuration, troubleshooting
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): system design and data flow
- [docs/AGENT_ROLES.md](docs/AGENT_ROLES.md): what each agent does
- [docs/EVALUATION.md](docs/EVALUATION.md): golden set and metrics
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md): limitations, challenges faced, future work

## Privacy

Documents, embeddings and the vector database stay on your computer (`data/`). Only the question and the
retrieved passages are sent to the Gemini API when an answer is generated. The API key lives only in `.env`,
which is excluded from version control.
