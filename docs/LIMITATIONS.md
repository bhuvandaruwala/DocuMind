# Limitations, Challenges and Future Work

## Known limitations
1. **Scanned PDFs are not supported.** Only PDFs with a text layer can be read. Image-only pages produce a clear error. OCR (for example Tesseract) would add a large native dependency for the examiner.
2. **Table reasoning is summary-based.** Aggregates are pre-computed per column and per category at ingestion.
   Questions that need ad-hoc filtering across several columns ("total Electronics revenue in the North region for July and August")
   are answered only when the needed rows are retrieved, and the model must add them up. That is less reliable than a text-to-pandas tool.
3. **Small embedding model.** all-MiniLM-L6-v2 is fast and free but weaker than large hosted embedding models on long or highly
   technical text. Hybrid BM25 search partly compensates for this.
4. **Pattern-based safety filters.** Prompt-injection and jailbreak detection uses phrase patterns plus prompt hardening. A cleverly paraphrased
   injection could get past the filter. The strict prompt and the Validator's checks are the second and third lines of defence.
5. **Free-tier rate limits.** Gemini free-tier quotas (requests per minute and per day) limit throughput. With the LLM fact-check on, each
   question uses about 3 calls. The full evaluation pauses between questions for this reason.
6. **Single user, no authentication.** It is a local desktop app. There are no user accounts, no per-user document isolation and no audit log.
7. **Text only.** Charts and images inside PDFs or spreadsheets are ignored. Excel formulas are read as their last saved values.
8. **English-centred.** The prompts, keyword patterns and BM25 tokeniser assume English text.
9. **Data sent to the LLM provider.** The retrieved passages go to Google's API. Avoid uploading confidential documents.

## Challenges faced and how they were solved
| Challenge | Resolution |
|---|---|
| Avoiding paid API keys | Switched the default LLM to **Google Gemini's free tier** and added a Settings panel to paste, test and save the key without editing files. Without a key the app still runs in retrieval-only mode |
| Model churn: `gemini-pro` and the 1.x/2.0 models have been retired | Model IDs are configurable in the UI, with a "Custom…" option. The defaults are current free-tier models, and a clear message appears when a model is not found |
| PyTorch-based sentence-transformers is a large, fragile install on Windows | Used the same MiniLM model through ONNX Runtime (bundled with Chroma), so there is no PyTorch dependency |
| The embedding model download can be blocked by firewalls | Automatic fallback to an offline keyword-hashing embedder, with a UI warning |
| Aggregate questions over CSV/Excel fail with plain vector search | Generated **table summary documents** (totals, counts, highest/lowest per category) at ingestion, plus a summary-only search for aggregate questions |
| Exact identifiers (asset tags, product names) are missed by small embeddings | Added **BM25** keyword search and Reciprocal Rank Fusion |
| An injected instruction split across two lines, and therefore across chunks, slipped past a per-line filter | Moved sanitisation to **document level, before chunking**, with sentence-level splitting across line breaks. Removed values are stored so leaks can be detected |
| Gemini 3 models return message content as a list of blocks | `message_text()` flattens content blocks and ignores thinking blocks |
| Testing agents without spending API quota | `tests/fakes.py` provides a scripted fake LLM, so the whole graph, including the retry and refusal paths, is tested offline |
| Windows console encoding (cp1252) crashes on emoji output | The launcher prints ASCII only. The `.bat` file sets UTF-8 (`chcp 65001`, `PYTHONUTF8=1`) |

## Future work
- A text-to-pandas tool agent for arbitrary table questions, with sandboxed execution.
- OCR for scanned PDFs, and DOCX/PPTX loaders.
- A cross-encoder re-ranker for better passage ordering.
- Streaming answers in the UI, and conversation memory for follow-up questions.
- An LLM-based injection classifier in addition to the patterns.
- Per-user knowledge bases with authentication, for a shared deployment.
