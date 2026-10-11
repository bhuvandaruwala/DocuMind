# Setup and Deployment Guide

The application runs locally on **macOS** (Apple Silicon or Intel) and **Windows 10/11**.
It does not use Docker. One launcher script creates an isolated Python environment, installs the dependencies and starts the app.

## 1. Prerequisites

| Item | Notes |
|---|---|
| Python 3.11 or 3.12 (3.10-3.13 supported) | <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"**. On macOS you can also use `brew install python@3.12` |
| Internet on first run | Installs packages and downloads the MiniLM embedding model (~80 MB, cached in your home folder) |
| Google Gemini API key (free) | <https://aistudio.google.com/app/apikey>. Sign in with a Google account and click **Create API key**. No credit card is needed for the free tier |
| Approx 1.5 GB disk, 4 GB RAM | Everything runs on the CPU. No GPU is needed |

## 2. Start the app

### macOS
1. Unzip `DocuMind_Project.zip` (for example to `~/Desktop/DocuMind_Project`).
2. Double-click **`start_mac.command`**.
   If macOS says the file "cannot be opened because it is from an unidentified developer", right-click it, choose **Open**, then click **Open** again.
   Or open **Terminal** and run:
   ```bash
   cd ~/Desktop/DocuMind_Project
   python3 start.py
   ```

### Windows
1. Unzip `DocuMind_Project.zip` (right-click, **Extract All…**). Do not run it from inside the zip.
2. Double-click **`start_windows.bat`**.
   Or open **Command Prompt** in the folder and run:
   ```bat
   py start.py
   ```
   If Windows SmartScreen appears, click **More info**, then **Run anyway**.

### What happens
```
[start] Creating virtual environment in .venv (first run only)...
[start] Installing dependencies (first run takes 2-5 minutes)...
[start] Created .env from .env.example
[start] Starting the app at http://localhost:8501
```
The browser opens automatically. If Streamlite asks for email, simply press enter and the browser will open, If it does not, open the printed address. Keep the terminal window open while you use the app. Press **Ctrl+C** to stop.

## 3. Configure Gemini (Settings panel)

1. In the sidebar, open **⚙️ Settings**.
2. Paste the key into **Gemini API key**. It usually starts with `AIza`.
3. Choose a model. **gemini-3.5-flash-lite** is the default because it has the most generous free-tier limits.
   **gemini-3.5-flash** and **gemini-3.8-flash** give stronger answers. **Custom…** accepts any other model ID.
4. Click **🔌 Test**. A message like "Connected to Gemini / gemini-3.5-flash-lite" confirms the key works.
5. Click **💾 Save** to keep the settings in the local `.env` file for the next run.


## 4. Use the app

1. **Knowledge base** (sidebar): upload files and click **📥 Ingest**, or click **🧪 Samples**.
   Each file reports how many chunks it produced. Files that fail validation (wrong type, too large, corrupt or scanned PDF) are rejected one at a time, and the rest of the batch continues.
2. **💬 Ask**: type a question or click an example. You see live agent progress, then the answer with
   a status badge (✅ Verified, ⚠️ Could not fully verify, 🔍 Not found, ⛔ Blocked), a confidence score,
   the cited sources and the full agent trace.
3. **📊 Evaluation**: run the golden set in *retrieval* mode (instant, no LLM) or *full* mode (uses Gemini, about 3-5 minutes).
4. **🧭 How it works**: the architecture diagram and agent descriptions.

## 5. Manual installation (without start.py)

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # or requirements-lock.txt for the exact tested versions
copy .env.example .env               # macOS: cp .env.example .env
streamlit run app/ui.py
```

## 6. Tests and evaluation

```bash
python3 start.py --test       # 32 tests, offline, about 10 s (a scripted fake LLM replaces Gemini)
python3 start.py --eval       # retrieval metrics on the golden set
.venv/bin/python -m evaluation.run_eval --mode full     # full pipeline with Gemini
```
Reports are written to `evaluation/results/`.

## 7. Packaging for submission

```bash
python3 start.py --package     # Windows: py start.py --package
```
This creates `dist/DocuMind_Project.zip` with the source, sample data, tests and docs. It leaves out `.venv`, `data/`, `.env` (your API key) and caches.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Python was not found` (Windows) | Reinstall Python with **Add to PATH** ticked, or run `py start.py` |
| `python3: command not found` (macOS) | Install Python from python.org. The installer adds `python3` |
| `ImportError` / DLL load failed for onnxruntime (Windows) | Install the Microsoft Visual C++ Redistributable (<https://aka.ms/vs/17/release/vc_redist.x64.exe>), then run `py start.py --reinstall` |
| Dependency installation fails | Check your internet or proxy, then run `start.py --reinstall`. On Python 3.14 install 3.12 instead. As a last resort use `pip install -r requirements-lock.txt` |
| Yellow warning: *Could not load the local MiniLM embedding model* | The ~80 MB model download was blocked. The app automatically uses offline keyword embeddings. Allow downloads from `chroma-onnx-models.s3.amazonaws.com` and restart to get semantic search |
| *Gemini rejected the API key* | Copy the key again from AI Studio. Make sure there are no spaces. Click Test |
| *Model not found* | Pick another model in Settings. Google retires older model IDs over time |
| *Free-tier rate limit reached* | Wait a minute, use a Flash-Lite model, or turn off the LLM fact-check. The app already retries with backoff |
| Port 8501 busy | The launcher picks the next free port automatically, or use `--port 8600` |
| Start over with an empty knowledge base | Click **Clear knowledge base**, or delete the `data/` folder |

## 9. Uninstall

Delete the project folder. Everything (virtual environment, vector DB, uploads, `.env`) lives inside it.
The embedding model cache is in `~/.cache/chroma` (Windows: `%USERPROFILE%\.cache\chroma`).
