"""
Central configuration.

Every setting can come from three places, in this order of priority:
  1. Values changed at runtime in the Streamlit "Settings" panel
  2. The local .env file in the project folder (written by the Settings panel)
  3. The defaults defined below

"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv, set_key

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"            # runtime data (vector DB, uploads) - gitignored
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
EVAL_DIR = BASE_DIR / "evaluation"

# Gemini models that offer a free tier in Google AI Studio (checked Oct 2026).
# Any other model ID can be typed into the Settings panel.
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]
DEFAULT_GEMINI_MODEL = GEMINI_MODELS[0]

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls")
NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # ---- LLM ----
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_timeout_s: int = 60

    # ---- Embeddings / vector store ----
    # local = all-MiniLM-L6-v2 (ONNX, runs on CPU, ~80 MB one-time download)
    # hash  = offline keyword-hashing embeddings (no download; lower quality)
    embedding_backend: str = "local"

    # ---- Ingestion ----
    chunk_size: int = 800
    chunk_overlap: int = 120
    table_rows_per_chunk: int = 8
    max_file_mb: int = 20

    # ---- Retrieval / agents ----
    top_k: int = 6
    max_agent_loops: int = 2              # total passes through the graph (1 retry)
    llm_validation: bool = True           # Validator also asks the LLM to check grounding
    max_question_chars: int = 1000

    extra: dict = field(default_factory=dict)

    @property
    def llm_model_name(self) -> str:
        return self.gemini_model

    @property
    def llm_ready(self) -> bool:
        return bool(self.gemini_api_key.strip())

    def masked_key(self) -> str:
        k = self.gemini_api_key.strip()
        if not k:
            return "(not set)"
        return k[:4] + "…" + k[-4:] if len(k) > 10 else "••••"

    def copy(self, **changes) -> "Settings":
        return replace(self, **changes)

    def as_dict(self, hide_secrets: bool = True) -> dict:
        d = asdict(self)
        if hide_secrets:
            d["gemini_api_key"] = self.masked_key()
        return d


def load_settings() -> Settings:
    """Read settings from environment variables / .env file."""
    load_dotenv(ENV_FILE, override=True)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if key.strip().lower().startswith("your_"):   # placeholder copied from .env.example
        key = ""
    return Settings(
        gemini_api_key=key.strip(),
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
        llm_timeout_s=_env_int("LLM_TIMEOUT_S", 60),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "local").strip().lower() or "local",
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
        table_rows_per_chunk=_env_int("TABLE_ROWS_PER_CHUNK", 8),
        max_file_mb=_env_int("MAX_FILE_MB", 20),
        top_k=_env_int("TOP_K", 6),
        max_agent_loops=_env_int("MAX_AGENT_LOOPS", 2),
        llm_validation=_env_bool("LLM_VALIDATION", True),
        max_question_chars=_env_int("MAX_QUESTION_CHARS", 1000),
    )


def save_settings(s: Settings, include_key: bool = True) -> Path:
    """Persist the user-editable settings to the local .env file."""
    ENV_FILE.touch(exist_ok=True)
    values = {
        "GEMINI_MODEL": s.gemini_model,
        "EMBEDDING_BACKEND": s.embedding_backend,
        "TOP_K": str(s.top_k),
        "MAX_AGENT_LOOPS": str(s.max_agent_loops),
        "LLM_VALIDATION": "true" if s.llm_validation else "false",
    }
    if include_key:
        values["GOOGLE_API_KEY"] = s.gemini_api_key
    for k, v in values.items():
        set_key(str(ENV_FILE), k, v, quote_mode="never")
    return ENV_FILE


def ensure_dirs() -> None:
    for d in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    st = load_settings()
    for k, v in st.as_dict().items():
        print(f"{k:22s} {v}")
