import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import SAMPLE_DATA_DIR, Settings  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DATA_DIR


@pytest.fixture()
def settings() -> Settings:
    # offline embeddings + LLM fact-check on; no real API key needed (tests use a scripted LLM)
    return Settings(embedding_backend="hash", gemini_api_key="", llm_validation=True, max_agent_loops=2)


@pytest.fixture()
def store(tmp_path):
    from app.store import VectorStore

    return VectorStore("hash", persist_dir=tmp_path / "chroma")


@pytest.fixture()
def loaded_store(store, settings, sample_dir):
    from app.ingest.pipeline import ingest_folder

    results = ingest_folder(sample_dir, store, settings)
    assert all(r.ok for r in results), [r.message for r in results if not r.ok]
    return store
