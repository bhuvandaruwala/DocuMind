"""Configuration round-trip and LLM factory errors (no network)."""

import pytest

import app.config as config
from app.llm import LLMError, get_llm, message_text


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", env)
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "TOP_K"):
        monkeypatch.delenv(var, raising=False)
    s = config.Settings(gemini_api_key="AIzaTESTKEY1234567890", gemini_model="gemini-3.5-flash", top_k=7)
    config.save_settings(s)
    loaded = config.load_settings()
    assert loaded.gemini_api_key == "AIzaTESTKEY1234567890"
    assert loaded.gemini_model == "gemini-3.5-flash" and loaded.top_k == 7
    assert loaded.masked_key().startswith("AIza") and "TESTKEY" not in loaded.masked_key()


def test_placeholder_key_is_ignored(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=your_google_api_key_here\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert config.load_settings().gemini_api_key == ""


def test_missing_key_gives_friendly_error():
    with pytest.raises(LLMError, match="API key"):
        get_llm(config.Settings(gemini_api_key=""))


def test_gemini_client_builds_without_network():
    llm = get_llm(config.Settings(gemini_api_key="AIzaFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE12"))
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"


def test_message_text_handles_content_blocks():
    class M:
        content = [{"type": "text", "text": "Hello "}, {"type": "thinking", "thinking": "x"}, {"type": "text", "text": "world"}]
    assert message_text(M()) == "Hello world"
    assert message_text("plain") == "plain"
