"""
Provider-agnostic LLM factory (LangChain chat models).

Default provider: Google Gemini (free tier from Google AI Studio).

All agents call `invoke_llm()` which adds retry-with-backoff for rate limits
(HTTP 429 / RESOURCE_EXHAUSTED on the Gemini free tier) and turns provider errors
into a single friendly `LLMError`.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised with a user-friendly message when the LLM cannot be used."""


def get_llm(settings: Settings):
    if not settings.gemini_api_key.strip():
        raise LLMError(
            "Gemini API key is not set. Open the sidebar ⚙️ Settings, paste a free key from "
            "https://aistudio.google.com/app/apikey and press Save."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:  # pragma: no cover - dependency is in requirements.txt
        raise LLMError("Package langchain-google-genai is missing. Run the setup script again.") from e
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key.strip(),
        max_retries=1,                 # our own backoff loop handles the rest
        timeout=settings.llm_timeout_s,
    )


def message_text(msg: Any) -> str:
    """Extract plain text from a LangChain message (handles Gemini content-block lists)."""
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _is_rate_limit(err: Exception) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return any(s in text for s in ("429", "resource_exhausted", "resourceexhausted", "rate limit", "quota"))


def _friendly(err: Exception, settings: Settings) -> str:
    text = str(err)
    low = text.lower()
    if "api key not valid" in low or "api_key_invalid" in low or "permission_denied" in low or "401" in low:
        return "Gemini rejected the API key. Check the key in ⚙️ Settings (it usually starts with 'AIza')."
    if "not found" in low and "model" in low or "404" in low:
        return (f"Model '{settings.llm_model_name}' was not found for this key. "
                "Pick another model in ⚙️ Settings.")
    if _is_rate_limit(err):
        return ("The Gemini free-tier rate limit was reached. Wait a minute and try again, "
                "or switch to a Flash-Lite model in ⚙️ Settings.")
    if "connection" in low or "connect" in low or "timed out" in low or "timeout" in low:
        return "Network error while calling the LLM. Check your internet connection and try again."
    return f"LLM call failed: {text[:300]}"


def invoke_llm(llm, prompt: Any, settings: Settings, retries: int = 3) -> str:
    """Call the model with exponential backoff on rate limits. Returns plain text."""
    delay = 4.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return message_text(llm.invoke(prompt)).strip()
        except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
            last = e
            if _is_rate_limit(e) and attempt < retries:
                wait = delay * (2 ** attempt) + random.uniform(0, 1.5)
                log.warning("Rate limited (attempt %s); sleeping %.1fs", attempt + 1, wait)
                time.sleep(wait)
                continue
            break
    raise LLMError(_friendly(last, settings)) from last


def test_connection(settings: Settings) -> tuple[bool, str]:
    """Used by the Settings panel 'Test connection' button."""
    try:
        llm = get_llm(settings)
        reply = invoke_llm(llm, "Reply with exactly the word: OK", settings, retries=0)
        return True, f"Connected to Gemini / {settings.llm_model_name}. Model replied: {reply[:40]!r}"
    except LLMError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, _friendly(e, settings)
