"""
Reliability and safety controls.

Input side
  * validate_upload   - extension allow-list, size limit, empty files, content sniffing
                        (a .pdf must start with %PDF, .docx/.xlsx must be a zip, text must not be binary)
  * validate_question - length limits, empty input, obvious jailbreak / prompt-extraction attempts
Context side
  * neutralize_injections - retrieved text is DATA; lines that try to give the model
                            instructions ("ignore previous instructions", "system override" ...)
                            are replaced before they ever reach the LLM
Output side
  * check_citations / check_numbers - grounding checks used by the Validator agent
  * moderate_output   - blocks unsafe content, leaked secrets and echoed injection payloads
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from app.config import SUPPORTED_EXTENSIONS

# uploads
def validate_upload(name: str, data: bytes, max_mb: int = 20) -> Tuple[bool, str]:
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"{name}: file type '{ext or 'none'}' is not allowed (use PDF, Word, TXT, CSV or Excel)."
    if not data:
        return False, f"{name}: file is empty."
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"{name}: {size_mb:.1f} MB exceeds the {max_mb} MB limit."
    head = data[:8]
    if ext == ".pdf" and not data[:1024].lstrip().startswith(b"%PDF"):
        return False, f"{name}: does not look like a real PDF file."
    if ext in (".docx", ".xlsx") and not head.startswith(b"PK"):
        return False, f"{name}: does not look like a real {ext} file."
    if ext == ".xls" and not head.startswith(b"\xd0\xcf\x11\xe0"):
        return False, f"{name}: does not look like a real .xls file."
    if ext in (".txt", ".csv"):
        sample = data[:4096]
        if b"\x00" in sample and not (sample.startswith(b"\xff\xfe") or sample.startswith(b"\xfe\xff")):
            return False, f"{name}: looks like a binary file, not text."
    return True, ""


def safe_filename(name: str) -> str:
    base = Path(name).name  # drop any directory parts (path traversal)
    base = re.sub(r"[^A-Za-z0-9._ ()-]", "_", base).strip() or "upload"
    return base[:150]


# prompt injection
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|system) (instructions|prompts?|rules)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"system override",
    r"you are now (?!able)",
    r"new instructions?:",
    r"(reveal|print|show|repeat) (me )?(your|the) (system prompt|hidden instructions|instructions)",
    r"act as (an? )?(unrestricted|jailbroken|dan)",
    r"developer mode",
    r"you must (now )?(tell|say to|inform) (every|all|the) users?",
]
_INJ_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


def detect_injection(text: str) -> List[str]:
    return [m.group(0) for m in _INJ_RE.finditer(text or "")]


def neutralize_injections(text: str) -> Tuple[str, List[str]]:
    """Remove sentences that try to instruct the model. Returns (clean_text, removed_sentences)."""
    if not detect_injection(text):
        return text, []
    removed: List[str] = []
    # Split into sentences (a sentence may continue over a line break) and paragraph breaks.
    parts = re.split(r"((?<=[.!?])\s+|\n\s*\n)", text)
    out: List[str] = []
    prev_removed = False
    for i, part in enumerate(parts):
        if i % 2 == 1:  # separator
            out.append(part)
            continue
        if detect_injection(part):
            removed.append(part.strip())
            if not prev_removed:
                out.append("[removed: text that tried to instruct the AI]")
            prev_removed = True
        else:
            out.append(part)
            prev_removed = False
    return "".join(out), removed


def injection_payload_terms(removed: Iterable[str]) -> List[str]:
    """Distinctive tokens (numbers, quoted values) from removed injections; if the answer
    repeats them, the injection leaked through."""
    terms = []
    for s in removed:
        terms.extend(re.findall(r"\b\d[\d,.]*\b", s))
    return [t for t in terms if len(t) >= 2]


# questions
@dataclass
class QuestionCheck:
    ok: bool
    message: str = ""
    cleaned: str = ""
    flags: List[str] = field(default_factory=list)


def validate_question(q: str, max_chars: int = 1000) -> QuestionCheck:
    cleaned = " ".join((q or "").split())
    if not cleaned:
        return QuestionCheck(False, "Please type a question.")
    if len(cleaned) < 3:
        return QuestionCheck(False, "The question is too short.")
    if len(cleaned) > max_chars:
        return QuestionCheck(False, f"The question is too long ({len(cleaned)} characters, limit {max_chars}).")
    hits = detect_injection(cleaned)
    if hits:
        return QuestionCheck(
            False,
            "This request tries to change the assistant's instructions, so it was blocked. "
            "Please ask a question about the uploaded documents.",
            cleaned,
            hits,
        )
    return QuestionCheck(True, "", cleaned)


# grounding
_CITE_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def extract_citations(answer: str) -> List[str]:
    found: List[str] = []
    for grp in _CITE_RE.findall(answer or ""):
        for c in grp.split(","):
            c = c.strip()
            if c and c not in found:
                found.append(c)
    return found


def check_citations(answer: str, valid_labels: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return (cited_valid, cited_invalid)."""
    valid = set(valid_labels)
    cited = extract_citations(answer)
    return [c for c in cited if c in valid], [c for c in cited if c not in valid]


_NUM_RE = re.compile(r"(?<![A-Za-z\d])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def check_numbers(answer: str, context_texts: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Every number in the answer should appear somewhere in the retrieved context.
    Returns (grounded_numbers, ungrounded_numbers). Citation labels and list markers are ignored."""
    body = _CITE_RE.sub(" ", answer or "")
    body = re.sub(r"(?m)^\s*\d+[.)]\s", " ", body)  # numbered list markers
    nums = {_norm_num(n) for n in _NUM_RE.findall(body)}
    nums = {n for n in nums if n not in {"0", "1", "2", "3"}}  # tiny numbers are too ambiguous to check
    ctx_nums = set()
    for t in context_texts:
        ctx_nums.update(_norm_num(n) for n in _NUM_RE.findall(t or ""))
    grounded = sorted(n for n in nums if n in ctx_nums)
    ungrounded = sorted(n for n in nums if n not in ctx_nums)
    return grounded, ungrounded


# output moderation
_SECRET_RE = re.compile(r"\b(AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,})\b")
UNSAFE_PATTERNS = re.compile(
    r"\b(how to (make|build) (a )?(bomb|explosive|weapon)|kill yourself|credit card number is)\b", re.IGNORECASE
)


def moderate_output(answer: str, injection_terms: Iterable[str] = ()) -> Tuple[bool, str, str]:
    """Return (ok, reason, possibly_redacted_answer)."""
    text = _SECRET_RE.sub("[redacted secret]", answer or "")
    if UNSAFE_PATTERNS.search(text):
        return False, "unsafe content in the generated answer", text
    if re.search(r"(system prompt|my instructions are)", text, re.IGNORECASE) and "SOURCES" in text:
        return False, "answer appears to leak the system prompt", text
    for term in injection_terms:
        if re.search(rf"(?<![\d.,]){re.escape(term)}(?![\d])", text):
            return False, f"answer repeats a value ('{term}') that only appeared in a prompt-injection", text
    return True, "", text


def summarize_flags(flags: Dict[str, List[str]]) -> str:
    return "; ".join(f"{k}: {', '.join(v)}" for k, v in flags.items() if v)
