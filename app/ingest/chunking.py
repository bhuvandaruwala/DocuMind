"""
Data preparation for semantic search.

* Prose (PDF pages, TXT files): recursive character splitting. We try to cut on
  paragraph breaks first, then line breaks, then sentence ends, then spaces, so a
  chunk rarely ends in the middle of a sentence. Chunks are ~800 characters with
  ~120 characters of overlap so facts that straddle a boundary are not lost.
* Tables: already grouped into small row blocks (header repeated) by the loader,
  so they are kept whole unless a single block is unusually large.

Every chunk gets a stable, unique id and a human-readable `location` used in citations.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

from app.ingest.loaders import Document

SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", "; ", ", ", " ")


def _split(text: str, size: int, seps: Sequence[str]) -> List[str]:
    """Split text into pieces no longer than `size`, preferring the earliest separator."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            parts = text.split(sep)
            pieces: List[str] = []
            for j, p in enumerate(parts):
                piece = p + (sep if j < len(parts) - 1 else "")
                if len(piece) > size:
                    pieces.extend(_split(piece, size, seps[i + 1 :]))
                elif piece:
                    pieces.append(piece)
            return pieces
    return [text[k : k + size] for k in range(0, len(text), size)]  # no separator at all


def recursive_split(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    pieces = _split(text, chunk_size, SEPARATORS)

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        # start the next chunk with the tail of the previous one (overlap), cut at a word boundary
        tail = current[-overlap:] if overlap and current else ""
        if tail and " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        current = (tail + piece) if len(tail) + len(piece) <= chunk_size else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def location_label(meta: Dict) -> str:
    parts = [meta.get("source", "?")]
    if meta.get("page"):
        parts.append(f"page {meta['page']}")
    if meta.get("sheet"):
        parts.append(f"sheet '{meta['sheet']}'")
    if meta.get("rows"):
        parts.append("summary" if meta["rows"] == "all" else f"rows {meta['rows']}")
    return ", ".join(parts)


def chunk_documents(docs: List[Document], chunk_size: int = 800, overlap: int = 120) -> List[Dict]:
    """Return a list of {id, text, metadata} dicts ready for the vector store."""
    out: List[Dict] = []
    counters: Dict[str, int] = {}
    for doc in docs:
        meta = dict(doc.metadata)
        kind = meta.get("kind", "text")
        meta.setdefault("kind", kind)
        if kind.startswith("table") and len(doc.text) <= chunk_size * 3:
            texts = [doc.text]
        else:
            texts = recursive_split(doc.text, chunk_size, overlap)
        for t in texts:
            src = meta.get("source", "doc")
            idx = counters.get(src, 0)
            counters[src] = idx + 1
            m = {**meta, "chunk_index": idx, "location": location_label(meta)}
            digest = hashlib.sha1(f"{src}|{idx}|{t}".encode("utf-8")).hexdigest()[:12]
            out.append({"id": f"{src}::{idx}::{digest}", "text": t, "metadata": m})
    return out
