"""
End-to-end ingestion: validate -> load -> sanitize -> chunk -> embed -> store.

Each file is processed independently; one corrupt file never stops the batch.
Re-uploading a file with the same name replaces its previous chunks.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from app.config import UPLOAD_DIR, Settings, ensure_dirs
from app.guardrails import injection_payload_terms, neutralize_injections, safe_filename, validate_upload
from app.ingest.chunking import chunk_documents
from app.ingest.loaders import IngestionError, load_document
from app.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    file: str
    ok: bool
    chunks: int = 0
    message: str = ""
    seconds: float = 0.0
    injections_removed: int = 0


def ingest_file(path: Path, store: VectorStore, settings: Settings) -> IngestResult:
    t0 = time.time()
    try:
        docs = load_document(path, rows_per_chunk=settings.table_rows_per_chunk)
        removed_total = 0
        for d in docs:
            clean, removed = neutralize_injections(d.text)
            if removed:
                removed_total += len(removed)
                d.text = clean
                d.metadata["injection_removed"] = True
                d.metadata["injection_terms"] = ",".join(injection_payload_terms(removed))
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        store.delete_source(path.name)
        store.add_chunks(chunks)
        msg = f"{len(docs)} section(s) -> {len(chunks)} chunks"
        if removed_total:
            msg += f"; ⚠️ removed {removed_total} prompt-injection sentence(s)"
        return IngestResult(path.name, True, len(chunks), msg, round(time.time() - t0, 2), removed_total)
    except IngestionError as e:
        return IngestResult(path.name, False, 0, str(e), round(time.time() - t0, 2))
    except Exception as e:  # noqa: BLE001
        log.exception("Unexpected ingestion failure for %s", path)
        return IngestResult(path.name, False, 0, f"{path.name}: unexpected error ({e})", round(time.time() - t0, 2))


def ingest_uploads(files: Iterable[Tuple[str, bytes]], store: VectorStore, settings: Settings) -> List[IngestResult]:
    """files: iterable of (original_name, bytes) from the UI uploader."""
    ensure_dirs()
    results = []
    for name, data in files:
        ok, err = validate_upload(name, data, settings.max_file_mb)
        if not ok:
            results.append(IngestResult(name, False, 0, err))
            continue
        target = UPLOAD_DIR / safe_filename(name)
        target.write_bytes(data)
        results.append(ingest_file(target, store, settings))
    return results


def ingest_folder(folder: Path, store: VectorStore, settings: Settings) -> List[IngestResult]:
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and not p.name.startswith("."))
    return ingest_uploads(((p.name, p.read_bytes()) for p in files), store, settings)


def clear_uploads() -> None:
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    ensure_dirs()
