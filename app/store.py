"""
Vector knowledge store + search primitives used by the Retriever.

* Chroma PersistentClient stores chunks, metadata and embeddings on disk in
  data/chroma_db, so the knowledge base survives application restarts.
* Cosine distance; we compute embeddings ourselves (app/embeddings.py) so the same
  code works with the semantic model or the offline fallback.
* Besides vector search, a small in-memory BM25 keyword index is built from the
  stored chunks. Keyword search catches exact terms (asset tags, product names,
  numbers) that small embedding models often miss. The Retriever fuses both.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from app.config import CHROMA_DIR, ensure_dirs
from app.embeddings import get_embedder, tokenize

log = logging.getLogger(__name__)


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tfs), dtype=np.float32)
        if not q or not self.tfs:
            return out
        for i, tf in enumerate(self.tfs):
            s = 0.0
            dl = self.lens[i] or 1
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[i] = s
        return out


def _clean_meta(meta: Dict) -> Dict:
    """Chroma metadata values must be str/int/float/bool."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


class VectorStore:
    def __init__(self, embedding_backend: str = "local", persist_dir=CHROMA_DIR):
        import chromadb

        ensure_dirs()
        self.embedder = get_embedder(embedding_backend)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # one collection per embedding backend, so vector sizes never clash
        self.collection_name = f"docs_{self.embedder.name}"
        self.col = self.client.get_or_create_collection(
            self.collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._lock = threading.Lock()
        self._bm25: Optional[BM25] = None
        self._corpus: Optional[Dict[str, list]] = None

    # write
    def add_chunks(self, chunks: List[Dict], batch: int = 64) -> int:
        if not chunks:
            return 0
        with self._lock:
            for i in range(0, len(chunks), batch):
                part = chunks[i : i + batch]
                texts = [c["text"] for c in part]
                self.col.upsert(
                    ids=[c["id"] for c in part],
                    documents=texts,
                    metadatas=[_clean_meta(c["metadata"]) for c in part],
                    embeddings=self.embedder.embed(texts),
                )
            self._invalidate()
        return len(chunks)

    def delete_source(self, source: str) -> None:
        with self._lock:
            self.col.delete(where={"source": source})
            self._invalidate()

    def reset(self) -> None:
        with self._lock:
            self.client.delete_collection(self.collection_name)
            self.col = self.client.get_or_create_collection(
                self.collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
            )
            self._invalidate()

    def _invalidate(self) -> None:
        self._bm25 = None
        self._corpus = None

    # read
    def count(self) -> int:
        return self.col.count()

    def _load_corpus(self) -> Dict[str, list]:
        if self._corpus is None:
            data = self.col.get(include=["documents", "metadatas"])
            self._corpus = {"ids": data["ids"], "docs": data["documents"], "metas": data["metadatas"]}
            self._bm25 = BM25(self._corpus["docs"])
        return self._corpus

    def list_sources(self) -> Dict[str, Dict]:
        corpus = self._load_corpus()
        out: Dict[str, Dict] = {}
        for m in corpus["metas"]:
            s = out.setdefault(m.get("source", "?"), {"chunks": 0, "file_type": m.get("file_type", "")})
            s["chunks"] += 1
        return dict(sorted(out.items()))

    def vector_search(self, query: str, k: int = 10, where: Optional[Dict] = None) -> List[Dict]:
        n = self.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(k, n),
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        hits = []
        for i, cid in enumerate(res["ids"][0]):
            hits.append(
                {
                    "id": cid,
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "similarity": round(1.0 - float(res["distances"][0][i]), 4),
                    "embedding": np.asarray(res["embeddings"][0][i], dtype=np.float32),
                }
            )
        return hits

    def keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        corpus = self._load_corpus()
        if not corpus["ids"]:
            return []
        scores = self._bm25.scores(query)
        order = np.argsort(-scores)[:k]
        top = float(scores[order[0]]) if len(order) else 0.0
        hits = []
        for idx in order:
            s = float(scores[idx])
            if s <= 0:
                break
            hits.append(
                {
                    "id": corpus["ids"][idx],
                    "text": corpus["docs"][idx],
                    "metadata": corpus["metas"][idx],
                    "bm25": round(s, 4),
                    "bm25_norm": round(s / top, 4) if top else 0.0,
                }
            )
        return hits

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.embedder.embed(texts), dtype=np.float32)


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lam: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance: pick relevant but mutually diverse candidates."""
    if len(cand_vecs) == 0:
        return []
    def norm(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(n == 0, 1, n)
    q = norm(query_vec.reshape(1, -1))[0]
    c = norm(cand_vecs)
    rel = c @ q
    selected: List[int] = []
    remaining = list(range(len(c)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sim_sel = c[remaining] @ c[selected].T
            scores = lam * rel[remaining] - (1 - lam) * sim_sel.max(axis=1)
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


_STORES: Dict[str, VectorStore] = {}
_STORES_LOCK = threading.Lock()


def get_store(embedding_backend: str = "local") -> VectorStore:
    with _STORES_LOCK:
        if embedding_backend not in _STORES:
            _STORES[embedding_backend] = VectorStore(embedding_backend)
        return _STORES[embedding_backend]
