"""
Embedding models.

Two backends, both free and running on the local CPU:

* "local" - sentence-transformers all-MiniLM-L6-v2 exported to ONNX (via chromadb).
            384-dimensional semantic vectors. Downloads ~80 MB once on first use and
            caches it in the user's home folder. No PyTorch needed, works on macOS
            (Intel + Apple Silicon) and Windows.
* "hash"  - a keyword-hashing embedding (unigrams + bigrams, signed feature hashing).
            Needs no download at all; used automatically if the local model cannot be
            downloaded (e.g. offline exam machine or a blocked network).
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from typing import List, Sequence

import numpy as np

log = logging.getLogger(__name__)

STOPWORDS = set(
    """a an the and or but if of to in on at by for with from into over under about as is are was were be been
    being do does did done have has had having this that these those it its it's i you he she we they them our your
    their my me what which who whom whose when where why how all any each few more most other some such no nor not
    only own same so than too very can will just should would could may might must shall there here then also
    please tell give show list find much many""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def _stem(tok: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "ed", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> List[str]:
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        t = t.replace(",", "")
        if t in STOPWORDS or len(t) < 2 and not t.isdigit():
            continue
        toks.append(_stem(t))
    return toks


class HashingEmbedder:
    name = "hash"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        for f, c in counts.items():
            h = zlib.crc32(f.encode("utf-8"))
            sign = 1.0 if (h >> 31) & 1 else -1.0
            weight = (1.0 + math.log(c)) * (0.6 if "_" in f else 1.0)
            v[h % self.dim] += sign * weight
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vec(t).tolist() for t in texts]


class LocalMiniLMEmbedder:
    name = "local"

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()
        self._ef(["warm-up"])  # triggers the one-time download / load now, so failures surface early

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = self._ef(list(texts))
        return [np.asarray(v, dtype=np.float32).tolist() for v in out]


_CACHE: dict[str, object] = {}
_LAST_WARNING: str = ""


def get_embedder(backend: str):
    """Return an embedder; falls back to hashing if the local model is unavailable."""
    global _LAST_WARNING
    backend = (backend or "local").lower()
    if backend in _CACHE:
        return _CACHE[backend]
    if backend == "local":
        try:
            emb = LocalMiniLMEmbedder()
            _CACHE["local"] = emb
            return emb
        except Exception as e:  # noqa: BLE001
            _LAST_WARNING = (
                "Could not load the local MiniLM embedding model (it downloads ~80 MB on first run). "
                f"Falling back to offline keyword embeddings. Reason: {str(e)[:160]}"
            )
            log.warning(_LAST_WARNING)
            emb = HashingEmbedder()
            _CACHE["local"] = emb
            return emb
    emb = HashingEmbedder()
    _CACHE[backend] = emb
    return emb


def embedding_warning() -> str:
    return _LAST_WARNING
