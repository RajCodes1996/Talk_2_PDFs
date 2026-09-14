"""
vector_store.py
Embeds text chunks using a local HuggingFace sentence-transformer model
and stores/searches them in memory.

Flow:
  chunks -> embed (sentence-transformers) -> search -> similarity results

FAISS is used when available. If it cannot be imported, the code falls back
to a pure NumPy cosine-similarity search so the app still runs.
"""

from typing import List, Tuple
import os

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    faiss = None

# ── Detect best device for encoding ──────────────────────────────────────────
def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Module-level model cache (loaded once per process) ────────────────────────
_MODEL_CACHE: dict = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading it only once."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(
            model_name, device=_get_device()
        )
    return _MODEL_CACHE[model_name]


# Small, fast embedding model for this accessibility use case.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class FAISSVectorStore:
    """
    In-memory vector store.

    Uses FAISS flat inner-product search when available, and falls back to a
    NumPy cosine-similarity search when it is not.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        # Reuse cached model — avoids reloading weights on every upload
        self.model = _get_model(model_name)
        self.index = None
        self.chunks: List[str] = []
        self.dimension = None
        self._embeddings = None
        self._use_faiss = faiss is not None

    def build(self, chunks: List[str]) -> None:
        """
        Embed all chunks and build the search index.
        """
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        self.chunks = chunks

        # Larger batch + multi-process workers saturate CPU/GPU better
        # num_proc > 0 only helps on Linux/Mac; keep 0 on Windows to avoid spawn overhead
        _num_workers = 0 if os.name == "nt" else 4
        embeddings = self.model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            num_workers=_num_workers,
            convert_to_numpy=True,
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings.shape[1]

        if self._use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            self._embeddings = None
        else:
            self.index = None
            self._embeddings = embeddings

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search the vector store for the most relevant chunks.
        """
        if self.index is None and self._embeddings is None:
            raise RuntimeError("Index not built. Call build() first.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        if self._use_faiss and self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    results.append((self.chunks[idx], float(score)))
            return results

        if top_k <= 0:
            return []

        scores = np.matmul(self._embeddings, query_embedding[0])
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in top_indices]

    def is_ready(self) -> bool:
        return (self.index is not None or self._embeddings is not None) and len(self.chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
