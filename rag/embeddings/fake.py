"""Deterministic hash-based embedder: a zero-dependency fallback, not semantically meaningful.

The real embedding model (bge-m3) replaces this in M1; kept as the default and test fallback.
"""

from __future__ import annotations

import numpy as np


class HashEmbedder:
    """Hash-seeded random unit vectors. Same text gives the same vector (per process), cached."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        self.cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


# Back-compat alias for the original name.
SimpleEmbedder = HashEmbedder
