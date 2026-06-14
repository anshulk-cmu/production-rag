"""Deterministic hash-based embedder: a zero-dependency fallback, not semantically meaningful.

The real embedding model (bge-m3) replaces this in M1; kept as the default and test fallback.
"""

from __future__ import annotations

import hashlib

import numpy as np


class HashEmbedder:
    """Deterministic hash-seeded unit vectors. Same text -> same vector across runs; cached."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        # Stable md5-based seed (reproducible across processes) + a local RNG (no global state).
        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        self.cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


# Back-compat alias for the original name.
SimpleEmbedder = HashEmbedder
