"""Semantic retrieval: cosine similarity over an injected embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rag.embeddings.fake import HashEmbedder

from .base import BaseRetriever, RetrievalResult

if TYPE_CHECKING:
    from rag.interfaces import Embedder

# Back-compat alias for the relocated embedder.
SimpleEmbedder = HashEmbedder


class SemanticRetriever(BaseRetriever):
    """Ranks documents by cosine similarity. Defaults to the HashEmbedder fallback."""

    def __init__(self, embedder: Embedder | None = None, embedding_dim: int = 384):
        super().__init__(name="SemanticRetriever")
        self.embedding_dim = embedding_dim
        self.embedder = embedder or HashEmbedder(dim=embedding_dim)
        self.doc_embeddings: dict[str, np.ndarray] = {}

    def _build_index(self) -> None:
        self.doc_embeddings = {
            doc_id: self.embedder.embed(content) for doc_id, content in self.documents.items()
        }

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        if not self.documents:
            return []
        query_embedding = self.embedder.embed(query)
        scores = [
            (doc_id, float(np.dot(query_embedding, emb)))
            for doc_id, emb in self.doc_embeddings.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                document_id=doc_id,
                content=self.documents[doc_id],
                score=score,
                metadata=self.metadata.get(doc_id),
            )
            for doc_id, score in scores[:k]
        ]

    def estimate_tokens(self, text: str) -> int:
        return len(text.split()) // 2 + 1
