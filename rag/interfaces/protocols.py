"""Component contracts. Every swappable part implements one of these Protocols.

Protocols are runtime_checkable, so existing classes satisfy them by shape (no inheritance).
The dataclasses below are the data the contracts exchange and live here so the contracts are
self-contained; RetrievalResult is the existing type from rag.retrieval.base.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from rag.retrieval.base import RetrievalResult


@dataclass
class Document:
    doc_id: str
    text: str
    source: str | None = None
    metadata: dict | None = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int = 0
    end: int = 0
    metadata: dict | None = None


@dataclass
class GradeResult:
    label: str  # correct | ambiguous | incorrect
    score: float
    action: str  # use | refine | discard
    per_context: list[float] = field(default_factory=list)


@dataclass
class RouteDecision:
    strategy: str
    retriever: str
    use_transforms: bool = False
    reason: str = ""


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    contexts: list[RetrievalResult] = field(default_factory=list)
    grade: GradeResult | None = None
    transforms_used: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, document: str) -> float: ...

    def score_batch(self, query: str, documents: list[str]) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    dim: int

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None
    ) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...

    def __len__(self) -> int: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]: ...


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, source: str) -> list[Document]: ...


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str: ...

    def stream(self, prompt: str, **kwargs) -> Iterable[str]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, contexts: list[RetrievalResult]) -> GenerationResult: ...


@runtime_checkable
class QueryTransform(Protocol):
    def transform(self, query: str) -> list[str]: ...


@runtime_checkable
class RetrievalGrader(Protocol):
    def grade(self, query: str, contexts: list[RetrievalResult]) -> GradeResult: ...


@runtime_checkable
class Router(Protocol):
    def route(self, query: str) -> RouteDecision: ...
