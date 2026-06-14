import numpy as np

from rag.embeddings import HashEmbedder, SimpleEmbedder
from rag.interfaces import Embedder
from rag.retrieval import SemanticRetriever


def test_hash_embedder_satisfies_protocol():
    assert isinstance(HashEmbedder(), Embedder)


def test_embed_shape_and_norm():
    e = HashEmbedder(dim=16)
    v = e.embed("hello")
    assert v.shape == (16,)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embed_deterministic_and_cached():
    e = HashEmbedder(dim=16)
    a = e.embed("same text")
    b = e.embed("same text")
    assert np.array_equal(a, b)
    assert "same text" in e.cache


def test_embed_batch_shape():
    e = HashEmbedder(dim=8)
    assert e.embed_batch(["a", "b", "c"]).shape == (3, 8)
    assert e.embed_batch([]).shape == (0, 8)


def test_simpleembedder_is_alias():
    assert SimpleEmbedder is HashEmbedder


def test_semantic_retriever_uses_injected_embedder():
    e = HashEmbedder(dim=32)
    r = SemanticRetriever(embedder=e)
    r.index_documents({"d1": "alpha", "d2": "beta"})
    assert r.embedder is e
    assert len(r.retrieve("alpha", k=2)) == 2


def test_semantic_retriever_default_embedder():
    assert isinstance(SemanticRetriever().embedder, HashEmbedder)
