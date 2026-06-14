from rag.interfaces import Reranker
from rag.retrieval import BM25Retriever, CrossEncoderReranker, HybridRetriever
from rag.retrieval.base import RetrievalResult
from rag.retrieval.reranker import SimpleRerankerModel


def test_retrieval_result_repr():
    r = RetrievalResult("d1", "content", 0.5)
    assert "d1" in repr(r) and "0.5" in repr(r)


def test_index_documents_and_accessors(corpus):
    r = BM25Retriever()
    r.index_documents(corpus, {"doc_1": {"src": "x"}})
    assert r.get_document("doc_1") == corpus["doc_1"]
    assert r.get_metadata("doc_1") == {"src": "x"}
    assert r.get_metadata("missing") == {}
    assert len(r.documents) == len(corpus)


def test_bm25_ranks_exact_term_first(corpus):
    r = BM25Retriever()
    r.index_documents(corpus)
    assert r.retrieve("term frequency", k=4)[0].document_id == "doc_2"


def test_bm25_empty_query_and_no_docs(corpus):
    r = BM25Retriever()
    r.index_documents(corpus)
    assert r.retrieve("", k=5) == []
    assert BM25Retriever().retrieve("x", k=5) == []


def test_bm25_tokenize():
    assert BM25Retriever()._tokenize("Hello World") == ["hello", "world"]


def test_hybrid_returns_results_and_no_docs(corpus):
    r = HybridRetriever()
    r.index_documents(corpus)
    res = r.retrieve("term frequency ranking", k=3)
    assert 1 <= len(res) <= 3
    assert all(isinstance(x, RetrievalResult) for x in res)
    assert HybridRetriever().retrieve("x", k=5) == []


def test_hybrid_set_weights_normalizes():
    r = HybridRetriever()
    r.set_weights(semantic=3, bm25=1)
    assert abs(r.semantic_weight - 0.75) < 1e-9
    assert abs(r.bm25_weight - 0.25) < 1e-9


def test_rerank_orders_by_overlap():
    out = CrossEncoderReranker().rerank(
        "machine learning",
        [
            RetrievalResult("d1", "totally unrelated text here", 0.9),
            RetrievalResult("d2", "machine learning models on data", 0.1),
        ],
    )
    assert out[0].document_id == "d2"


def test_rerank_keeps_original_score_in_metadata():
    out = CrossEncoderReranker().rerank(
        "machine learning", [RetrievalResult("d1", "machine learning", 0.42)]
    )
    assert out[0].metadata["original_score"] == 0.42
    assert "reranker_score" in out[0].metadata


def test_rerank_empty_and_top_k():
    rr = CrossEncoderReranker()
    assert rr.rerank("q", []) == []
    cands = [RetrievalResult(f"d{i}", "machine learning text", 0.1) for i in range(5)]
    assert len(rr.rerank("machine learning", cands, k=2)) == 2


def test_reranker_model_satisfies_protocol():
    assert isinstance(SimpleRerankerModel(), Reranker)


def test_reranker_score_batch():
    out = SimpleRerankerModel().score_batch("machine learning", ["machine learning text", "x"])
    assert len(out) == 2 and all(isinstance(v, float) for v in out)


def test_hybrid_score_is_monotonic_with_rank(corpus):
    r = HybridRetriever()
    r.index_documents(corpus)
    res = r.retrieve("term frequency ranking", k=4)
    scores = [x.score for x in res]
    assert scores == sorted(scores, reverse=True)  # stored score == RRF rank order
