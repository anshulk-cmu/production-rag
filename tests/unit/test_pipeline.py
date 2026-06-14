from evaluation import Evaluator
from rag import BM25Retriever, HybridRetriever, RAGPipeline


def test_pipeline_query_with_rerank(corpus):
    rag = RAGPipeline(retriever=BM25Retriever())
    rag.index_documents(corpus)
    res = rag.query("term frequency", k=4, rerank=True, rerank_k=2)
    assert len(res) <= 2
    assert all(hasattr(r, "document_id") for r in res)


def test_pipeline_no_rerank_no_optimize(corpus):
    rag = RAGPipeline(retriever=BM25Retriever())
    rag.index_documents(corpus)
    res = rag.query("term frequency", k=4, optimize_query=False, rerank=False)
    assert res[0].document_id == "doc_2"


def test_pipeline_batch_and_stats(corpus):
    rag = RAGPipeline(retriever=BM25Retriever())
    rag.index_documents(corpus)
    out = rag.batch_query(
        ["term frequency", "rank fusion"], k=3, optimize_query=False, rerank=False
    )
    assert len(out) == 2
    stats = rag.get_stats()
    assert stats["queries_processed"] == 2
    assert stats["documents_indexed"] == len(corpus)
    assert stats["retriever"] == "BM25Retriever"


def test_pipeline_format_results(corpus):
    rag = RAGPipeline(retriever=BM25Retriever())
    rag.index_documents(corpus)
    res = rag.query("term frequency", optimize_query=False, rerank=False)
    out = rag.format_results(res)
    assert "Score:" in out and "ID:" in out


def test_evaluator_evaluate(corpus, gold):
    r = BM25Retriever()
    r.index_documents(corpus)
    result = Evaluator().evaluate(r, gold, k_values=[1, 5], verbose=False)
    assert result.retriever_name == "BM25Retriever"
    assert result.num_queries == len(gold)
    assert "precision_at_1" in result.avg_metrics
    assert 0.0 <= result.avg_metrics["map"] <= 1.0


def test_evaluator_compare(corpus, gold):
    retrievers = [BM25Retriever(), HybridRetriever()]
    for r in retrievers:
        r.index_documents(corpus)
    out = Evaluator().compare(retrievers, gold, k_values=[1, 5])
    assert set(out) == {"BM25Retriever", "HybridRetriever"}
