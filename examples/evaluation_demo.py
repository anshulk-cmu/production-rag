#!/usr/bin/env python3
"""
Evaluation benchmark example - Compare retrievers on test set.

This demonstrates:
1. Creating test queries with ground truth
2. Evaluating different retrieval strategies
3. Comparing metrics (Precision, Recall, NDCG, MRR)
4. Displaying results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import SemanticRetriever, BM25Retriever, HybridRetriever
from evaluation import Evaluator


# Sample documents
DOCUMENTS = {
    "doc_1": "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and models that enable computers to learn from and make predictions based on data.",
    "doc_2": "Deep learning uses neural networks with multiple layers to progressively extract features from raw input.",
    "doc_3": "Natural language processing (NLP) is a branch of AI that helps computers understand human language.",
    "doc_4": "Computer vision is a field that trains computers to interpret visual information using deep learning.",
    "doc_5": "Artificial intelligence encompasses technologies that enable machines to sense and learn.",
    "doc_6": "Data science combines statistics and programming to extract insights from data.",
    "doc_7": "Neural networks are computing systems inspired by biological neurons.",
    "doc_8": "Large language models are trained on massive text data and can generate human-like text.",
    "doc_9": "Reinforcement learning teaches agents to make decisions through rewards and penalties.",
    "doc_10": "Transfer learning enables models trained on one task to be adapted for other tasks.",
}

# Test queries with ground truth relevance
# Format: (query, {set of relevant document IDs})
TEST_QUERIES = [
    (
        "machine learning algorithms",
        {"doc_1", "doc_9", "doc_10"}
    ),
    (
        "deep learning neural networks",
        {"doc_2", "doc_7", "doc_4"}
    ),
    (
        "natural language processing NLP",
        {"doc_3", "doc_8"}
    ),
    (
        "computer vision images",
        {"doc_4", "doc_2"}
    ),
    (
        "artificial intelligence",
        {"doc_5", "doc_1", "doc_3", "doc_4"}
    ),
    (
        "data science analytics",
        {"doc_6", "doc_1"}
    ),
    (
        "neural networks deep learning",
        {"doc_7", "doc_2", "doc_4"}
    ),
    (
        "transfer learning models",
        {"doc_10", "doc_2", "doc_9"}
    ),
]


def create_retrievers() -> list:
    """Create different retriever strategies"""
    retrievers = [
        SemanticRetriever(),
        BM25Retriever(),
        HybridRetriever(semantic_weight=0.5),
    ]

    # Index documents in each retriever
    for retriever in retrievers:
        retriever.index_documents(DOCUMENTS)

    return retrievers


def main():
    """Run evaluation benchmark"""
    print("\n" + "="*70)
    print("EVALUATION FRAMEWORK - Retriever Comparison")
    print("="*70)

    print(f"\nDataset:")
    print(f"  Documents: {len(DOCUMENTS)}")
    print(f"  Test queries: {len(TEST_QUERIES)}")

    # Create retrievers
    retrievers = create_retrievers()

    # Create evaluator
    evaluator = Evaluator()

    # Evaluate each retriever
    print(f"\n{'='*70}")
    results = evaluator.compare(
        retrievers=retrievers,
        test_queries=TEST_QUERIES,
        k_values=[1, 5, 10]
    )

    # Print formatted comparison
    print("\n" + "="*70)
    print("RETRIEVER COMPARISON RESULTS")
    print("="*70)

    evaluator.print_comparison(results)

    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED METRICS BY RETRIEVER")
    print("="*70)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Number of queries: {result.num_queries}")
        print(f"  Average Metrics:")
        for metric, value in result.avg_metrics.items():
            print(f"    {metric}: {value:.3f}")

    # Find best performer
    print("\n" + "="*70)
    print("BEST PERFORMER BY METRIC")
    print("="*70)

    best_by_metric = {}

    for metric in ["precision_at_10", "recall_at_10", "ndcg_at_10", "mrr"]:
        best_retriever = max(
            results.items(),
            key=lambda x: x[1].avg_metrics.get(metric, 0)
        )
        best_by_metric[metric] = best_retriever

    for metric, (name, result) in best_by_metric.items():
        value = result.avg_metrics.get(metric, 0)
        print(f"{metric:>20}: {name:<20} ({value:.3f})")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    semantic_result = results.get("SemanticRetriever")
    bm25_result = results.get("BM25Retriever")
    hybrid_result = results.get("HybridRetriever")

    if hybrid_result:
        hybrid_ndcg = hybrid_result.avg_metrics.get("ndcg_at_10", 0)
        semantic_ndcg = semantic_result.avg_metrics.get("ndcg_at_10", 0) if semantic_result else 0
        bm25_ndcg = bm25_result.avg_metrics.get("ndcg_at_10", 0) if bm25_result else 0

        print(f"\n✓ Hybrid Retriever combines semantic + lexical search")
        print(f"  NDCG@10: {hybrid_ndcg:.3f}")

        if hybrid_ndcg > max(semantic_ndcg, bm25_ndcg):
            print(f"  → Outperforms individual strategies")
        else:
            print(f"  → Consider adjusting weights or strategies")

    print("\n" + "="*70)
    print("KEY OBSERVATIONS")
    print("="*70)

    print("""
1. Semantic Retrieval: Good for understanding query intent
   - Captures meaning and context
   - May miss exact keyword matches

2. BM25 Retrieval: Excellent for exact term matching
   - Fast and efficient
   - May miss semantic relationships

3. Hybrid Retrieval: Best of both worlds
   - Uses Reciprocal Rank Fusion (RRF) to combine results
   - Higher recall through ensemble approach
   - Recommended for production systems
    """)

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
