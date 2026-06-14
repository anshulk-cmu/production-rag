#!/usr/bin/env python3
"""
Basic RAG example - Load documents, retrieve, and display results.

This demonstrates:
1. Indexing documents
2. Querying with different strategies
3. Displaying results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGPipeline, SemanticRetriever, BM25Retriever, HybridRetriever


# Sample documents
DOCUMENTS = {
    "doc_1": "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and models that enable computers to learn from and make predictions based on data.",
    "doc_2": "Deep learning uses neural networks with multiple layers (hence 'deep') to progressively extract higher level features from raw input. It has revolutionized computer vision and natural language processing.",
    "doc_3": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a meaningful way.",
    "doc_4": "Computer vision is a field of AI that trains computers to interpret and understand the visual world using digital images and deep learning models.",
    "doc_5": "Artificial intelligence encompasses technologies that enable machines to sense, comprehend, act, and learn in response to human instructions or environmental inputs.",
    "doc_6": "Data science combines statistics, programming, and domain expertise to extract meaningful insights from data. It's the foundation of modern AI systems.",
    "doc_7": "Neural networks are computing systems inspired by biological neurons that are interconnected. They're the basis of modern deep learning.",
    "doc_8": "Large language models like GPT are trained on massive amounts of text data and can generate human-like text, answer questions, and assist with various tasks.",
}


def example_1_basic_semantic():
    """Example 1: Basic semantic retrieval"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Semantic Retrieval")
    print("="*70)

    # Create retriever and index documents
    retriever = SemanticRetriever()
    retriever.index_documents(DOCUMENTS)

    # Query
    query = "What is machine learning?"
    print(f"\nQuery: {query}")

    results = retriever.retrieve(query, k=3)

    print("\nResults:")
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] Score: {result.score:.3f}")
        print(f"    {result.content[:100]}...")


def example_2_bm25_retrieval():
    """Example 2: BM25 lexical retrieval"""
    print("\n" + "="*70)
    print("EXAMPLE 2: BM25 Retrieval")
    print("="*70)

    retriever = BM25Retriever()
    retriever.index_documents(DOCUMENTS)

    query = "neural networks deep learning"
    print(f"\nQuery: {query}")

    results = retriever.retrieve(query, k=3)

    print("\nResults:")
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] Score: {result.score:.3f}")
        print(f"    {result.content[:100]}...")


def example_3_hybrid_retrieval():
    """Example 3: Hybrid retrieval combining semantic + BM25"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Hybrid Retrieval (Semantic + BM25)")
    print("="*70)

    retriever = HybridRetriever(semantic_weight=0.5)
    retriever.index_documents(DOCUMENTS)

    query = "How do computers learn from data?"
    print(f"\nQuery: {query}")

    results = retriever.retrieve(query, k=3)

    print("\nResults:")
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] Score: {result.score:.3f}")
        print(f"    {result.content[:100]}...")


def example_4_rag_pipeline():
    """Example 4: Full RAG pipeline with query optimization and reranking"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Full RAG Pipeline")
    print("="*70)

    # Create retriever
    retriever = HybridRetriever()

    # Create RAG pipeline
    rag = RAGPipeline(retriever=retriever)

    # Index documents
    rag.index_documents(DOCUMENTS)

    # Query
    query = "Tell me about artificial intelligence and machine learning"
    print(f"\nQuery: {query}")

    results = rag.query(
        query,
        k=10,
        optimize_query=True,
        rerank=True,
        rerank_k=3
    )

    print("\n" + "="*70)
    print("TOP RESULTS AFTER RERANKING")
    print("="*70)

    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] Relevance Score: {result.score:.3f}")
        print(f"Document ID: {result.document_id}")
        print(f"Content: {result.content[:150]}...")


def example_5_batch_query():
    """Example 5: Batch querying"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Query Processing")
    print("="*70)

    retriever = HybridRetriever()
    rag = RAGPipeline(retriever=retriever)
    rag.index_documents(DOCUMENTS)

    queries = [
        "machine learning algorithms",
        "neural networks",
        "natural language processing",
    ]

    print(f"\nProcessing {len(queries)} queries...")

    results_list = rag.batch_query(queries, k=5, rerank_k=3)

    for query, results in zip(queries, results_list):
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"Top result: {results[0].document_id} (score: {results[0].score:.3f})")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("PRODUCTION RAG SYSTEM - BASIC EXAMPLES")
    print("="*70)

    example_1_basic_semantic()
    example_2_bm25_retrieval()
    example_3_hybrid_retrieval()
    example_4_rag_pipeline()
    example_5_batch_query()

    print("\n" + "="*70)
    print("✓ ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
