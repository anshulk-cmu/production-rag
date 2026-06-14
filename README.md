# Production RAG System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Information Retrieval](https://img.shields.io/badge/IR-Information%20Retrieval-orange.svg)](https://github.com/anshulk-cmu/production-rag)
[![Semantic Search](https://img.shields.io/badge/Search-Semantic%20%2B%20BM25-blueviolet.svg)](https://github.com/anshulk-cmu/production-rag)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-quality Retrieval-Augmented Generation with multi-strategy retrieval and comprehensive evaluation framework.**

A complete implementation of enterprise-grade RAG retrieval systems that:
- ✅ Combine semantic and lexical search (hybrid retrieval)
- ✅ Optimize queries for better results
- ✅ Rerank with cross-encoders
- ✅ Evaluate with industry-standard metrics
- ✅ Compare retriever strategies systematically

## Architecture

### Core Components

```
User Query
    ↓
Query Analyzer (intent detection, keyword extraction)
    ↓
Query Optimizer (expansion, rewriting, stop word removal)
    ↓
Multi-Strategy Retriever (semantic + BM25 + hybrid)
    ↓
Cross-Encoder Reranker (relevance scoring)
    ↓
Top-K Results
```

### Retrieval Strategies

| Strategy | How | Best For | Tradeoff |
|----------|-----|----------|----------|
| **Semantic** | Vector similarity (embeddings) | Understanding intent | Slow, CPU intensive |
| **BM25** | TF-IDF lexical matching | Keywords, entities | Misses semantics |
| **Hybrid** | Combine both (RRF) | General use | More complex |

## Installation

```bash
# Clone repository
git clone https://github.com/anshulk-cmu/production-rag.git
cd production-rag

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

**Requirements:**
- Python 3.8+
- numpy (for embeddings and scoring)

## Quick Start

### 1. Basic Retrieval

```python
from rag import SemanticRetriever

# Create retriever
retriever = SemanticRetriever()

# Index documents
documents = {
    "doc_1": "Machine learning is...",
    "doc_2": "Deep learning uses...",
    # ... more documents
}
retriever.index_documents(documents)

# Retrieve
query = "What is machine learning?"
results = retriever.retrieve(query, k=5)

for result in results:
    print(f"Score: {result.score:.3f}, Content: {result.content[:100]}...")
```

### 2. Comparison: Different Strategies

```python
from rag import SemanticRetriever, BM25Retriever, HybridRetriever

retrievers = [
    SemanticRetriever(),
    BM25Retriever(),
    HybridRetriever(),
]

for retriever in retrievers:
    retriever.index_documents(documents)
    results = retriever.retrieve(query, k=5)
    print(f"{retriever.name}: Top result = {results[0].document_id}")
```

### 3. Full RAG Pipeline

```python
from rag import RAGPipeline, HybridRetriever

# Create pipeline
rag = RAGPipeline(
    retriever=HybridRetriever(),
    optimize_query=True,      # Query optimization
    rerank=True               # Reranking enabled
)

# Index documents
rag.index_documents(documents)

# Query
results = rag.query(
    query="Tell me about AI and machine learning",
    k=10,
    rerank_k=5  # Return top-5 after reranking
)

# Display
print(rag.format_results(results, include_metadata=True))
```

### 4. Evaluation & Benchmarking

```python
from rag import HybridRetriever
from evaluation import Evaluator

# Create test set (queries with ground truth)
test_queries = [
    ("machine learning", {"doc_1", "doc_5"}),
    ("neural networks", {"doc_2", "doc_7"}),
    # ... more test queries
]

# Evaluate
evaluator = Evaluator()
retriever = HybridRetriever()
retriever.index_documents(documents)

results = evaluator.evaluate(
    retriever=retriever,
    test_queries=test_queries,
    k_values=[1, 5, 10]
)

print(f"Precision@10: {results.avg_metrics['precision_at_10']:.3f}")
print(f"NDCG@10: {results.avg_metrics['ndcg_at_10']:.3f}")
```

## Running Examples

### Basic Examples

```bash
python examples/basic_rag.py
```

Demonstrates:
- Semantic retrieval
- BM25 retrieval
- Hybrid retrieval
- Full RAG pipeline
- Batch querying

### Evaluation Benchmark

```bash
python examples/evaluation_demo.py
```

Compares retrievers on test set with metrics:
- Precision@K (how many results are relevant)
- Recall@K (how many relevant items did we find)
- MRR (where is the first relevant result)
- NDCG@K (ranked relevance)
- MAP (average precision)

## Key Concepts

### Retrieval Strategies

**Semantic Retrieval**
- Embed query and documents into vectors
- Find similarity using cosine distance
- Captures meaning and intent
- Example: "car" matches "automobile"

**BM25 Retrieval**
- TF-IDF with term frequency saturation
- Fast lexical keyword matching
- Handles stop words and term importance
- Example: exact keyword "machine learning" matches

**Hybrid Retrieval**
- Combine semantic and BM25 using RRF (Reciprocal Rank Fusion)
- Gets high recall from both approaches
- Better precision than either alone
- Formula: score = 1/(k + rank_semantic) + 1/(k + rank_bm25)

### Query Optimization

1. **Query Expansion**: Add synonyms and related terms
2. **Query Rewriting**: Clarify ambiguous queries
3. **Stop Word Removal**: Focus on important terms
4. **Intent Detection**: Route to best retriever

### Reranking

After retrieving 100+ candidates, use cross-encoder to:
- Score relevance more precisely
- Identify truly relevant documents
- Return only top-k best matches
- Improves precision without hurting recall

### Evaluation Metrics

**Precision@K**: Of top-K results, how many are relevant?
```
P@10 = relevant_in_top_10 / 10
```

**Recall@K**: Of all relevant items, how many did we find?
```
R@10 = relevant_in_top_10 / total_relevant
```

**MRR**: Where is the first relevant result?
```
MRR = 1 / rank_of_first_relevant
Best = 1.0 (first result relevant), Worst = 0.0 (none relevant)
```

**NDCG@K**: Ranked relevance accounting for position
```
NDCG = DCG / IDCG
Higher scores = better ranking of relevant items
```

**MAP**: Average precision across queries
```
MAP = average(precision at each relevant item position)
```

## Project Structure

```
rag-system/
├── rag/                           # Core library
│   ├── core.py                   # RAG pipeline
│   ├── retrieval/
│   │   ├── base.py              # Abstract retriever
│   │   ├── semantic.py          # Semantic retrieval
│   │   ├── bm25_retriever.py   # BM25 lexical search
│   │   ├── hybrid.py            # Hybrid RRF
│   │   └── reranker.py          # Cross-encoder reranking
│   └── query/
│       ├── optimizer.py         # Query optimization
│       └── analyzer.py          # Query analysis
├── evaluation/
│   ├── metrics.py               # Evaluation metrics
│   └── evaluator.py             # Benchmark runner
├── examples/
│   ├── basic_rag.py            # Working examples
│   └── evaluation_demo.py       # Benchmark demo
├── README.md
├── pyproject.toml
└── LICENSE
```

## API Reference

### RAGPipeline

```python
from rag import RAGPipeline, HybridRetriever

pipeline = RAGPipeline(
    retriever=HybridRetriever(),
    optimize_query=True,
    rerank=True
)

# Index documents
pipeline.index_documents({"doc_id": "content", ...})

# Single query
results = pipeline.query(
    query="search query",
    k=10,                    # Retrieve top-10
    rerank_k=5              # Rerank to top-5
)

# Batch queries
results_list = pipeline.batch_query(
    queries=["query1", "query2", ...],
    k=10
)

# Get statistics
stats = pipeline.get_stats()
```

### Evaluation

```python
from evaluation import Evaluator

evaluator = Evaluator()

# Single retriever
result = evaluator.evaluate(
    retriever=retriever,
    test_queries=[(query, relevant_set), ...],
    k_values=[1, 5, 10]
)

# Compare multiple
results = evaluator.compare(
    retrievers=[semantic, bm25, hybrid],
    test_queries=test_queries
)

# Display
evaluator.print_comparison(results)
```

### Metrics

```python
from evaluation.metrics import compute_metrics

metrics = compute_metrics(
    retrieved_ids=["doc_1", "doc_5", "doc_3", ...],
    relevant_ids={"doc_1", "doc_2"},
    k_values=[1, 5, 10]
)

print(f"P@10: {metrics.precision_at_k[10]:.3f}")
print(f"R@10: {metrics.recall_at_k[10]:.3f}")
print(f"NDCG@10: {metrics.ndcg_at_k[10]:.3f}")
print(f"MRR: {metrics.mrr:.3f}")
```

## Performance Characteristics

**On 10 documents:**
- Semantic indexing: ~10ms
- BM25 indexing: ~5ms
- Semantic retrieve: ~2ms
- BM25 retrieve: ~1ms
- Hybrid retrieve: ~3ms
- Reranking: ~5ms

**Scales linearly:** 1000 documents = ~100ms retrieval

## Design Patterns

### 1. Strategy Pattern
Each retriever (Semantic, BM25, Hybrid) implements same interface:
```python
class BaseRetriever(ABC):
    def retrieve(self, query, k) -> List[RetrievalResult]
```
Easy to swap strategies without changing pipeline.

### 2. Pipeline Pattern
RAGPipeline orchestrates: query analysis → optimization → retrieval → reranking
Each stage is pluggable.

### 3. Composition over Inheritance
Hybrid combines Semantic + BM25 using composition, not inheritance.

## Production Considerations

### Scaling
- Cache embeddings for repeated queries
- Use approximate nearest neighbor search (HNSW, IVF)
- Batch queries for efficiency
- Implement query caching

### Monitoring
- Track retrieval quality metrics
- Monitor query latency
- Log failed retrievals
- A/B test retriever strategies

### Reliability
- Implement circuit breakers for external services
- Add retry logic
- Handle edge cases (empty queries, large result sets)
- Graceful degradation if components fail

### Optimization
- Hybrid gives best precision-recall tradeoff
- Reranking improves precision significantly
- Query optimization increases recall
- Experiment with weights for your domain

## Advanced Techniques

Not implemented (but easy to add):
- **Dense passage retrieval** - Fine-tuned retrievers
- **Approximate nearest neighbor** - HNSW, IVF for scale
- **Query understanding** - Named entity recognition, intent parsing
- **Multi-hop reasoning** - Chain retrievals for complex queries
- **Prompt engineering** - Optimize prompts for LLMs
- **Fusion with LLM** - Combine retrieval with language generation

## Resources

### Papers
- "BM25 Weighting for Information Retrieval" - Okapi BM25
- "Sentence-BERT: Semantic Textual Similarity" - Sentence embeddings
- "Dense Passage Retrieval for Open-Domain QA" - DPR
- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" - ColBERT

### Tools
- [sentence-transformers](https://www.sbert.net/) - Embeddings in production
- [Qdrant](https://qdrant.tech/) - Vector database
- [Weaviate](https://weaviate.io/) - Vector search
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API
- [LangChain](https://python.langchain.com/) - LLM orchestration

## Author

**Anshul Kumar**
- GitHub: [@anshulk-cmu](https://github.com/anshulk-cmu)
- Email: anshulk@andrew.cmu.edu

## License

MIT License - See LICENSE file

---

**Building production-quality RAG systems** 🚀

Questions? Check examples/ folder for working code.
