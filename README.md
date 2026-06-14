# production-rag

A retrieval-augmented generation library built from the ground up, with the components that are
usually hidden inside third-party packages implemented directly in readable Python. The BM25 ranker,
the rank fusion, and the evaluation metrics are all written out in the source.

The project began as a teaching codebase (credited below) and is being developed into a
production- and research-grade system: real open-source models from Hugging Face, rigorous
evaluation, and observability integrated from the start. Development and testing happen locally on a
12 GB GPU first; a hosted demonstration is the final step.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Current status

The retrieval plumbing is implemented and tested; the embedding and reranking models are still
placeholders pending replacement. This section states the current state precisely.

Implemented and verified: BM25 from scratch (tunable k1 and b), hybrid retrieval with reciprocal
rank fusion, and the standard information-retrieval metrics (precision, recall, MRR, NDCG). The
foundation added during the rebuild includes a single configuration layer that selects the entire
model stack through one environment variable, typed contracts for every swappable component, and an
observability layer (Prometheus metrics and structured logs) that every component reports into. The
local GPU budget was measured rather than estimated: bge-m3, bge-reranker-v2-m3, and Llama-3.2-3B
together occupy approximately 8.1 GB of a 12 GB card in bf16/fp16.

Placeholder, pending replacement in the next milestone: the semantic embedder currently returns
hash-based vectors rather than learned embeddings, and the reranker is a word-overlap heuristic
rather than a cross-encoder.

The complete roadmap is documented in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). It covers real
embeddings and reranking, document chunking and ingestion, grounded answer generation with
citations, contextual retrieval, a first-class knowledge-graph retriever, corrective and agentic
retrieval, persistent memory, an evaluation suite (a curated gold set and a BEIR subset), and a
Gradio demonstration on Hugging Face Spaces.

## Query flow

```
query
  -> query analyzer      (intent, keywords, entities)
  -> query optimizer     (rewrite, expand, drop stop words)
  -> retriever           (semantic | BM25 | hybrid)
  -> cross-encoder rerank
  -> top-k results
```

The three retrieval strategies share a single interface, so substituting one for another is a
one-line change.

| Strategy | Ranking method | Strengths | Limitations |
|----------|----------------|-----------|-------------|
| Semantic | vector cosine similarity | meaning, paraphrase | requires a real embedding model |
| BM25 | TF-IDF with term saturation | exact terms, names | does not capture meaning |
| Hybrid | reciprocal rank fusion of both | most queries | additional moving parts |

Hybrid fuses the two ranked lists with `score = 1/(k + rank_semantic) + 1/(k + rank_bm25)`. Because
fusion depends on rank position rather than raw scores, it is stable across retrievers whose scores
live on different scales.

## Installation

The project uses conda as its default environment manager.

```bash
conda activate production-rag     # Python 3.11, CUDA torch
pip install -e ".[dev]"           # core library and development tools
pytest -q                         # 27 tests
```

The core install is intentionally light (numpy, pydantic, prometheus-client). Heavier components are
provided through optional dependency groups: `embeddings, vectorstore, hosted, llm, loaders, graph,
serve, eval, gpu, monitoring, local, all`. For the full local GPU stack, install `".[local]"` with a
CUDA build of torch.

## Configuration

Settings are read from environment variables and a local `.env` file (see
[.env.example](.env.example)). Moving between a laptop GPU, a cloud instance, and a free Hugging Face
Space requires changing a single value, `RAG_PROFILE`:

| Profile | Target | Embedder | Reranker | LLM |
|---------|--------|----------|----------|-----|
| `local` | 12 GB GPU (default) | bge-m3 | bge-reranker-v2-m3 | Llama-3.2-3B-Instruct |
| `cloud` | AWS EC2 24/48 GB | bge-m3 | bge-reranker-v2-m3 | Llama-3.1-8B-Instruct |
| `zerogpu` | HF Space (GPU) | bge-m3 | bge-reranker-v2-m3 | Llama-3.2-3B-Instruct |
| `cpu` | HF Space (free CPU) | bge-small-en-v1.5 | ms-marco-MiniLM | Qwen2.5-1.5B-Instruct |
| `fake` | tests and CI | built-in | built-in | echo |

Weights always load in bf16/fp16 and are never weight-quantized. The Hugging Face token is read from
`HF_TOKEN` in `.env`; the earlier `HF-Token` spelling remains supported.

## Usage

Index documents and run the full pipeline:

```python
from rag import HybridRetriever, RAGPipeline

docs = {
    "doc_1": "Machine learning trains models on data instead of hand-written rules.",
    "doc_2": "BM25 ranks documents by term frequency with length normalization.",
}

rag = RAGPipeline(retriever=HybridRetriever())
rag.index_documents(docs)

results = rag.query("how does ranking work?", k=10, rerank=True, rerank_k=5)
print(rag.format_results(results))
```

Compare strategies on the same corpus:

```python
from rag import SemanticRetriever, BM25Retriever, HybridRetriever

for retriever in (SemanticRetriever(), BM25Retriever(), HybridRetriever()):
    retriever.index_documents(docs)
    top = retriever.retrieve("term frequency", k=5)
    print(retriever.name, "->", top[0].document_id)
```

Evaluate a retriever against a labeled set:

```python
from rag import HybridRetriever
from evaluation import Evaluator

test_queries = [("term frequency", {"doc_2"}), ("learning from data", {"doc_1"})]

retriever = HybridRetriever()
retriever.index_documents(docs)

result = Evaluator().evaluate(retriever, test_queries, k_values=[1, 5, 10])
print(result.avg_metrics["precision_at_10"], result.avg_metrics["ndcg_at_10"])
```

## Demos and tools

```bash
python examples/basic_rag.py         # semantic, BM25, hybrid, full pipeline, batch
python examples/evaluation_demo.py   # compares retrievers on a small labeled set
python scripts/gpu_budget.py         # loads the local stack and reports VRAM usage
```

## Observability

Every component logs and measures itself through `rag/observability`. Metrics are Prometheus
instruments: per-stage latency histograms (which yield p50, p95, and p99 rather than an average
alone), request and token counters, cache hit and miss counts, model-load times, index size,
evaluation scores, and GPU, RAM, and CPU usage. Logs are structured and can be shipped to Loki. The
metrics and logs are viewed in Grafana through the `infra/observability/` stack, with a terminal
table available for quick inspection.

## Repository layout

```
rag/
  config/         settings and model registry (the profile switch)
  observability/  logging, Prometheus metrics, watch
  interfaces/     typed contracts every component implements
  utils/          device helpers and GPU memory cleanup
  retrieval/      base, semantic, bm25, hybrid, reranker
  query/          analyzer, optimizer
  core.py         the pipeline that connects the stages
evaluation/       IR metrics and benchmark runner
examples/         runnable demos
scripts/          gpu_budget and related tools
tests/            unit tests
docs/             project plan and roadmap
```

The structure follows three deliberate choices. Retrievers sit behind one `BaseRetriever` interface,
so the pipeline does not depend on which strategy is in use. Hybrid is built by composing a semantic
retriever and a BM25 retriever rather than inheriting from either. The pipeline is a sequence of
pluggable stages, which makes the planned additions (real models, generation, a graph retriever) a
matter of implementing contracts rather than rewriting the core.

## Author

**Anshul Kumar**. [@anshulk-cmu](https://github.com/anshulk-cmu), anshulk@andrew.cmu.edu

## Acknowledgments

This project started from the original [production-rag](https://github.com/KazKozDev/production-rag)
codebase by **Artem Kazakov Kozlov ([@KazKozDev](https://github.com/KazKozDev))**, with thanks.

## License

MIT. See [LICENSE](LICENSE).
