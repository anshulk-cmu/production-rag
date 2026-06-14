# Master Plan — Production + Research-Grade Text-Only RAG (Hugging Face)

## Context — why this plan exists

`production-rag` today is an elegant **teaching skeleton**: a numpy-only, retrieval-only RAG
library with clean abstractions (Strategy/Pipeline/Composition) but **fake internals**. A
full line-by-line read confirmed the core problem: the "AI" is simulated.

- Embeddings are **hash-seeded random vectors** (`SimpleEmbedder`) — zero semantic meaning.
- The reranker is a **word-overlap heuristic** (`SimpleRerankerModel`) — not a real cross-encoder.
- There is **no chunking, no persistence, no LLM generation, no config, no tests, no UI**.
- The evaluation module has a **MAP bug** (reports NDCG@max_k as MAP) and unrecorded timings.

The goal is to evolve this into a **genuinely production- and research-grade, TEXT-ONLY RAG
system** that:
1. Uses the **best open-source models on Hugging Face** (no paid APIs — no OpenAI/Cohere/Anthropic).
2. Is **deployed on Hugging Face Spaces with Gradio** (free tier).
3. Is **as comprehensive as possible**, including a serious **evaluation & metrics** suite.
4. Adds modern techniques surfaced by research: **contextual retrieval, GraphRAG (first-class),
   query transformation (HyDE / multi-query / RAG-Fusion), LLM generation with citations, and
   agentic / corrective RAG (CRAG / Self-RAG / routing)**.

Intended outcome: a repo that reads as a credible portfolio/research artifact under Anshul's
identity — a working hosted demo, reproducible benchmarks, and clean engineering — with **no
mention of any AI assistant anywhere in code, commits, or docs** (hard project rule).

### User-confirmed decisions (2026-06-14)
- **Local-first = full-quality testing.** Primary development + runtime target is the user's **local
  machine with a 12GB VRAM CUDA GPU**, running the **best models at full quality** (fp16 weights;
  only the KV *cache* is quantized, near-lossless). Everything is built, run, and validated locally
  (M0–M6) BEFORE any Hugging Face push.
- **LLM = a 3B model (non-Qwen) + PolarQuant KV-cache quantization.** Default
  `meta-llama/Llama-3.2-3B-Instruct` (fp16 weights = full quality) with **PolarQuant** polar-
  transform KV-cache quantization (~4.2× cache compression, ~14% faster decode, near-lossless;
  supports Llama). This frees VRAM for longer contexts within 12GB. (Note: "SpectralQuant" did not
  surface as a distinct published method — closest is TurboQuant/ICLR'26, which builds on
  PolarQuant; using PolarQuant unless you mean another specific method.)
- **Cloud GPU testing via AWS EC2 (24GB / 48GB).** For higher-performance / larger-model testing,
  spin up a GPU EC2 instance using the **AWS CLI already configured in the `D:\golgi_vcc` project**
  (mirror its `infrastructure/launch-instances.sh` + `setup-vpc.sh` pattern). 24GB → `g5.xlarge`
  (A10G) / `g6.xlarge` (L4); 48GB → `g6e.xlarge` (L40S) / A6000. A `cloud` profile runs bigger
  models at full precision (e.g., 8B fp16) for performance comparison.
- **Persistence memory.** The RAG has a **persistent memory layer** (conversation history + durable
  extracted facts + retrievable past interactions) that survives restarts — local on disk, on HF in
  the Dataset. It augments retrieval with relevant prior context.
- **HF = demo only.** The Hugging Face Space is the **last step** (M7), a public demo, pushed only
  once the system builds and works locally. Its free-tier (CPU default + ZeroGPU opt-in) stack is a
  scaled-down mirror of the local stack, not the development target.
- **Serving profiles:** `local` (CUDA 12GB, full quality — **default for dev**) → `cloud` (AWS EC2
  24/48GB, bigger models) → `zerogpu` (HF demo upgrade) → `cpu` (HF free fallback) → `fake` (CI/tests).
- **GraphRAG:** **First-class feature** (core retrieval path, on by default) — construction runs at
  full quality on the local GPU, then is cached/persisted; cost mitigated by build-once + persist.
- **Evaluation:** **Bundled gold Q&A set + public BEIR subset** (recognized, citable retrieval
  numbers) plus full RAG generation metrics. Benchmarks are run locally on the GPU (or on EC2).
- **Scope:** maximally comprehensive.
- **Modality:** **text only** (no multimodal / images).

### HF connectivity & token — verified
A read-only `whoami-v2` check using the token in `.env` returned:
- **HF reachable:** yes (HTTP 200).
- **Token role:** `write` (classic token named `production-rag`) → can create repos, upload, push Spaces/Datasets.
- **HF username:** `anshul2048` (distinct from GitHub `anshulk-cmu`). So HF Space/Dataset repos live under `anshul2048/…`.
- An actual **write round-trip** (create temp repo → upload → delete) is deferred to **M0 task 1** because plan mode forbids external writes; the write *scope* is already confirmed.

---

## Top 10 improvements (headline changes)

| # | Improvement | Replaces / Adds | Why it matters |
|---|-------------|-----------------|----------------|
| 1 | **Real embeddings** behind an injected `Embedder` interface (bge-small CPU / bge-m3 ZeroGPU) | Fake `SimpleEmbedder` | The single biggest correctness fix — retrieval becomes actually semantic |
| 2 | **Real reranking** behind an injected `Reranker` (MiniLM CPU / bge-reranker-v2-m3 ZeroGPU) | Fake `SimpleRerankerModel` | Large precision lift on top-k ordering |
| 3 | **Ingestion + advanced chunking + loaders** (recursive/token/semantic/parent-child; txt/md/pdf/html text) | Nothing (docs were passed pre-split) | Real documents become usable; chunk quality drives everything downstream |
| 4 | **Vector store + persistence** (FAISS + HF-Dataset push/pull) | In-memory recompute every run | Survives HF's ephemeral disk; fast cold starts; scalable search |
| 5 | **Contextual Retrieval** (Anthropic technique, open LLM) — context-augmented chunks into dense **and** BM25 | Naive chunk indexing | ~49% fewer retrieval failures (67% with rerank) per published results |
| 6 | **GraphRAG (first-class)** — LightRAG/HippoRAG-style dual-level entity-relation graph + PageRank traversal, fused with vector + BM25 | No graph reasoning | Multi-hop / global questions; research-grade depth |
| 7 | **Query transformation** — HyDE, multi-query, **RAG-Fusion (proper RRF)** + fix dead strategy path | Naive synonym expansion + pairwise-average merge | Higher recall; principled fusion |
| 8 | **LLM answer generation with grounded citations** + prompt registry | Retrieval-only (no "G" in RAG) | Makes it an actual RAG product, not just a retriever |
| 9 | **Agentic / Corrective RAG** — CRAG grading + fallback, Self-RAG reflection, query routing | Static one-shot retrieval | Fewer hallucinations; adapts effort to query difficulty |
| 10 | **Comprehensive evaluation + Gradio HF Space** — fix MAP; IR metrics + RAGAS-style generation metrics (open judges); gold set + BEIR; latency/system; config/secrets; tests/CI | Buggy IR-only eval, no UI, no infra | Reproducible, citable, hosted, maintainable |

---

## Additional confirmed features (added after the top-10; part of the locked scope)

- **#11 Persistence memory** — `rag/memory/`: conversation history + durable fact extraction +
  retrievable past-interaction memory; persisted locally (SQLite + vector memory) and on HF (Dataset).
- **#12 PolarQuant KV-cache quantization** — applied to the local 3B LLM for ~4.2× smaller cache,
  faster decode, near-lossless; longer contexts within 12GB VRAM.
- **#13 AWS EC2 (24/48GB) cloud-GPU testing** — optional higher-tier test env via the AWS CLI from
  `D:\golgi_vcc`; bigger models at full precision.
- **#14 Free hosted vector DB option** — `QdrantVectorStore` (Qdrant Cloud free tier) behind the
  VectorStore ABC; alternative to local FAISS and a clean fix for HF ephemeral-disk persistence.

> **Scope is LOCKED.** This document is the **final scope of the project**. Work is tracked against
> these milestones; any scope change requires an explicit update to this plan file.

---

## Research findings (June 2026) that shaped this plan

The web research below drives the technique and model choices. Each finding maps to a Top-10 item.

**Embedding models (→ #1).** No single leaderboard winner; pick on workload. For an open,
HF-hosted, text RAG: `BAAI/bge-m3` is the standout for production because one model gives **dense +
sparse + ColBERT multi-vector** retrieval, 100+ languages, and 8k context — ideal for hybrid. Small
strong options for CPU: `bge-small/base-en-v1.5`, `gte-base`, `nomic-embed-text-v1.5`,
`Qwen3-Embedding-0.6B`. Top-of-leaderboard (NV-Embed-v2 7B, Qwen3-Embedding-8B) are too heavy for
12GB alongside an LLM. **Decision:** bge-m3 (local/zerogpu), bge-small (cpu).

**Rerankers (→ #2).** 2026 picks: `Qwen3-Reranker-{0.6B,4B,8B}` (Apache-2.0, 100+ lang, 32k),
`BAAI/bge-reranker-v2-m3` (strong, efficient, multilingual), `mxbai-rerank-v2` (Qwen2.5-based),
`jina-reranker-v3` (listwise, top BEIR nDCG), ColBERT for long docs. **Decision:**
bge-reranker-v2-m3 (local/zerogpu), MiniLM cross-encoder (cpu).

**Contextual Retrieval — Anthropic technique (→ #5).** Prepend an LLM-generated 1–2 sentence
situating context to each chunk **before** embedding AND before BM25 indexing. Reported retrieval-
failure reductions: 35% (contextual embeddings), 49% (+ contextual BM25), **67% (+ reranking)**.
We implement it with an **open** HF LLM (not the Anthropic API), at ingest time, cached.

**GraphRAG family (→ #6, first-class).** Microsoft **GraphRAG** builds entity-relation graphs +
LLM community summaries (powerful but LLM-expensive at build + query). **LightRAG** adds dual-level
(local entity + global relation) indexing for cheaper retrieval. **HippoRAG** uses Personalized
PageRank for 10–30× cheaper multi-hop. Consensus: graph value scales with query complexity /
multi-hop. **Decision:** dual-level + PageRank `GraphRetriever`, build-once + persist, fused with
vector+BM25 via RRF.

**Chunking (→ #3).** Baseline `RecursiveCharacterTextSplitter` at **400–512 tokens, 10–20%
overlap** is hard to beat; semantic chunking helps only sometimes and costs an embedding per
sentence; a 2026 analysis found a "context cliff" ~2,500 tokens. **Decision:** recursive default;
semantic + parent-child available behind config.

**Query transforms (→ #7).** HyDE (hypothetical answer → embed), multi-query (N paraphrases),
RAG-Fusion (multi-query + **RRF** over result lists), decomposition (sub-questions). These replace
the current naive synonym expansion + pairwise-average merge.

**Agentic / Corrective RAG (→ #9).** **CRAG** grades retrieved docs (correct/ambiguous/incorrect)
and falls back (rewrite / broaden / refuse) before generating. **Self-RAG** reflects (relevant?
supported? useful?) to gate generation and self-critique. **Agentic RAG** routes/plans multi-step
retrieval — reported to lift multi-hop accuracy dramatically over static RAG. **Decision:** CRAG +
Self-RAG + a query Router, all toggleable.

**Evaluation (→ #10).** **RAGAS** defines the standard generation metrics — **faithfulness, answer
relevancy, context precision, context recall** — and supports **custom open LLM + embedding
backends**, so we evaluate with **open HF judges, no paid keys**. Faithfulness can also be done with
a small open **NLI** model (cheaper, deterministic). Combine with classic IR metrics (P@k, R@k,
MRR, NDCG, MAP) on a gold set + a **BEIR** subset.

**HF free-tier hosting (→ deploy).** CPU Basic = **2 vCPU / 16GB RAM / 50GB ephemeral disk**;
**ZeroGPU** = dynamic **H200**, Gradio-only, `@spaces.GPU`, limited daily free minutes. Ephemeral
disk ⇒ persist the index/graph to a **private HF Dataset** and pull on boot. (Local dev has none of
these limits — full-speed 12GB GPU.)

### Sources
- [Best embedding models 2026 (Milvus)](https://milvus.io/blog/choose-embedding-model-rag-2026.md) · [MTEB leaderboard (Modal)](https://modal.com/blog/mteb-leaderboard-article) · [Open-source embeddings (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [Best reranker models 2026 (SiliconFlow)](https://www.siliconflow.com/articles/en/best-reranker-model-for-document-retrieval) · [Top rerankers for RAG (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/) · [jina-reranker-v3](https://jina.ai/models/jina-reranker-v3/)
- [Anthropic Contextual Retrieval (Claude cookbook)](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide) · [Contextual retrieval guide (DataCamp)](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)
- [Practical GraphRAG (arXiv 2507.03226)](https://arxiv.org/html/2507.03226v3) · [When to use Graphs in RAG (arXiv 2506.05690)](https://arxiv.org/html/2506.05690v3) · [GraphRAG vs HippoRAG vs PathRAG (Medium)](https://medium.com/graph-praxis/graphrag-vs-hipporag-vs-pathrag-vs-og-rag-choosing-the-right-architecture-for-your-knowledge-graph-a4745e8b125f)
- [Best chunking strategies 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) · [Chunking strategies (Weaviate)](https://weaviate.io/blog/chunking-strategies-for-rag)
- [RAG architectures 2025 (Humanloop)](https://humanloop.com/blog/rag-architectures) · [CRAG (EmergentMind)](https://www.emergentmind.com/topics/corrective-retrieval-augmented-generation-crag)
- [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) · [RAG eval metrics (Confident AI)](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [HF Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) · [Spaces ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu) · [Best open-source LLMs 2026 (HF blog)](https://huggingface.co/blog/daya-shankar/open-source-llms)

---

## Architecture — target package layout

Keep existing dirs (`rag/retrieval`, `rag/query`, `evaluation`, `examples`); extend with new
subpackages. Core stays numpy-only so the **fakes always import** even without optional deps;
heavy models live behind optional-dependency groups. Markers: (NEW) (REFACTOR) (KEEP).

```
production-rag/
├── app.py                         (NEW)  Gradio entrypoint — HF Space reads this
├── README.md                      (REFACTOR)  GitHub README (already re-branded to Anshul)
├── README_HF.md                   (NEW)  HF Space card (YAML frontmatter); copied to Space as README.md
├── requirements.txt               (REFACTOR)  Space/runtime install (CPU torch wheels)
├── requirements-dev.txt           (NEW)  optional pinned dev set
├── pyproject.toml                 (REFACTOR)  optional-deps groups + register new subpackages
├── .env                           (local only, gitignored)  HF_TOKEN=...
├── .env.example                   (NEW)  documents required env keys (no secret)
│
├── rag/
│   ├── __init__.py                (REFACTOR)  export new public API (keep old names working)
│   ├── core.py                    (REFACTOR)  add answer(); fix timing; keep query() contract
│   │
│   ├── config/                    (NEW)
│   │   ├── settings.py            RAGSettings (pydantic-settings) + get_settings() singleton
│   │   └── model_registry.py      curated HF model IDs + profile presets {local, cloud, zerogpu, cpu, fake}
│   │
│   ├── observability/             (NEW — built first in M0; used by every later component)
│   │   ├── logging.py            configure_logging(); named loggers; Loki handler + JSON fallback
│   │   ├── metrics.py            prometheus_client registry: @timed/counter/gauge/histogram; /metrics server
│   │   └── watch.py              live metrics table (CLI --watch; reused by the Gradio tab)
│   │
│   ├── interfaces/                (NEW)
│   │   └── protocols.py           Embedder, Reranker, VectorStore, Chunker, DocumentLoader,
│   │                              LLMClient, Generator, QueryTransform, RetrievalGrader, Router,
│   │                              GraphRetrieverProto  (typing.Protocol, @runtime_checkable)
│   │
│   ├── embeddings/                (NEW)
│   │   ├── fake.py                HashEmbedder (was SimpleEmbedder; deterministic md5 seed option)
│   │   ├── sentence_transformer.py  STEmbedder (bge-small/base, gte, nomic) — batched, normalized
│   │   ├── bge_m3.py              BGEM3Embedder (dense + sparse + ColBERT multi-vector) — ZeroGPU
│   │   └── hf_inference.py        HFInferenceEmbedder (feature-extraction API; zero local RAM)
│   │
│   ├── vectorstore/               (NEW)
│   │   ├── base.py                VectorStore ABC (add/search/save/load/__len__)
│   │   ├── numpy_store.py         NumpyVectorStore (zero-dep flat cosine; .npz)
│   │   ├── faiss_store.py         FaissVectorStore (IndexFlatIP → IVF/HNSW at scale; .faiss) — local default
│   │   ├── qdrant_store.py        QdrantVectorStore (free hosted Qdrant Cloud; same ABC) — hosted option
│   │   └── hf_persistence.py      push_index()/pull_index() to a PRIVATE HF Dataset repo
│   │
│   ├── loaders/                   (NEW)  TEXT-ONLY (extract text layer only)
│   │   ├── base.py                DocumentLoader ABC + Document dataclass
│   │   ├── text.py                TxtLoader, MarkdownLoader
│   │   ├── pdf.py                 PdfLoader (pypdf text only)
│   │   ├── html.py                HtmlLoader (BeautifulSoup text only)
│   │   └── directory.py           DirectoryLoader (glob dispatch by extension)
│   │
│   ├── chunking/                  (NEW)
│   │   ├── base.py                Chunker ABC + Chunk dataclass
│   │   ├── recursive.py           RecursiveCharacterChunker (400-512 tok, 10-20% overlap) — default
│   │   ├── token.py               TokenChunker (HF tokenizer-aware)
│   │   ├── semantic.py            SemanticChunker (embedding-distance breakpoints)
│   │   └── hierarchical.py        ParentChildChunker (small-to-big retrieval)
│   │
│   ├── ingestion/                 (NEW)
│   │   └── pipeline.py            IngestionPipeline: load → chunk → (contextualize) → embed →
│   │                              index → (build graph) → persist
│   │
│   ├── retrieval/                 (KEEP dir, REFACTOR files)
│   │   ├── base.py                (KEEP)  RetrievalResult, BaseRetriever — the universal contract
│   │   ├── semantic.py            (REFACTOR)  inject Embedder; vectorized matrix cosine; VectorStore
│   │   ├── bm25_retriever.py      (REFACTOR)  keep from-scratch algo; add serialize(); tokenizer hook
│   │   ├── hybrid.py              (REFACTOR)  inject sub-retrievers; fix score/sort; triple-fusion ready
│   │   ├── reranker.py            (REFACTOR)  inject Reranker model; keep fake fallback; batch fast path
│   │   ├── dense_store_retriever.py (NEW)  VectorStoreRetriever over a VectorStore backend
│   │   └── contextual.py          (NEW)  ContextualRetriever (index-time wrapper; Anthropic technique)
│   │
│   ├── graph/                     (NEW — FIRST-CLASS GraphRAG)
│   │   ├── extractor.py           EntityRelationExtractor (LLM-based; dual-level keywords)
│   │   ├── builder.py             KnowledgeGraphBuilder (networkx; entity dedup/merge; embeds nodes/edges)
│   │   ├── store.py               GraphStore (save/load graph + node/edge embeddings; persist to Hub)
│   │   ├── community.py           community detection + LLM community summaries (GraphRAG global)
│   │   └── retriever.py           GraphRetriever(BaseRetriever): dual-level (local entity / global
│   │                              community) + Personalized PageRank traversal (HippoRAG-style)
│   │
│   ├── llm/                       (NEW)
│   │   ├── base.py                LLMClient ABC (generate / stream)
│   │   ├── transformers_local.py  LocalTransformersLLM (CPU small / ZeroGPU bf16)
│   │   ├── hf_inference.py        HFInferenceLLM (InferenceClient.chat_completion)
│   │   └── echo.py                EchoLLM (deterministic fake for CI/tests)
│   │
│   ├── generation/               (NEW)
│   │   ├── prompts.py             PromptTemplate registry (qa, hyde, context_gen, grade,
│   │   │                          decompose, entity_extract, community_summary, fact_extract)
│   │   └── generator.py           Generator + GenerationResult (context packing, [n] citations, refusal)
│   │
│   ├── memory/                   (NEW — persistence memory)
│   │   ├── base.py               MemoryStore ABC (add_turn / get_history / search / persist / load)
│   │   ├── conversation.py       ConversationMemory (per-session history; SQLite-backed; survives restarts)
│   │   ├── semantic_memory.py    SemanticMemory (vector store of past turns/facts; retrievable as context)
│   │   └── facts.py              FactExtractor (LLM distills durable user/world facts from turns)
│   │
│   ├── query/                    (KEEP dir, REFACTOR + extend)
│   │   ├── analyzer.py            (REFACTOR)  align suggest_strategy with a shared Strategy enum
│   │   ├── optimizer.py           (REFACTOR)  map all strategies; explicit fallback (no silent no-op)
│   │   └── transforms/            (NEW)
│   │       ├── base.py            QueryTransform ABC
│   │       ├── hyde.py            HyDETransform (hypothetical doc → embedding)
│   │       ├── multi_query.py     MultiQueryTransform (N paraphrases)
│   │       ├── rag_fusion.py      RAGFusion (multi-query + RRF over result lists)
│   │       └── decomposition.py   DecompositionTransform (sub-questions)
│   │
│   ├── grading/                  (NEW)
│   │   ├── base.py               RetrievalGrader ABC + GradeResult
│   │   ├── crag.py               CRAGGrader (correct/ambiguous/incorrect + use/refine/discard)
│   │   └── self_rag.py           SelfRAGGrader (relevance + support + utility reflection)
│   │
│   ├── routing/                  (NEW — agentic)
│   │   └── router.py             Router ABC + HeuristicRouter (default) + LLMRouter (opt-in)
│   │
│   └── utils/                    (NEW)
│       ├── device.py             cpu/cuda detection, dtype, ZeroGPU-safe import of `spaces`
│       ├── timing.py             perf timers feeding retrieval_stats
│       └── caching.py            model + embedding cache helpers
│
├── evaluation/                  (KEEP dir, REFACTOR + extend)
│   ├── metrics.py               (REFACTOR)  FIX MAP; add per-query AP helper; graded NDCG; Hit@k
│   ├── retrieval_eval.py        (REFACTOR of evaluator.py)  add latency capture
│   ├── generation_metrics.py    (NEW)  faithfulness, answer relevancy, context P/R, answer correctness
│   ├── judges.py                (NEW)  open-model judges: NLIJudge + LLMJudge (RAGAS w/ HF backend)
│   ├── system_metrics.py        (NEW)  latency/throughput/tokens/peak-RAM/GPU-seconds
│   ├── pipeline_evaluator.py    (NEW)  end-to-end eval (retrieval + generation + system)
│   ├── datasets.py              (NEW)  gold-set loader + BEIR adapter (SciFact/NFCorpus subsets)
│   ├── report.py                (NEW)  JSON + Markdown emitters; config-diff comparison tables
│   ├── run.py                   (NEW)  CLI: python -m evaluation.run
│   └── configs/                 (NEW)  named pipeline configs to benchmark/compare
│
├── examples/                    (KEEP; add additive demos)
│   ├── basic_rag.py             (KEEP — back-compat proof)
│   ├── evaluation_demo.py       (KEEP — back-compat proof)
│   ├── generation_demo.py       (NEW)  ingest → chunk → embed → answer()
│   └── advanced_rag_demo.py     (NEW)  contextual + graph + RAG-fusion + CRAG
│
├── app/                         (NEW)  Gradio UI modules
│   ├── state.py                 lazy singletons; boot index/graph restore from Hub
│   ├── ui_query.py              ask tab: answer + citations + retrieved chunks + graph context
│   ├── ui_ingest.py             upload/ingest tab (session index over base index)
│   └── ui_eval.py               eval dashboard tab (capped live metrics)
│
├── data/
│   ├── corpus/                  bundled sample corpus (text)
│   ├── gold/qa.jsonl            hand-curated gold Q&A set
│   └── index/                   prebuilt FAISS + graph + manifest (committed or pulled from Hub)
│
├── docs/
│   ├── ARCHITECTURE.md          (NEW)  component diagram, data flow, CPU/ZeroGPU split, persistence
│   ├── EVALUATION.md            (NEW)  metric definitions (corrected MAP), judge methodology, schema
│   └── MODEL_CARD.md            (NEW)  model choices + measured results + limitations + repro cmd
│
├── tests/                       (NEW — the missing suite)
│   ├── conftest.py              fixtures: tiny corpus, gold set, deterministic fake models
│   ├── unit/                    per-module tests (metrics/bm25/chunker/vectorstore/transforms/config/graph)
│   ├── integration/             pipeline end-to-end (tiny LLM); CRAG fallback; graph retrieval
│   └── eval/                    eval smoke test (3-query gold set → report files)
│
├── infra/
│   ├── observability/          (NEW)  Grafana monitoring stack (local + EC2)
│   │   ├── docker-compose.yml   Prometheus + Loki + Grafana + Alloy/Promtail
│   │   ├── prometheus.yml       scrape config (app /metrics, pushgateway)
│   │   ├── loki-config.yml      Loki + Alloy/Promtail log pipeline
│   │   └── grafana/            provisioned datasources + dashboards (JSON): perf, quality, logs
│   └── aws/                    (NEW)  EC2 GPU test env — mirrors the D:\golgi_vcc AWS-CLI pattern
│       ├── setup-vpc.sh        VPC + subnet + security group + SSH key pair
│       ├── launch-gpu.sh       aws ec2 run-instances (g5/g6 = 24GB, g6e = 48GB; Deep Learning AMI)
│       ├── bootstrap.sh        install CUDA torch + project, pull repo, run benchmarks
│       └── teardown.sh         terminate instances + cleanup
│
└── .github/workflows/ci.yml     (NEW)  ruff + black + mypy + pytest(+cov), py3.10/3.11
```

**Layering (no cycles):** `interfaces` ← `embeddings|vectorstore|chunking|loaders|llm|graph` ←
`retrieval|generation|query.transforms|grading|routing` ← `core` ← `app`. `evaluation` depends
only on `retrieval`/`core` public types. Heavy modules are lazy-imported behind optional deps.

---

## Model stack (open-source, Hugging Face only)

**Default dev profile is `local`** (CUDA 12GB, full quality, full speed). `RAG_PROFILE` selects the profile;
resolved in `rag/config/model_registry.py`. The HF demo uses `zerogpu`/`cpu`.

| Role | **`local` (CUDA 12GB — primary, full quality)** | `zerogpu` (HF demo, H200) | `cpu` (HF free fallback) | `fake` (CI/tests) |
|------|-----------------------------------------------|----------------------------|---------------------------|-------------------|
| Embedder | `BAAI/bge-m3` on GPU (dense+sparse+ColBERT, 8k) | `BAAI/bge-m3` | `BAAI/bge-small-en-v1.5` (384d) | `HashEmbedder` |
| Reranker | `BAAI/bge-reranker-v2-m3` on GPU | `BAAI/bge-reranker-v2-m3` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `SimpleRerankerModel` |
| LLM (gen + graph + judge) | `meta-llama/Llama-3.2-3B-Instruct` **fp16 + PolarQuant KV cache** | `Llama-3.2-3B-Instruct` bf16 | `Qwen/Qwen2.5-1.5B-Instruct` (slow) | `EchoLLM` |
| Faithfulness judge | `cross-encoder/nli-deberta-v3-base` on GPU + 3B LLM judge | same | NLI only | NLI or stub |

**`cloud` profile (AWS EC2 24/48GB):** same components at **full precision, no quantization** —
e.g., `Llama-3.1-8B-Instruct` fp16 + `bge-m3` + `bge-reranker-v2-m3`, for performance comparison
against the local 12GB run. Provisioned via the AWS CLI (see "AWS EC2 testing" below).

**12GB VRAM budget (local stack), full quality:** load embedder + reranker + a **3B LLM at
full-quality fp16 weights**; only the KV *cache* is quantized (PolarQuant, near-lossless).
- **LLM:** `Llama-3.2-3B-Instruct` **fp16 ≈ ~6.5GB** (full quality), with **PolarQuant** shrinking
  the KV cache ~4.2× so long contexts fit without spilling. (4-bit weights available via
  `RAG_QUANTIZATION` if VRAM is tight, but fp16 is the default for full-quality local testing.)
- **Embedder:** `bge-m3` ≈ **~2.3GB**.
- **Reranker:** `bge-reranker-v2-m3` ≈ **~2.3GB**, or lazy/CPU via `RAG_RERANKER_DEVICE={cuda,cpu}`.
- **Total:** ~6.5 + 2.3 + 2.3 ≈ **~11.1GB** → fits 12GB; PolarQuant preserves headroom for KV +
  activations. Flags: `RAG_QUANTIZATION={none,4bit,8bit}`, `RAG_KV_QUANT={polarquant,none}`,
  `RAG_RERANKER_DEVICE={cuda,cpu}`.
- **Cloud escape hatch (`cloud` profile, EC2 24/48GB):** drop quantization; run bigger models at
  full precision (e.g., Llama-3.1-8B fp16) for a performance/quality ceiling comparison.
- GraphRAG entity/relation extraction and contextual-retrieval context generation run on the local
  **GPU at full quality** — this is exactly why local-first matters (those LLM-heavy ingest steps
  are painful on CPU / quota-limited on ZeroGPU).

**16GB-RAM CPU budget (HF `cpu` fallback only):** ~2GB baseline + 0.3 (embed) + 0.1 (rerank) + ~6
(1.5B LLM fp32) + 0.8 (NLI) ≈ **~9GB**. bge-m3 + 7B are reserved for `local`/`zerogpu`.

**ZeroGPU rules (HF demo):** decorate only **leaf compute** functions (`@spaces.GPU(duration=...)`
on encode / rerank / generate / extract), batch aggressively (daily minutes are the budget), warm-
cache models, and keep the `spaces` import behind a no-op shim so the library runs locally/CI
without it. The **same leaf functions** run un-decorated at full speed under the `local` profile.

---

## Cross-cutting decisions & justifications

- **FAISS over Chroma.** HF free disk is **ephemeral** (50GB, wiped on restart) and there's no
  persistent DB process. FAISS serializes to a single file → trivial push/pull to a private HF
  **Dataset** repo (`anshul2048/production-rag-index`). `IndexFlatIP` on normalized vectors ==
  exact cosine (matches current semantics); upgrade to IVF/HNSW only at scale. Default backend
  is the zero-dep `NumpyVectorStore`; FAISS is opt-in via the `vectorstore` extra.
- **Free hosted vector DB option (`QdrantVectorStore`).** Behind the same `VectorStore` ABC, a
  **Qdrant Cloud free-tier** backend (`hosted` extra; `QDRANT_URL`/`QDRANT_API_KEY` in env) gives
  durable, server-side persistence — the cleanest fix for HF's ephemeral disk (no push/pull dance)
  and a drop-in store for both the index and semantic memory. Local dev defaults to FAISS for
  full-quality offline runs; the HF demo can use FAISS+Dataset **or** hosted Qdrant.
- **Persistence model (solves ephemeral disk).** Index dir = `vectors.faiss` + `bm25.json` +
  `docstore.json(.gz)` + `graph.pkl` + `manifest.json` (embedder id, dim, chunker params, corpus
  hash, build time). **Boot logic:** `pull_index` from Hub → if present and manifest matches the
  current embedder, load and skip re-embedding; else rebuild from bundled corpus and push back.
  User uploads embed into a session index layered over the base index.
- **Keep the from-scratch BM25** (do NOT add `rank_bm25`). It's correct, dependency-free, and a
  deliberate from-scratch IR showcase central to the project's identity. Pin its scores in tests.
- **GraphRAG cost mitigation.** Entity/relation extraction + community summaries are many LLM
  calls — done **once** at ingest for the bundled corpus, cached in `graph.pkl`, persisted to the
  Hub. Runtime traversal (PageRank / dual-level) is cheap CPU. Uploaded docs trigger bounded
  graph extension with a size cap + UI warning.
- **Backward compatibility.** `from rag import RAGPipeline, SemanticRetriever, BM25Retriever,
  HybridRetriever` keeps working; `query()` keeps returning `List[RetrievalResult]`. New power is
  additive: `answer()` returns a `GenerationResult`; all new collaborators are injected, default
  to fakes/None.
- **Python pin.** Project currently allows 3.8; HF Spaces + torch + `spaces` are happiest on
  **3.10/3.11**. Pin `requires-python = ">=3.10"`; **default env = conda `production-rag` (Python
  3.11)**; CI matrix 3.10 + 3.11.
- **Secrets.** Rename `.env` key `HF-Token` → **`HF_TOKEN`** (hyphen is a non-standard env id);
  add a one-line back-compat shim mapping the old key. On the Space set `HF_TOKEN` as a **Space
  secret** (never commit `.env`, already gitignored). Gated models (Llama) need license
  acceptance on the `anshul2048` account.

---

## AWS EC2 cloud-GPU testing (golgi pattern)

For higher-performance / larger-model testing beyond the 12GB local card, spin up a GPU EC2
instance with the **AWS CLI already configured in `D:\golgi_vcc`** (its
`infrastructure/launch-instances.sh` shows the exact pattern: `aws ec2 describe-images` for a recent
AMI, `aws ec2 run-instances` with key pair + security group + subnet, `aws ec2 wait
instance-running`, then `describe-instances` for IPs; region us-east-1; teardown script).

`infra/aws/` mirrors this for GPU:
- **Instance types:** 24GB → `g5.xlarge` (A10G) or `g6.xlarge` (L4); 48GB → `g6e.xlarge` (L40S).
- **AMI:** AWS **Deep Learning AMI (Ubuntu, CUDA)** instead of Amazon Linux, so torch/CUDA work out
  of the box.
- **`launch-gpu.sh`** parameterizes instance type + (optional) `--instance-market-options` spot to
  cut cost; **`bootstrap.sh`** clones the repo, `pip install -e ".[local,...]"`, pulls models, runs
  `python -m evaluation.run` with `RAG_PROFILE=cloud`; **`teardown.sh`** terminates everything.
- **Discipline:** GPU instances bill per hour — `teardown.sh` immediately after each run; never
  commit AWS keys (use the existing CLI profile / IAM role from golgi).
- **Purpose:** establish a full-precision quality/perf ceiling (e.g., Llama-3.1-8B fp16) to compare
  against the quantized local 12GB run; numbers feed the eval comparison table. Optional tier.

## The four bug fixes (folded into the milestones)

1. **MAP** (`evaluation/metrics.py`): `compute_metrics` sets `map_score = ndcg_at_k[max_k]`. Extract
   `compute_average_precision(retrieved, relevant)` and use it; keep `compute_map` for corpus-level.
   Regression test asserts `map_score != ndcg` on a crafted case. → **M0**.
2. **Dead strategy path**: `QueryAnalyzer.suggest_strategy` returns `hybrid/semantic` that
   `QueryOptimizer.optimize` ignores. Introduce a shared `Strategy` enum; split the two axes
   (retriever-selection vs query-rewrite); explicit logged fallback for unknown values. → **M0/M5**.
3. **Unrecorded timings**: wrap retrieve/rerank/generate in `perf_counter`, increment the existing
   `retrieval_stats` fields, add `total_generation_time`; `get_stats()` returns averages. → **M0/M4**.
4. **Hybrid score/sort mismatch**: stored `score` must equal the sort key. `fusion="rrf"` (default)
   stores the RRF score (per-leg scores/ranks in metadata); `fusion="weighted_norm"` min-max
   normalizes each leg before the weighted average. → **M1**.

---

## Dependencies

**Two install paths** (same codebase):
- **Local dev (primary):** CUDA torch build + quantization libs for full-speed 12GB GPU —
  `pip install -e ".[local,embeddings,vectorstore,llm,loaders,graph,eval,dev]"` with torch from the
  CUDA wheel index (e.g. `--index-url https://download.pytorch.org/whl/cu121`). `bitsandbytes`
  enables 4-bit; `accelerate` enables `device_map`.
- **HF Space (demo):** CPU torch wheels via `requirements.txt` (below); `spaces` provides ZeroGPU.

A new `local` optional group pins the GPU dev set:
```
local = ["torch>=2.2", "accelerate>=0.33", "bitsandbytes>=0.43",
         "sentence-transformers>=3.0", "FlagEmbedding>=1.2", "faiss-cpu>=1.8",
         "transformers>=4.44", "networkx>=3.3"]   # install torch from the CUDA wheel index
```

Hosted vector DB + monitoring extras + notes:
```
hosted     = ["qdrant-client>=1.10"]                         # free Qdrant Cloud
monitoring = ["prometheus-client>=0.20", "python-logging-loki>=0.3.1"]   # Grafana: Prometheus + Loki
```
The Grafana/Prometheus/Loki **services** run via `infra/observability/docker-compose.yml` (Docker,
free/OSS) — no Python deps for the servers; the `monitoring` extra only adds the client libraries the
app uses to expose `/metrics` and ship logs.
- **PolarQuant** is not on PyPI — vendor it or `pip install git+<polarquant repo>` behind the
  `local`/`gpu` path; gracefully fall back to KVQuant/KIVI or no KV-quant if unavailable.
- **Memory** uses stdlib `sqlite3` (no new dep) + the existing VectorStore for semantic memory.
- **AWS** uses the **AWS CLI already configured for `D:\golgi_vcc`** (no `boto3` needed).

`requirements.txt` (HF Space install; CPU torch wheels):
```
numpy>=1.26
pydantic>=2.7
pydantic-settings>=2.3
python-dotenv>=1.0
torch>=2.2
transformers>=4.44
sentence-transformers>=3.0
faiss-cpu>=1.8
huggingface_hub>=0.24
datasets>=2.20
networkx>=3.3
gradio>=4.44
spaces>=0.30
```
`pyproject.toml` optional groups (keep core light; install per environment):
```
[project.optional-dependencies]
embeddings = ["sentence-transformers>=3.0", "FlagEmbedding>=1.2"]
vectorstore = ["faiss-cpu>=1.8", "huggingface_hub>=0.24"]
llm = ["transformers>=4.44", "accelerate>=0.33", "huggingface_hub>=0.24"]
loaders = ["pypdf>=4.2", "beautifulsoup4>=4.12"]
graph = ["networkx>=3.3"]
serve = ["gradio>=4.44", "spaces>=0.30"]
eval = ["ragas>=0.2", "datasets>=2.20", "pandas>=2.2", "scikit-learn>=1.5"]
gpu = ["accelerate>=0.33", "bitsandbytes>=0.43"]
dev = ["pytest>=8", "pytest-cov>=5", "black>=24", "mypy>=1.10", "ruff>=0.5"]
all = [ ...union... ]
```
Also update `[tool.setuptools] packages` to register every new subpackage (`rag.config`,
`rag.interfaces`, `rag.embeddings`, `rag.vectorstore`, `rag.loaders`, `rag.chunking`,
`rag.ingestion`, `rag.graph`, `rag.llm`, `rag.generation`, `rag.query.transforms`,
`rag.grading`, `rag.routing`, `rag.utils`).

---

## Engineering standards & incremental build methodology

**Build one component at a time. Each component is fully finished — structure, logging, metrics,
tests, docs — before the next one starts.** No half-built modules; the repo stays green and runnable
at every step. A "component" = one module with a clear interface (e.g., the embedder, the FAISS
store, the chunker, the CRAG grader, memory).

**RULE — minimal, simple code and minimal, simple comments.** Always prefer the smallest, clearest
implementation that satisfies the component's contract. No speculative abstraction, no clever
one-liners, no over-engineering. Comments are sparse and explain only *why* (non-obvious intent),
never restate *what* the code already says — prefer readable plain code over comments. This applies
to every component and every commit.

**Definition of Done (per component) — all required before moving on:**
1. **Structure** — lives in its own module; implements its `Protocol`/ABC from `rag/interfaces`;
   exported via the package `__init__`; registered in `rag/config` where it's selectable; consistent
   file layout (`base.py` interface → `impl.py` → tests).
2. **Logging** — uses a module-level named logger from the central `rag/observability/logging.py`
   (`logging.getLogger(__name__)`); structured, leveled (DEBUG build details / INFO lifecycle /
   WARNING fallbacks / ERROR failures); **no `print()` in library code** (prints only in
   `examples/` and CLIs). Optional JSON formatter for machine-readable logs.
3. **Watch / observability** — emits per-call metrics to the central `rag/observability/metrics.py`
   registry (timers, counters, gauges): latency, call counts, cache-hit rate, batch sizes, token
   counts, VRAM/RAM. A `--watch` mode (CLI + Gradio) renders a live table of these; values also flow
   into `RAGPipeline.get_stats()` and the eval `system_metrics`.
4. **Metrics correctness** — where the component has quality metrics (retrievers, generation), it is
   wired into the evaluation harness so its effect is measurable, not asserted.
5. **Tests** — unit tests (happy path + edge cases + failure/fallback) with deterministic fakes;
   added to CI; coverage stays above the gate; an integration test when it joins a slice.
6. **Docs** — docstrings on public API; a short note in `docs/` / README when user-facing.

**Central observability module (`rag/observability/`, built in M0 so every later component uses it).
Everything — metrics, logs, traces, watch/monitors — is viewable in Grafana.**
- `logging.py` — `configure_logging(level, json=False)`; named-logger helper; single place to set
  format/handlers; respects `RAG_LOG_LEVEL` / `RAG_LOG_JSON`. Ships logs to **Loki** (Grafana's log
  store) via a `LokiHandler` (`python-logging-loki`) when `RAG_LOKI_URL` is set, with a structured
  **JSON-to-file** fallback that Grafana **Alloy/Promtail** tails into Loki.
- `metrics.py` — registry wrapping **`prometheus_client`** (Counter/Gauge/Histogram): `@timed("stage")`
  decorator + context manager, `counter()`, `gauge()`, `histogram()`, plus a stdlib snapshot for
  `get_stats()`/reports. `start_metrics_server(port)` exposes a **Prometheus `/metrics`** endpoint
  (or mount it in the Gradio app) for Prometheus to scrape. Metrics captured for every stage:
  retrieval/rerank/generate/graph/memory latency, call counts, cache-hit rate, batch/token counts,
  VRAM/RAM, GPU seconds.
- `watch.py` — live metrics table (CLI `--watch`; reused by the Gradio observability tab) for a
  no-Grafana quick view; the same metrics also flow to Prometheus.

**Grafana stack (local + EC2; `infra/observability/`):** a `docker-compose.yml` brings up
**Prometheus + Loki + Grafana (+ Alloy/Promtail)**, all OSS/free. Grafana is **provisioned as code**:
datasources (Prometheus, Loki) and pre-built dashboards (JSON in
`infra/observability/grafana/dashboards/`) for (a) **RAG performance** (per-stage latency, throughput,
cache hits, tokens, VRAM), (b) **retrieval/generation quality** (eval metrics pushed via Pushgateway
after `evaluation.run`), and (c) **logs** (Loki log panels filtered by component/level). `make
observability-up` / `-down` wrap compose. On the **HF demo** (no docker-compose), the in-app
observability tab shows the `watch` table, and optionally pushes to **Grafana Cloud free tier** if
`RAG_LOKI_URL`/`RAG_PROM_REMOTE_WRITE` are configured as Space secrets.

**Component build loop (applied to every item in the milestones):**
`interface → implementation → structured logging → metrics instrumentation → unit tests (green) →
wire into config/pipeline → integration/smoke → one conventional commit`. Then the next component.

**Per-component commit granularity:** each finished component is its own conventional commit (no AI
attribution, no Co-Authored-By), so history reads as a clean component-by-component build.

## Milestones, tasks, files, acceptance, commits

Conventional commits, **no AI attribution, no Co-Authored-By trailer**. Work on a branch per
milestone off `main`. Push to `origin` (GitHub) and the `hf` remote only when the user says go.
The repo stays runnable and test-green at every milestone.

**Local-first workflow:** M0–M6 are developed, run, and validated entirely on the **local 12GB CUDA
GPU at full quality** (`RAG_PROFILE=local`). The HF Space (M7) is the final **demo** step, attempted
only after the system builds and works locally and benchmarks are green. HF **write scope is already
verified**; the actual create/upload round-trip runs in M7 (not earlier), since HF is deploy-time.

### M0 — Foundations (local GPU env, config, secrets, deps, tests scaffold, bug fixes, interfaces)
**Tasks**
- [x] **Local GPU environment setup** (DONE): **conda is the default env manager** (not venv/pip).
  Env `production-rag` (renamed from `hdai`) — Python 3.11.15, torch 2.10.0+cu128, CUDA 12.8
  available, NVIDIA RTX 5070 Ti Laptop GPU (11.94 GB VRAM, cc 12.0). Remaining sub-step: quick load
  test of bge-m3 + bge-reranker-v2-m3 + Llama-3.2-3B-Instruct (fp16 + PolarQuant) to confirm the VRAM budget.
- [ ] Rename `.env` key `HF-Token` → `HF_TOKEN` (+ back-compat shim); add `.env.example`.
  (HF reachability + `write` scope already verified; full write round-trip deferred to M7.)
- [ ] `rag/config/settings.py` (`RAGSettings` via pydantic-settings; `.env` load; `get_settings()`)
  and `rag/config/model_registry.py` (profile presets cpu/zerogpu/fake).
- [ ] Rewrite `requirements.txt`; add `pyproject` optional-deps groups; pin `requires-python>=3.10`;
  register new subpackages; add `ruff`.
- [ ] `rag/interfaces/protocols.py` (all Protocols, `@runtime_checkable`).
- [ ] **`rag/observability/`** (logging + Prometheus metrics + Loki log shipping + watch) — built
  first so every later component logs/instruments through it; unit-tested; wired into `get_stats()`.
- [ ] **`infra/observability/`** Grafana stack: `docker-compose.yml` (Prometheus + Loki + Grafana +
  Alloy/Promtail) with provisioned datasources + dashboards; `make observability-up`. Verify the
  app's `/metrics` is scraped and logs appear in Grafana before building further components.
- [ ] Move `SimpleEmbedder` → `rag/embeddings/fake.py` `HashEmbedder` (keep alias + DeprecationWarning;
  add deterministic md5-seed option + `embed_batch`).
- [ ] **Fix bugs 1 & part of 2/3**: MAP + AP helper in `metrics.py`; shared `Strategy` enum stub;
  add `perf_counter` timing scaffolding in `core.py`.
- [ ] `tests/` scaffold + `conftest.py`; tests locking current behavior + MAP regression.
**Files:** `.env(.example)`, `rag/config/*`, `rag/interfaces/protocols.py`, `rag/embeddings/fake.py`,
`evaluation/metrics.py`, `rag/core.py`, `requirements.txt`, `pyproject.toml`, `tests/*`.
**Acceptance:** `pytest -m "not slow"` green; MAP test proves real AP ≠ NDCG; `black`/`ruff`/`mypy`
clean; `get_settings()` loads `HF_TOKEN`; HF write round-trip succeeds and cleans up.
**Commits:** `chore: pin python and restructure dependencies into extras` · `feat(config): add
settings and model registry with local/cloud/zerogpu/cpu profiles` · `feat(observability): add
structured logging and metrics registry` · `feat(interfaces): add component protocols` ·
`refactor(embeddings): relocate hash embedder behind embedder protocol` · `fix(evaluation): compute
true mean average precision instead of ndcg` · `test: scaffold suite and lock metric regressions`.

### M1 — Real embeddings + reranker + vector store + persistence
**Tasks**
- [ ] `rag/embeddings/sentence_transformer.py` (STEmbedder, batched, normalized) + `bge_m3.py` +
  `hf_inference.py`.
- [ ] Refactor `SemanticRetriever(embedder=…, vector_store=…)`; vectorized matrix cosine; default
  `HashEmbedder` (back-compat).
- [ ] Refactor `CrossEncoderReranker(model=…)`; add real `STCrossEncoderReranker`/`HFRerankerModel`;
  batch fast path; keep fake default.
- [ ] Refactor `HybridRetriever(semantic=…, bm25=…, fusion="rrf")`; **fix bug 4** (score==sort key);
  ensure the real embedder is shared into the dense leg (not silently fake).
- [ ] `rag/vectorstore/` (`NumpyVectorStore`, `FaissVectorStore`, base ABC) + `hf_persistence.py`
  (push/pull to private HF Dataset; manifest validation).
- [ ] `rag/vectorstore/qdrant_store.py` (`QdrantVectorStore`, free Qdrant Cloud; same ABC; env-configured)
  — optional hosted backend; selectable via `RAG_VECTORSTORE={faiss,numpy,qdrant}`.
- [ ] BM25 `serialize()`/`load()` + tokenizer hook.
**Files:** `rag/embeddings/*`, `rag/retrieval/{semantic,hybrid,reranker,bm25_retriever,dense_store_retriever}.py`,
`rag/vectorstore/*`, tests.
**Acceptance:** real semantic retrieval beats BM25 on a paraphrase query (test); FAISS save/load
round-trips; manifest mismatch triggers rebuild; reranker reorders a known case; bge-small fits CPU budget.
**Commits:** `feat(embeddings): add sentence-transformers and bge-m3 backends` · `feat(retrieval):
inject embedder into semantic retriever with vectorized search` · `feat(retrieval): replace
heuristic reranker with cross-encoder` · `fix(retrieval): make hybrid stored score match fusion
sort key` · `feat(vectorstore): add faiss store with hub persistence`.

### M2 — Ingestion + advanced chunking + contextual retrieval
**Tasks**
- [ ] `rag/loaders/` (txt/md/pdf/html text-only + directory dispatch).
- [ ] `rag/chunking/` (recursive default + token + semantic + parent-child); chunks feed the existing
  `index_documents` as `{chunk_id: text}` with metadata `{doc_id, parent_id, span}` (zero retriever change).
- [ ] `rag/ingestion/pipeline.py` (load → chunk → embed → index → persist).
- [ ] `rag/retrieval/contextual.py` (`ContextualRetriever` index-time wrapper: LLM writes 1-2
  sentence chunk context → index `context + chunk` into BOTH dense and BM25; docstore keeps original
  for display; contexts cached in the persisted index).
- [ ] Bundle `data/corpus/`; build + persist base index.
**Files:** `rag/loaders/*`, `rag/chunking/*`, `rag/ingestion/pipeline.py`, `rag/retrieval/contextual.py`,
`data/corpus/*`, `data/index/*`, tests.
**Acceptance:** multi-paragraph doc → N chunks with correct overlap; persisted index reloads and
answers; contextual flag measurably changes retrieved chunks; pdf/html extract text only.
**Commits:** `feat(loaders): add text/markdown/pdf/html loaders` · `feat(chunking): add recursive,
token, semantic, and parent-child chunkers` · `feat(ingestion): add end-to-end indexing pipeline` ·
`feat(retrieval): add contextual retrieval chunk augmentation` · `chore(data): add sample corpus
and prebuilt index`.

### M3 — GraphRAG (first-class)
**Tasks**
- [ ] `rag/graph/extractor.py` (LLM entity + relation + dual-level keyword extraction; prompts in registry).
- [ ] `rag/graph/builder.py` (networkx graph; entity dedup/merge; embed nodes + edges).
- [ ] `rag/graph/community.py` (community detection + LLM community summaries for global queries).
- [ ] `rag/graph/store.py` (save/load graph + embeddings; persist to Hub alongside the index).
- [ ] `rag/graph/retriever.py` (`GraphRetriever(BaseRetriever)`: local entity-centric + global
  community-centric dual-level retrieval; Personalized PageRank traversal; returns `RetrievalResult`s).
- [ ] Wire graph into ingestion (build once, persist) and into fusion: **triple-hybrid**
  (semantic + BM25 + graph) via RRF in `HybridRetriever`/pipeline.
- [ ] Cost guards: build-once + cache; upload size cap + UI warning.
**Files:** `rag/graph/*`, `rag/retrieval/hybrid.py`, `rag/ingestion/pipeline.py`, `rag/generation/prompts.py`, tests.
**Acceptance:** graph builds from the demo corpus and persists; a multi-hop question retrieves
connected entities a pure-vector search misses (test); graph retrieval returns valid
`RetrievalResult`s and fuses with vector+BM25; runtime traversal runs on CPU.
**Commits:** `feat(graph): add llm entity-relation extractor` · `feat(graph): build knowledge graph
with embeddings and communities` · `feat(graph): add dual-level pagerank graph retriever` ·
`feat(retrieval): fuse graph retrieval into hybrid pipeline` · `feat(graph): persist graph to hub`.

### M4 — LLM generation with grounded citations
**Tasks**
- [ ] `rag/llm/` (`LLMClient` ABC; `LocalTransformersLLM` CPU/ZeroGPU; `HFInferenceLLM`; `EchoLLM`).
- [ ] **PolarQuant KV-cache integration** in `LocalTransformersLLM` (Llama-3.2-3B fp16 + polar-
  transform KV quant; `RAG_KV_QUANT={polarquant,none}`; graceful fallback if the kernel/repo is
  unavailable). Verify VRAM drop + near-lossless output vs fp16 cache on a fixed prompt.
- [ ] `rag/generation/prompts.py` (qa/citation/refusal templates) + `generator.py`
  (context packing, numbered `[n]` citations, "insufficient context" refusal path).
- [ ] `RAGPipeline.answer(query, …) -> GenerationResult` (retrieve → rerank → generate + citations);
  record `total_generation_time` (**finish bug 3**); keep `query()` unchanged.
**Files:** `rag/llm/*`, `rag/generation/*`, `rag/core.py`, `examples/generation_demo.py`, tests (EchoLLM/tiny model).
**Acceptance:** end-to-end `answer()` returns a grounded answer with source IDs/citations; runs on
CPU with Qwen2.5-1.5B and on ZeroGPU when enabled; refusal path fires with no contexts.
**Commits:** `feat(llm): add transformers and hf-inference llm clients` · `feat(generation): add
grounded answer prompts and citation assembly` · `feat(core): add answer() generation path with
timing`.

### M5 — Query transforms + Agentic/Corrective RAG + Persistence memory
**Tasks**
- [ ] `rag/query/transforms/` (HyDE, multi-query, RAG-Fusion w/ proper RRF — replaces core.py
  pairwise-average merge, decomposition).
- [ ] **Finish bug 2**: align analyzer/optimizer on the shared `Strategy` enum; explicit fallback.
- [ ] `rag/grading/crag.py` (grade contexts via NLI/LLM → correct/ambiguous/incorrect → use/refine/discard;
  bounded re-retrieval loop; refusal on incorrect-with-no-fallback).
- [ ] `rag/grading/self_rag.py` (reflection: relevant? supported? useful? → gate generation).
- [ ] `rag/routing/router.py` (`HeuristicRouter` default reusing the fixed analyzer; `LLMRouter` opt-in
  to pick retriever/transforms/decomposition).
- [ ] Wire transforms/grader/router as injected collaborators into `RAGPipeline.answer()`.
- [ ] **Persistence memory** `rag/memory/`: `MemoryStore` ABC; `ConversationMemory` (SQLite, per-
  session history, survives restarts); `SemanticMemory` (vector store of past turns/facts,
  retrievable as extra context); `FactExtractor` (LLM distills durable facts). Wire into
  `RAGPipeline.answer()`: read relevant memory → add to context → write the turn back → persist
  (local disk; HF Dataset / Qdrant on the demo). Config: `RAG_MEMORY={off,conversation,semantic,full}`.
**Files:** `rag/query/transforms/*`, `rag/grading/*`, `rag/routing/*`, `rag/memory/*`, `rag/core.py`,
`rag/query/{analyzer,optimizer}.py`, `examples/advanced_rag_demo.py`, tests.
**Acceptance:** multi-query/fusion raises recall on a test; CRAG triggers fallback on an
out-of-corpus question instead of hallucinating; router selects bm25 vs hybrid vs graph by query
type; **memory persists across a simulated restart and a follow-up question uses prior context**;
each feature individually toggleable via config.
**Commits:** `feat(query): add hyde, multi-query, rag-fusion, and decomposition transforms` ·
`refactor(core): use rrf fusion for multi-query retrieval` · `fix(query): align analyzer
strategies with optimizer` · `feat(grading): add corrective rag grading and fallback` ·
`feat(grading): add self-rag reflection` · `feat(routing): add heuristic and llm query routers` ·
`feat(memory): add persistent conversation and semantic memory`.

### M6 — Comprehensive evaluation (gold set + BEIR + generation + system)
**Tasks**
- [ ] Confirm MAP fix; add graded NDCG, Hit@k, R-Precision to `metrics.py`.
- [ ] `evaluation/generation_metrics.py` (faithfulness via NLI claim-entailment [primary] +
  RAGAS-with-HF [secondary]; answer relevancy via embedder; context precision/recall; answer
  correctness via similarity+NLI).
- [ ] `evaluation/judges.py` (`NLIJudge` default; `LLMJudge` wrapping RAGAS with an open HF LLM —
  **no paid keys ever**).
- [ ] `evaluation/system_metrics.py` (per-stage latency, throughput, tokens, peak RAM, GPU seconds;
  emit explicit "cost = $0 (open models)").
- [ ] `evaluation/datasets.py` (gold-set loader; **BEIR adapter** for SciFact/NFCorpus with `--subset` cap).
- [ ] `data/gold/qa.jsonl` (~30-60 Q hand-curated over the demo corpus; schema with optional graded
  relevance / reference answer / reference contexts).
- [ ] `evaluation/pipeline_evaluator.py`, `report.py` (JSON + Markdown + `--compare` delta tables),
  `run.py` CLI; `evaluation/configs/*` (named pipeline configs to benchmark).
- [ ] Eval smoke test.
- [ ] **AWS EC2 cloud-GPU benchmarking (optional tier)** `infra/aws/`: `setup-vpc.sh`,
  `launch-gpu.sh` (g5/g6 = 24GB, g6e = 48GB; Deep Learning AMI), `bootstrap.sh`, `teardown.sh`,
  mirroring the `D:\golgi_vcc` AWS-CLI pattern (region us-east-1, key pair, SG/subnet, `aws ec2
  run-instances` + `wait`). Run the eval harness on EC2 with the `cloud` profile (8B fp16) and add
  its numbers to the comparison table. **Always `teardown.sh` after** to avoid cost.
**Files:** `evaluation/*`, `data/gold/qa.jsonl`, `tests/eval/test_eval_smoke.py`, `infra/aws/*`.
**Acceptance:** `python -m evaluation.run` emits `report.json` + `report.md` with retrieval +
generation + system tables; `--compare` shows deltas (hybrid>semantic>bm25; rerank lifts P@k;
graph/contextual/CRAG effects visible); BEIR subset produces recognized numbers; all judges open; smoke green.
**Commits:** `feat(evaluation): add graded ndcg, hit@k, r-precision` · `feat(evaluation): add gold
dataset and beir adapter` · `feat(evaluation): add open-model generation metrics and judges` ·
`feat(evaluation): add system latency/token metrics` · `feat(evaluation): add cli runner and
json/markdown reports` · `test(evaluation): add end-to-end smoke test`.

### M7 — Gradio app + HF Space deployment (the demo; only after local works)
**Tasks**
- [ ] **HF write round-trip test** (deploy gate): with `HF_TOKEN`, create a temp private repo
  (`anshul2048/_rag_write_test`), `upload_file` a tiny file, confirm, then `delete_repo` — prove
  end-to-end write before pushing the real Space/Dataset.
- [ ] `app.py` + `app/` (ask tab: answer + citations + retrieved chunks + graph context; ingest tab:
  upload → session index; eval tab: capped live metrics).
- [ ] `app/state.py` lazy singletons + boot restore (pull index/graph from Hub → verify manifest →
  rebuild if mismatch).
- [ ] ZeroGPU wiring (`@spaces.GPU` on encode/rerank/generate/extract; CPU no-op shim).
- [ ] `README_HF.md` with Space YAML frontmatter (`sdk: gradio`, `app_file: app.py`, …);
  set `HF_TOKEN` Space secret; `requirements.txt` final.
- [ ] Create Space `anshul2048/production-rag` + Dataset `anshul2048/production-rag-index`; push.
**Files:** `app.py`, `app/*`, `README_HF.md`, `requirements.txt`.
**Acceptance:** Space boots, answers end-to-end with sources + graph context, shows metrics; cold
start restores index from Hub; GPU funcs decorated; no secrets in repo; CPU profile usable 24/7.
**Commits:** `feat(app): add gradio query interface with citations` · `feat(app): add ingest and
evaluation tabs` · `feat(app): add lazy loading and hub index restore` · `chore(space): add hugging
face space configuration`.

### M8 — Docs, CI, release
**Tasks**
- [ ] Upgrade `README.md` (real stack, architecture diagram, model table, install extras, quickstart
  ingest→query→answer, eval how-to + sample results, Space link, `HF_TOKEN` config).
- [ ] `docs/ARCHITECTURE.md`, `docs/EVALUATION.md` (corrected MAP, judge methodology, schema),
  `docs/MODEL_CARD.md` (choices + measured results + limits + repro).
- [ ] `.github/workflows/ci.yml` (ruff/black/mypy/pytest+cov; py3.10+3.11; `--cov-fail-under=80`;
  HF-model `slow` job optional).
- [ ] **No-AI-attribution audit** (grep commits/docs/code for assistant refs + Co-Authored-By).
- [ ] Push to GitHub `origin`; tag `v2.0.0`.
**Files:** `README.md`, `docs/*`, `.github/workflows/ci.yml`.
**Acceptance:** CI green on a PR; docs match the implemented system + measured metrics; attribution
audit clean; tag pushed.
**Commits:** `ci: add lint, type-check, and test workflow` · `docs: document architecture and
evaluation` · `docs: publish model card and results` · `chore: release v2.0.0`.

---

## Consolidated TO-DO checklist

**Phase 0 — out of plan mode, immediate (local)**
- [ ] **Store this plan in the repo:** copy it to `d:\Production-Rag-Project\docs\PROJECT_PLAN.md`
  and commit (`docs: add project plan and roadmap`). This is the very first action.
- [x] Local GPU env (DONE): conda env `production-rag` (default), Python 3.11.15, torch 2.10+cu128, RTX 5070 Ti 11.94GB.
- [ ] Load-test bge-m3 + bge-reranker-v2-m3 + Llama-3.2-3B-Instruct (fp16 + PolarQuant) on the GPU (budget check).
- [ ] Rename `.env` key → `HF_TOKEN` (+ shim); add `.env.example`. (HF write scope already verified.)

**Foundations (M0)**
- [ ] `rag/config/` settings + model registry (local/cloud/zerogpu/cpu/fake profiles).
- [ ] `rag/observability/` logging + Prometheus metrics + Loki logs + watch (built first; all use it).
- [ ] `infra/observability/` Grafana stack (docker-compose: Prometheus+Loki+Grafana) + dashboards.
- [ ] `requirements.txt` + `pyproject` extras + `requires-python>=3.10` + register subpackages.
- [ ] `rag/interfaces/protocols.py`.
- [ ] Move `SimpleEmbedder` → `HashEmbedder` (alias kept).
- [ ] Fix MAP bug + AP helper + regression test; `Strategy` enum stub; timing scaffold.
- [ ] `tests/` scaffold + behavior-lock tests.

**Retrieval core (M1)**
- [ ] Real embedders (bge-small / bge-m3 / hf-inference).
- [ ] DI refactor: SemanticRetriever, CrossEncoderReranker, HybridRetriever (+ fix score/sort).
- [ ] FAISS + Numpy vector stores + HF-Dataset persistence + manifest.
- [ ] Qdrant hosted vector store (free Qdrant Cloud) behind the same ABC (`RAG_VECTORSTORE`).
- [ ] BM25 serialize/load.

**Ingestion (M2)**
- [ ] Loaders (txt/md/pdf/html text-only) + DirectoryLoader.
- [ ] Chunkers (recursive default + token + semantic + parent-child).
- [ ] Ingestion pipeline; bundle corpus; build + persist base index.
- [ ] Contextual retrieval wrapper (dense + BM25; cache contexts).

**GraphRAG — first-class (M3)**
- [ ] Entity/relation extractor (LLM) + dual-level keywords.
- [ ] Graph builder (networkx, dedup, node/edge embeddings) + communities + summaries.
- [ ] Graph store + Hub persistence.
- [ ] Dual-level + PageRank GraphRetriever.
- [ ] Triple-hybrid fusion (semantic + BM25 + graph) via RRF; cost guards.

**Generation (M4)**
- [ ] LLM clients (local transformers / hf-inference / echo).
- [ ] Prompt registry + Generator with `[n]` citations + refusal.
- [ ] PolarQuant KV-cache quant on the local 3B LLM (`RAG_KV_QUANT`; graceful fallback).
- [ ] `RAGPipeline.answer()` + finish timing fix.

**Agentic / Corrective + Memory (M5)**
- [ ] Query transforms (HyDE / multi-query / RAG-Fusion RRF / decomposition).
- [ ] Fix dead strategy path (shared enum + explicit fallback).
- [ ] CRAG grader + bounded re-retrieval + refusal.
- [ ] Self-RAG reflection; HeuristicRouter + LLMRouter.
- [ ] Persistence memory: ConversationMemory (SQLite) + SemanticMemory + FactExtractor; wire into
  `answer()`; survives restarts; `RAG_MEMORY` config.

**Evaluation (M6)**
- [ ] Confirm MAP; add graded NDCG / Hit@k / R-Precision.
- [ ] Generation metrics (faithfulness/answer-relevancy/context P-R/answer-correctness) with open judges.
- [ ] Judges (NLI default + RAGAS-with-HF).
- [ ] System metrics (latency/tokens/RAM/GPU-seconds; cost=$0).
- [ ] Gold set (`data/gold/qa.jsonl`) + BEIR adapter (SciFact/NFCorpus).
- [ ] Pipeline evaluator + reports (JSON/MD) + `--compare`; CLI `python -m evaluation.run`; smoke test.
- [ ] AWS EC2 cloud-GPU benchmarking (`infra/aws/*`, golgi pattern); run `cloud` profile; teardown after.

**Deploy (M7) — HF demo, only after local works**
- [ ] HF write round-trip test (create temp private repo → upload → delete) with `HF_TOKEN`.
- [ ] Gradio `app.py` + tabs (ask / ingest / eval).
- [ ] Lazy loading + boot index/graph restore from Hub.
- [ ] ZeroGPU decorators + CPU shim.
- [ ] Space README frontmatter; set `HF_TOKEN` Space secret.
- [ ] Create + push Space `anshul2048/production-rag` and Dataset `anshul2048/production-rag-index`.

**Docs / CI / release (M8)**
- [ ] README upgrade; `docs/ARCHITECTURE|EVALUATION|MODEL_CARD`.
- [ ] CI workflow (ruff/black/mypy/pytest+cov; py3.10/3.11).
- [ ] No-AI-attribution audit.
- [ ] Push to GitHub `origin`; tag `v2.0.0`.

---

## Verification (how we'll know it works, end-to-end)

- **Local GPU (primary):** under `RAG_PROFILE=local`, the full stack (bge-m3 + bge-reranker-v2-m3 +
  Llama-3.2-3B fp16 + PolarQuant KV cache) loads within 12GB VRAM and runs end-to-end ingest →
  graph build → retrieve → rerank → generate **at full quality**; this is the gate before any HF push.
- **Memory:** a follow-up question reuses prior-turn context; memory survives a process restart
  (SQLite + vector memory reload).
- **PolarQuant:** measured KV-cache VRAM drop + near-lossless output vs an fp16-cache baseline.
- **Cloud (optional):** the `cloud` EC2 profile (8B fp16) runs the eval harness and its numbers
  appear in the comparison table; instance is torn down after.
- **Unit/integration:** `pytest -m "not slow"` green; MAP regression proves real AP; FAISS/graph
  round-trips; CRAG fallback fires; graph retrieval beats vector on a multi-hop case.
- **Retrieval quality:** `python -m evaluation.run --config configs/hybrid_rerank.yaml` on the gold
  set and a BEIR subset shows hybrid > semantic > BM25 and rerank/contextual/graph lifts; numbers
  written to `report.md`.
- **Generation quality:** faithfulness/answer-relevancy/context-precision/recall reported with open
  judges (zero paid calls); refusal path verified on out-of-corpus questions.
- **System:** latency/token/RAM tables; CPU default stack stays within ~9GB; ZeroGPU path runs the
  big models.
- **Observability (Grafana):** `make observability-up` brings up Prometheus + Loki + Grafana; the
  app's `/metrics` is scraped and per-stage latency/throughput/cache/VRAM dashboards populate; logs
  are searchable in Grafana (Loki) filtered by component/level; eval-quality metrics show on the
  quality dashboard. This is the single pane of glass for all metrics, logs, and monitors.
- **Live demo:** the HF Space `anshul2048/production-rag` boots (restoring the index/graph from the
  Hub Dataset), answers a query with citations + graph context, and renders the eval tab.
- **Hygiene:** CI green; attribution audit finds nothing; git author remains Anshul Kumar; no
  secrets committed.

---

## Appendix A — Interface sketches (the contracts everything depends on)

`rag/interfaces/protocols.py` (all `@runtime_checkable`, so existing fakes satisfy them with no
inheritance change). New retriever params are keyword-only with fake defaults → 100% back-compat.

```python
@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> np.ndarray: ...                    # (dim,)
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...  # (N, dim)

@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, document: str) -> float: ...
    def score_batch(self, query: str, documents: list[str]) -> list[float]: ...

@runtime_checkable
class VectorStore(Protocol):
    dim: int
    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None) -> None: ...
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "VectorStore": ...
    def __len__(self) -> int: ...

@runtime_checkable
class Chunker(Protocol):
    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> list["Chunk"]: ...

@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, source: str) -> list["Document"]: ...

@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512,
                 temperature: float = 0.0, stop: list[str] | None = None) -> str: ...
    def stream(self, prompt: str, **kw) -> Iterable[str]: ...

@runtime_checkable
class QueryTransform(Protocol):
    def transform(self, query: str) -> list[str]: ...               # >= 1 variant

@runtime_checkable
class RetrievalGrader(Protocol):
    def grade(self, query: str, contexts: list[RetrievalResult]) -> "GradeResult": ...

@runtime_checkable
class Router(Protocol):
    def route(self, query: str) -> "RouteDecision": ...
```

Refactored retriever / pipeline signatures (defaults reproduce today's behavior exactly):
```python
SemanticRetriever(embedder: Embedder = None, embedding_dim: int = 384,
                  vector_store: VectorStore = None, name="SemanticRetriever")
CrossEncoderReranker(model: Reranker = None)
HybridRetriever(semantic: BaseRetriever = None, bm25: BaseRetriever = None,
                graph: BaseRetriever = None, weights=(0.4,0.3,0.3), k_rrf=60, fusion="rrf")
RAGPipeline(retriever, reranker=None, query_optimizer=None, query_analyzer=None, *,
            generator: Generator = None, query_transforms: list[QueryTransform] = None,
            grader: RetrievalGrader = None, router: Router = None, config: RAGConfig = None)
RAGPipeline.answer(query, k=10, rerank_k=5, **kw) -> GenerationResult
```

New dataclasses: `Chunk(chunk_id, doc_id, text, start, end, metadata)`,
`Document(doc_id, text, source, metadata)`,
`GenerationResult(answer, citations, contexts, grade, transforms_used, usage, timings)`,
`GradeResult(label, score, action, per_context_scores)`,
`RouteDecision(strategy, retriever_name, use_transforms, reason)`.

## Appendix B — Eval schema & config

Gold set `data/gold/qa.jsonl` (one record per line):
```json
{"id": "q001",
 "question": "How does BM25 length-normalize term frequency?",
 "relevant_doc_ids": ["chunk_3", "chunk_7"],
 "relevance_grades": {"chunk_3": 2, "chunk_7": 1},
 "reference_answer": "BM25 divides ... by (1 - b + b * dl/avgdl).",
 "reference_contexts": ["..."]}
```
- `relevant_doc_ids` (required) drive P@k/R@k/MRR/NDCG/MAP.
- `relevance_grades` (optional) enable graded NDCG.
- `reference_answer` (optional) enables answer correctness; `reference_contexts` (optional) enable
  reference-based context recall.

Pipeline config `evaluation/configs/hybrid_rerank.yaml` (named, comparable):
```yaml
name: hybrid_rerank
profile: local
retriever: hybrid          # semantic | bm25 | hybrid | graph
fusion: rrf
contextual: true
rerank: true
rerank_k: 5
query_transform: fusion    # none | hyde | multi | fusion
graph: true
crag: true
top_k: 10
```
`python -m evaluation.run --config evaluation/configs/hybrid_rerank.yaml --gold data/gold/qa.jsonl
--out reports/` → `report.json` + `report.md`. `--compare a.yaml b.yaml …` → delta table (bold winner
per metric). A capped subset runs live in the Gradio eval tab.

Report tables: **Retrieval** (P@k, R@k, MRR, NDCG@k, MAP, Hit@k, R-Precision), **Generation**
(faithfulness, answer relevancy, context precision, context recall, answer correctness), **System**
(per-stage latency, throughput, prompt/completion tokens, peak RAM/VRAM, GPU seconds, cost=$0).

## Appendix C — Techniques glossary (with citations)

- **BM25** — sparse lexical ranking with TF saturation (`k1`) + length norm (`b`); kept from-scratch.
- **Dense / semantic retrieval** — embed query+docs, rank by cosine; now real via bge-m3.
- **Hybrid + RRF** — combine rankings by `Σ weight · 1/(k_rrf + rank)`; robust, scale-free fusion.
- **Contextual Retrieval** — LLM-written chunk context prepended before embedding + BM25 ([Anthropic](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)).
- **GraphRAG** — entity-relation graph + community summaries ([MS GraphRAG](https://arxiv.org/html/2507.03226v3)); **LightRAG** dual-level; **HippoRAG** PageRank multi-hop.
- **HyDE** — embed a hypothetical answer to bridge the query↔doc vocabulary gap.
- **Multi-query / RAG-Fusion** — retrieve over paraphrases, fuse via RRF.
- **CRAG** — grade retrieved docs, then use / refine / refuse before generating ([CRAG](https://www.emergentmind.com/topics/corrective-retrieval-augmented-generation-crag)).
- **Self-RAG** — reflection tokens (relevant? supported? useful?) gate retrieval + generation.
- **Agentic RAG / routing** — route or plan multi-step retrieval by query type/confidence.
- **RAGAS metrics** — faithfulness, answer relevancy, context precision/recall ([RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)).
- **MAP / AP** — mean of per-query average precision (the metric currently mis-computed as NDCG).
- **PolarQuant** — KV-cache quantization via a polar transform; ~4.2× smaller cache, ~14% faster decode, near-lossless ([arXiv 2502.00527](https://arxiv.org/pdf/2502.00527)).
- **Persistence memory** — durable conversation history + LLM-extracted facts + retrievable past interactions, surviving restarts (local SQLite/vector; hosted Qdrant/HF Dataset on the demo).
- **Hosted vector DB** — managed vector store (Qdrant Cloud free tier) used as a `VectorStore` backend; durable persistence without local files.

## Risks & mitigations
- **12GB VRAM ceiling (local):** 7B fp16 won't fit alongside embedder+reranker → default to **4-bit
  NF4** for the LLM; lazy-load or CPU-place the reranker; `RAG_QUANTIZATION`/`RAG_RERANKER_DEVICE`
  flags to tune. Fallback to Qwen2.5-3B fp16 if 4-bit quality is insufficient.
- **CPU LLM latency** (HF `cpu` fallback only, 1.5B fp32 is slow): stream tokens + progress UI; the
  demo defaults to ZeroGPU for generation. (Not a concern locally — full-speed GPU.)
- **ZeroGPU daily quota:** default to CPU; GPU only on leaf functions; batch; warm-cache models.
- **Ephemeral disk:** persist index+graph to a private HF Dataset; pull on boot; manifest-gated rebuild.
- **GraphRAG construction cost:** build once for the bundled corpus, persist; cap upload-time extension.
- **Contextual-retrieval ingest cost:** one LLM call/chunk → compute once, cache in the persisted index.
- **Gated models (Llama):** accept license on `anshul2048`; otherwise default to Qwen (ungated).
- **Dependency weight on the Space:** keep core numpy-only; install only `serve`+needed extras on the Space.

## Out of scope (explicit)
- Multimodal/image retrieval (ColPali/ColQwen) — excluded per decision.
- **Paid** APIs / **paid** vector DB tiers / **paid** managed LLM endpoints. (Free hosted tiers —
  e.g., Qdrant Cloud free — ARE in scope; AWS EC2 is opt-in and user-funded for testing only.)
- Auth, multi-tenant, or horizontal scaling beyond a single free Space.
