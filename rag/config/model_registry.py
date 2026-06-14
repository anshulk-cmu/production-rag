"""Model choices per serving profile. Switch profiles, not code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSet:
    """The four model roles for one profile. None means use the built-in fake."""

    embedder: str | None  # None -> HashEmbedder
    reranker: str | None  # None -> SimpleRerankerModel
    llm: str | None  # None -> EchoLLM
    nli_judge: str | None  # None -> skip NLI faithfulness


MODEL_REGISTRY: dict[str, ModelSet] = {
    "local": ModelSet(
        embedder="BAAI/bge-m3",
        reranker="BAAI/bge-reranker-v2-m3",
        llm="meta-llama/Llama-3.2-3B-Instruct",
        nli_judge="cross-encoder/nli-deberta-v3-base",
    ),
    "cloud": ModelSet(
        embedder="BAAI/bge-m3",
        reranker="BAAI/bge-reranker-v2-m3",
        llm="meta-llama/Llama-3.1-8B-Instruct",
        nli_judge="cross-encoder/nli-deberta-v3-base",
    ),
    "zerogpu": ModelSet(
        embedder="BAAI/bge-m3",
        reranker="BAAI/bge-reranker-v2-m3",
        llm="meta-llama/Llama-3.2-3B-Instruct",
        nli_judge="cross-encoder/nli-deberta-v3-base",
    ),
    "cpu": ModelSet(
        embedder="BAAI/bge-small-en-v1.5",
        reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm="Qwen/Qwen2.5-1.5B-Instruct",
        nli_judge="cross-encoder/nli-deberta-v3-base",
    ),
    "fake": ModelSet(embedder=None, reranker=None, llm=None, nli_judge=None),
}
