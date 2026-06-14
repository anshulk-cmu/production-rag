"""Central settings: one control panel for the whole system, loaded from env/.env."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .model_registry import MODEL_REGISTRY, ModelSet

Profile = Literal["local", "cloud", "zerogpu", "cpu", "fake"]


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="RAG_", extra="ignore"
    )

    profile: Profile = "local"

    # Secrets / hosted services (read without the RAG_ prefix).
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "HF-Token")
    )
    qdrant_url: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_URL"))
    qdrant_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_API_KEY")
    )

    # Runtime knobs (RAG_* env vars).
    # Weights always load in bf16/fp16 (full quality); never weight-quantize.
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    kv_quant: Literal["polarquant", "none"] = "polarquant"
    reranker_device: str = "cuda"
    vectorstore: Literal["faiss", "numpy", "qdrant"] = "faiss"
    memory: Literal["off", "conversation", "semantic", "full"] = "full"

    # Retrieval / generation hyperparameters.
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10
    rerank_k: int = 5
    rrf_k: int = 60
    semantic_weight: float = 0.5
    max_new_tokens: int = 512
    temperature: float = 0.0

    # Logging + Grafana Cloud observability.
    log_level: str = "INFO"
    log_json: bool = False
    loki_url: str | None = None
    loki_user: str | None = None
    prom_url: str | None = None
    prom_user: str | None = None
    grafana_token: str | None = None

    @property
    def models(self) -> ModelSet:
        return MODEL_REGISTRY[self.profile]


@lru_cache
def get_settings() -> RAGSettings:
    return RAGSettings()
