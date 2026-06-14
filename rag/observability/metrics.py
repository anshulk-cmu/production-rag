"""Comprehensive metrics for the RAG system.

Prometheus instruments for latency, throughput, errors, tokens, cache, model loads,
index size, evaluation quality, and GPU/RAM/CPU resource usage. Histograms give
p50/p95/p99 in Grafana; snapshot() powers the in-app --watch view and get_stats().
"""

import functools
from contextlib import contextmanager

import psutil
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, start_http_server

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None

# Buckets span sub-millisecond to minutes so embed (ms) and generation (s) both resolve well.
_LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds", "Latency per pipeline stage", ["stage"], buckets=_LAT_BUCKETS
)
MODEL_LOAD = Histogram("rag_model_load_seconds", "Model load time", ["role"], buckets=_LAT_BUCKETS)
REQUESTS = Counter("rag_requests_total", "Requests per stage and status", ["stage", "status"])
TOKENS = Counter("rag_tokens_total", "LLM tokens processed", ["kind"])
CACHE = Counter("rag_cache_events_total", "Cache hits and misses", ["name", "result"])
DOCS_INDEXED = Gauge("rag_documents_indexed", "Documents or chunks indexed", ["kind"])
QUALITY = Gauge("rag_quality_score", "Evaluation quality scores", ["metric"])

GPU_MEM_ALLOCATED = Gauge("rag_gpu_memory_allocated_bytes", "Torch CUDA allocated", ["device"])
GPU_MEM_RESERVED = Gauge("rag_gpu_memory_reserved_bytes", "Torch CUDA reserved", ["device"])
GPU_UTIL = Gauge("rag_gpu_utilization_percent", "GPU utilization", ["device"])
RAM_USED = Gauge("rag_ram_used_bytes", "Process resident memory")
CPU_PERCENT = Gauge("rag_cpu_percent", "Process CPU percent")

# Cached process handle, primed so the first cpu_percent() is a real delta, not 0.0.
_PROC = psutil.Process()
_PROC.cpu_percent()


@contextmanager
def timer(stage: str):
    """Time a pipeline stage into the latency histogram."""
    with STAGE_LATENCY.labels(stage=stage).time():
        yield


def timed(stage: str):
    """Decorator form of timer()."""

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            with timer(stage):
                return fn(*args, **kwargs)

        return wrap

    return deco


def record_request(stage: str, ok: bool = True) -> None:
    REQUESTS.labels(stage=stage, status="ok" if ok else "error").inc()


def record_tokens(prompt: int = 0, completion: int = 0) -> None:
    if prompt:
        TOKENS.labels(kind="prompt").inc(prompt)
    if completion:
        TOKENS.labels(kind="completion").inc(completion)


def record_cache(name: str, hit: bool) -> None:
    CACHE.labels(name=name, result="hit" if hit else "miss").inc()


def record_model_load(role: str, seconds: float) -> None:
    MODEL_LOAD.labels(role=role).observe(seconds)


def set_documents(count: int, kind: str = "chunks") -> None:
    DOCS_INDEXED.labels(kind=kind).set(count)


def set_quality(metric: str, value: float) -> None:
    QUALITY.labels(metric=metric).set(value)


def sample_system() -> None:
    """Sample process RAM/CPU and, when available, GPU memory and utilization."""
    RAM_USED.set(_PROC.memory_info().rss)
    CPU_PERCENT.set(_PROC.cpu_percent())
    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            GPU_MEM_ALLOCATED.labels(device=str(i)).set(torch.cuda.memory_allocated(i))
            GPU_MEM_RESERVED.labels(device=str(i)).set(torch.cuda.memory_reserved(i))
        _sample_gpu_util()


def _sample_gpu_util() -> None:
    if pynvml is None:
        return
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            GPU_UTIL.labels(device=str(i)).set(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        pass


def snapshot() -> dict[str, float]:
    """Current value of every rag_* metric, flattened for the watch view / get_stats()."""
    out: dict[str, float] = {}
    for family in REGISTRY.collect():
        if not family.name.startswith("rag"):
            continue
        for sample in family.samples:
            labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
            key = f"{sample.name}{{{labels}}}" if labels else sample.name
            out[key] = sample.value
    return out


def start_metrics_server(port: int = 9100) -> None:
    """Expose Prometheus /metrics on `port` for scraping."""
    start_http_server(port)
