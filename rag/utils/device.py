"""Device helpers: CUDA detection, memory readout, and GPU garbage clearing."""

from __future__ import annotations

import gc

from rag.observability import get_logger

try:
    import torch
except ImportError:
    torch = None

_log = get_logger(__name__)


def cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def gpu_memory_gb() -> tuple[float, float]:
    """(allocated, reserved) GB on the current CUDA device; (0, 0) on CPU."""
    if not cuda_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3


def free_gpu() -> float:
    """Release cached GPU memory (Python gc + CUDA caching allocator). Returns GB freed."""
    before = gpu_memory_gb()[1]
    gc.collect()
    if cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    after = gpu_memory_gb()[1]
    freed = max(0.0, before - after)
    _log.debug("free_gpu: reserved %.2f -> %.2f GB (freed %.2f)", before, after, freed)
    return freed
