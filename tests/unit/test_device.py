from rag.utils.device import cuda_available, free_gpu, gpu_memory_gb


def test_cuda_available_is_bool():
    assert isinstance(cuda_available(), bool)


def test_gpu_memory_gb_returns_tuple():
    alloc, reserved = gpu_memory_gb()
    assert alloc >= 0.0 and reserved >= 0.0


def test_free_gpu_returns_nonneg_float():
    freed = free_gpu()
    assert isinstance(freed, float) and freed >= 0.0
