import logging

from rag.observability import (
    configure_logging,
    get_logger,
    record_cache,
    record_tokens,
    render,
    sample_system,
    snapshot,
    timed,
    timer,
)


def test_timer_records_latency():
    with timer("embed"):
        pass
    assert any("rag_stage_latency_seconds_count" in k and "embed" in k for k in snapshot())


def test_timed_decorator():
    @timed("retrieve")
    def f():
        return 1

    assert f() == 1
    assert any("rag_stage_latency_seconds_count" in k and "retrieve" in k for k in snapshot())


def test_record_tokens():
    record_tokens(prompt=12, completion=8)
    snap = snapshot()
    assert any("rag_tokens_total" in k and "prompt" in k for k in snap)
    assert any("rag_tokens_total" in k and "completion" in k for k in snap)


def test_record_cache():
    record_cache("embedder", hit=True)
    assert any("rag_cache_events_total" in k and "hit" in k for k in snapshot())


def test_sample_system_sets_ram():
    sample_system()
    assert any("rag_ram_used_bytes" in k for k in snapshot())


def test_render_returns_table():
    record_tokens(prompt=1)
    assert "rag_tokens_total" in render(refresh_system=False)


def test_get_logger_is_logger():
    assert isinstance(get_logger("rag.test"), logging.Logger)


def test_configure_logging_idempotent():
    import rag.observability.logging as log_mod

    log_mod._CONFIGURED = False
    logging.getLogger("rag").handlers.clear()
    configure_logging(level="INFO")
    n1 = len(logging.getLogger("rag").handlers)
    configure_logging(level="DEBUG")
    n2 = len(logging.getLogger("rag").handlers)
    assert n1 == 1 and n2 == 1
