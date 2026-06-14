import json
import logging
import types

from rag.observability import (
    configure_logging,
    get_logger,
    record_cache,
    record_model_load,
    record_request,
    record_tokens,
    render,
    sample_system,
    set_documents,
    set_quality,
    snapshot,
    timed,
    timer,
)
from rag.observability.logging import JsonFormatter, _add_loki


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


def test_configure_logging_idempotent(monkeypatch):
    import rag.observability.logging as log_mod
    from rag.config.settings import get_settings

    # Disable Loki so the test adds no network handler and stays at one handler.
    monkeypatch.setenv("RAG_LOKI_URL", "")
    get_settings.cache_clear()
    log_mod._CONFIGURED = False
    logging.getLogger("rag").handlers.clear()
    configure_logging(level="INFO")
    n1 = len(logging.getLogger("rag").handlers)
    configure_logging(level="DEBUG")
    n2 = len(logging.getLogger("rag").handlers)
    get_settings.cache_clear()
    assert n1 == 1 and n2 == 1


def test_json_formatter_emits_valid_json():
    record = logging.LogRecord("rag.x", logging.INFO, __file__, 1, "hello", None, None)
    out = json.loads(JsonFormatter().format(record))
    assert out["level"] == "INFO" and out["logger"] == "rag.x" and out["message"] == "hello"


def test_metric_helpers_appear_in_snapshot():
    record_request("retrieve", ok=True)
    set_documents(5, kind="chunks")
    set_quality("ndcg_at_10", 0.8)
    record_model_load("embedder", 1.2)
    snap = snapshot()
    assert any("rag_requests_total" in k and "retrieve" in k for k in snap)
    assert any("rag_documents_indexed" in k for k in snap)
    assert any("rag_quality_score" in k and "ndcg_at_10" in k for k in snap)
    assert any("rag_model_load_seconds_count" in k for k in snap)


def test_loki_url_normalization():
    log = logging.getLogger("rag._loki_test")
    log.handlers.clear()
    s = types.SimpleNamespace(loki_url="https://loki.example", loki_user="u", grafana_token="t")
    _add_loki(log, s, logging.Formatter(), "INFO")
    handler = log._loki_listener.handlers[0]
    assert handler.push_url == "https://loki.example/loki/api/v1/push"
    log._loki_listener.stop()
    log.handlers.clear()
