"""Structured logging: one setup, optional JSON, optional Loki shipping to Grafana."""

import json as _json
import logging
import sys

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return _json.dumps(data)


def configure_logging(
    level: str | None = None, json_logs: bool | None = None, loki_url: str | None = None
) -> None:
    """Configure the 'rag' logger once. Unset args fall back to settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    from rag.config import get_settings

    s = get_settings()
    level = (level or s.log_level).upper()
    json_logs = s.log_json if json_logs is None else json_logs
    loki_url = s.loki_url if loki_url is None else loki_url

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger("rag")
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    if loki_url:
        _add_loki(root, loki_url, level)
    _CONFIGURED = True


def _add_loki(root: logging.Logger, url: str, level: str) -> None:
    try:
        import logging_loki
    except ImportError:
        root.warning("loki_url set but python-logging-loki not installed; skipping Loki")
        return
    handler = logging_loki.LokiHandler(url=url, tags={"app": "production-rag"}, version="1")
    handler.setLevel(level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
