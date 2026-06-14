"""Structured logging: one setup, optional JSON, optional non-blocking Loki shipping to Grafana."""

import base64
import json as _json
import logging
import sys
import time
import urllib.request
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

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


class LokiHandler(logging.Handler):
    """Push each record to a Grafana/Loki endpoint over HTTP basic auth (stdlib only)."""

    def __init__(self, push_url: str, user: str, token: str):
        super().__init__()
        self.push_url = push_url
        self._auth = "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = {
                "app": "production-rag",
                "level": record.levelname.lower(),
                "logger": record.name,
            }
            payload = {
                "streams": [
                    {"stream": stream, "values": [[str(time.time_ns()), self.format(record)]]}
                ]
            }
            req = urllib.request.Request(
                self.push_url, data=_json.dumps(payload).encode(), method="POST"
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", self._auth)
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:
            self.handleError(record)


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configure the 'rag' logger once. Unset args fall back to settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    from rag.config import get_settings

    s = get_settings()
    level = (level or s.log_level).upper()
    json_logs = s.log_json if json_logs is None else json_logs

    fmt = (
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    root = logging.getLogger("rag")
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    if s.loki_url and s.loki_user and s.grafana_token:
        _add_loki(root, s, fmt, level)
    _CONFIGURED = True


def _add_loki(root: logging.Logger, s, fmt: logging.Formatter, level: str) -> None:
    push_url = s.loki_url.rstrip("/")
    if not push_url.endswith("/loki/api/v1/push"):
        push_url += "/loki/api/v1/push"
    loki = LokiHandler(push_url, s.loki_user, s.grafana_token)
    loki.setFormatter(fmt)
    loki.setLevel(level)
    # Non-blocking: records go to a queue; a background thread does the HTTP push.
    queue: Queue = Queue(-1)
    listener = QueueListener(queue, loki, respect_handler_level=True)
    listener.start()
    qh = QueueHandler(queue)
    qh.setLevel(level)
    root.addHandler(qh)
    root._loki_listener = listener  # keep a reference so the worker thread isn't collected


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
