# Observability — Grafana Cloud

Default observability backend. Logs ship straight from the app; metrics ship via Grafana Alloy.
A local Prometheus + Loki + Grafana docker-compose can be added here later as an optional offline
alternative, but Grafana Cloud is the default (no Docker, works for both local and the HF demo).

## Logs (no agent needed)
`rag.observability.configure_logging()` attaches a non-blocking Loki handler whenever `RAG_LOKI_URL`,
`RAG_LOKI_USER`, and `RAG_GRAFANA_TOKEN` are set in `.env`. Logs land in Grafana Cloud →
Explore → Loki, filtered by `{app="production-rag"}`.

## Metrics (Grafana Alloy)
1. Expose the app metrics endpoint: `from rag.observability import start_metrics_server; start_metrics_server(9100)`.
2. Install Alloy (a single binary, no Docker): https://grafana.com/docs/alloy/latest/set-up/install/
3. Export the env vars and run Alloy:
   ```bash
   export $(grep -E '^RAG_PROM_|^RAG_GRAFANA_TOKEN' .env | xargs)
   alloy run infra/observability/alloy/config.alloy
   ```
Metrics land in Grafana Cloud → Explore → Prometheus (e.g. `rag_stage_latency_seconds`,
`rag_gpu_memory_allocated_bytes`).

## Dashboard
Import `grafana/dashboards/rag-overview.json` via Grafana → Dashboards → New → Import (pick your
Prometheus data source when prompted). It's a starter; extend panels in the UI as needed.
