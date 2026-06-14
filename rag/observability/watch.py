"""Quick terminal view of current metrics (no Grafana needed)."""

from .metrics import sample_system, snapshot


def render(refresh_system: bool = True) -> str:
    if refresh_system:
        sample_system()
    snap = snapshot()
    if not snap:
        return "(no metrics yet)"
    width = max(len(k) for k in snap)
    return "\n".join(f"{k:<{width}}  {v:.4f}" for k, v in sorted(snap.items()))
