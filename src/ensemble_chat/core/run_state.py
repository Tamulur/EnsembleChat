from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ensemble_chat.core.session_state import SessionState


class RunPhase(str, Enum):
    IDLE = "idle"
    RUNNING_SINGLE = "running_single"
    RUNNING_MULTI = "running_multi"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class RunMetadata:
    button_label: Optional[str] = None
    models: List[str] = field(default_factory=list)
    aggregator: Optional[str] = None
    iteration: int = 0


def _ensure_ephemeral_fields(s: SessionState) -> None:
    if not hasattr(s, "_run_phase"):
        s._run_phase = RunPhase.IDLE.value
    if not hasattr(s, "_run_meta"):
        s._run_meta = RunMetadata()


def start_single(s: SessionState, label: str) -> None:
    _ensure_ephemeral_fields(s)
    s._run_phase = RunPhase.RUNNING_SINGLE.value
    s._run_meta = RunMetadata(button_label=label, models=[label], aggregator=None, iteration=0)
    print(f"[CORE][run_state] -> RUNNING_SINGLE label={label}")


def start_multi(s: SessionState, models: List[str], aggregator: str, button_label: str | None = None) -> None:
    _ensure_ephemeral_fields(s)
    s._run_phase = RunPhase.RUNNING_MULTI.value
    s._run_meta = RunMetadata(button_label=(button_label or "Multi"), models=list(models), aggregator=aggregator, iteration=1)
    print(f"[CORE][run_state] -> RUNNING_MULTI models={models} agg={aggregator}")


def mark_final(s: SessionState) -> None:
    _ensure_ephemeral_fields(s)
    s._run_phase = RunPhase.COMPLETED.value
    print(f"[CORE][run_state] -> COMPLETED")


def mark_cancelled(s: SessionState) -> None:
    _ensure_ephemeral_fields(s)
    s._run_phase = RunPhase.CANCELLED.value
    print(f"[CORE][run_state] -> CANCELLED")


def clear(s: SessionState) -> None:
    _ensure_ephemeral_fields(s)
    s._run_phase = RunPhase.IDLE.value
    s._run_meta = RunMetadata()
    print(f"[CORE][run_state] -> IDLE")


__all__ = [
    "RunPhase",
    "RunMetadata",
    "start_single",
    "start_multi",
    "mark_final",
    "mark_cancelled",
    "clear",
]


def snapshot(s: SessionState) -> dict:
    """Return a read-only snapshot of the current run state for logging/diagnostics."""
    phase = getattr(s, "_run_phase", RunPhase.IDLE.value)
    meta = getattr(s, "_run_meta", RunMetadata())
    try:
        meta_dict = {
            "button_label": meta.button_label,
            "models": list(meta.models) if isinstance(meta.models, list) else [],
            "aggregator": meta.aggregator,
            "iteration": int(meta.iteration),
        }
    except Exception:
        meta_dict = {"button_label": None, "models": [], "aggregator": None, "iteration": 0}
    snap = {"phase": phase, "meta": meta_dict}
    print(f"[CORE][run_state] snapshot -> {snap}")
    return snap


__all__.append("snapshot")


