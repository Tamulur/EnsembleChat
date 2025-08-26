from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union, Any


@dataclass(frozen=True)
class StatusEvent:
    text: str

    def to_dict(self) -> Dict:
        return {"type": "status", "text": self.text}


@dataclass(frozen=True)
class FinalEvent:
    text: str

    def to_dict(self) -> Dict:
        return {"type": "final", "text": self.text}


@dataclass(frozen=True)
class CancelledEvent:
    def to_dict(self) -> Dict:
        return {"type": "cancelled"}


@dataclass(frozen=True)
class ProgressEvent:
    done: int
    total: int

    def to_dict(self) -> Dict:
        return {"type": "progress", "done": self.done, "total": self.total}


@dataclass(frozen=True)
class IterationEvent:
    iteration: int

    def to_dict(self) -> Dict:
        return {"type": "iteration", "iteration": self.iteration}


@dataclass(frozen=True)
class ErrorEvent:
    message: str

    def to_dict(self) -> Dict:
        return {"type": "error", "message": self.message}


@dataclass(frozen=True)
class RunStartedEvent:
    label: str

    def to_dict(self) -> Dict:
        return {"type": "run_started", "label": self.label}


@dataclass(frozen=True)
class RunCancelledEvent:
    label: str

    def to_dict(self) -> Dict:
        return {"type": "run_cancelled", "label": self.label}


@dataclass(frozen=True)
class RunCompletedEvent:
    label: str

    def to_dict(self) -> Dict:
        return {"type": "run_completed", "label": self.label}


LogicEvent = Union[
    StatusEvent,
    FinalEvent,
    CancelledEvent,
    ProgressEvent,
    IterationEvent,
    ErrorEvent,
    RunStartedEvent,
    RunCancelledEvent,
    RunCompletedEvent,
]


def event_from_raw(raw: Union[Dict[str, Any], LogicEvent]) -> LogicEvent:
    """Best-effort conversion from existing events (dict or already-typed) to typed events."""
    # If it's already a typed event, return as-is
    if isinstance(raw, (StatusEvent, FinalEvent, CancelledEvent, ProgressEvent, IterationEvent, ErrorEvent)):
        return raw
    t = (raw or {}).get("type")
    if t == "status":
        return StatusEvent(text=str(raw.get("text", "")))
    if t == "final":
        return FinalEvent(text=str(raw.get("text", "")))
    if t == "cancelled":
        return CancelledEvent()
    if t == "progress":
        try:
            return ProgressEvent(done=int(raw.get("done", 0)), total=int(raw.get("total", 0)))
        except Exception:
            return ProgressEvent(done=0, total=0)
    if t == "iteration":
        try:
            return IterationEvent(iteration=int(raw.get("iteration", 0)))
        except Exception:
            return IterationEvent(iteration=0)
    if t == "error":
        return ErrorEvent(message=str(raw.get("message", "")))
    if t == "run_started":
        return RunStartedEvent(label=str(raw.get("label", "")))
    if t == "run_cancelled":
        return RunCancelledEvent(label=str(raw.get("label", "")))
    if t == "run_completed":
        return RunCompletedEvent(label=str(raw.get("label", "")))
    # Fallback: treat as status to avoid UI changes
    return StatusEvent(text=str(raw))


__all__ = [
    "StatusEvent",
    "FinalEvent",
    "CancelledEvent",
    "ProgressEvent",
    "IterationEvent",
    "ErrorEvent",
    "RunStartedEvent",
    "RunCancelledEvent",
    "RunCompletedEvent",
    "LogicEvent",
    "event_from_raw",
]


def get_event_type(ev: Union[LogicEvent, Dict[str, Any], None]) -> str:
    """Return canonical type string (e.g., 'status', 'final') for typed or dict events."""
    if isinstance(ev, StatusEvent):
        return "status"
    if isinstance(ev, FinalEvent):
        return "final"
    if isinstance(ev, CancelledEvent):
        return "cancelled"
    if isinstance(ev, ProgressEvent):
        return "progress"
    if isinstance(ev, IterationEvent):
        return "iteration"
    if isinstance(ev, ErrorEvent):
        return "error"
    if isinstance(ev, RunStartedEvent):
        return "run_started"
    if isinstance(ev, RunCancelledEvent):
        return "run_cancelled"
    if isinstance(ev, RunCompletedEvent):
        return "run_completed"
    try:
        return str((ev or {}).get("type") or "")
    except Exception:
        return ""


__all__.append("get_event_type")


