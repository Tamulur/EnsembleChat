import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ensemble_chat.core.history import ChatHistory
from ensemble_chat.core.sanitization import sanitize_pairs_for_display
from ensemble_chat.core.utils import CostTracker, timestamp_id
from ensemble_chat.core.paths import project_root


def _neutralize_angle_brackets_text(text: str):
    # Backwards-compatible alias
    from ensemble_chat.core.sanitization import neutralize_angle_brackets
    return neutralize_angle_brackets(text)


def _sanitize_pairs_for_display(pairs: List[Tuple[str, str]]):
    return sanitize_pairs_for_display(pairs)


class SessionState:
    def __init__(self):
        self.pdf_path: str | None = None
        self.chat_history = ChatHistory()
        self.cost_tracker = CostTracker()
        self.chat_id = timestamp_id()
        # Per-model chat history for tabs
        self.model_histories = {"ChatGPT": [], "Claude": [], "Gemini": []}
        # Resubmissions tab history (list of chatbot tuples)
        self.resubmissions_history = []
        # Settings (populated by caller)
        self.selected_openai_model = None
        self.selected_claude_model = None
        self.selected_gemini_model = None
        self.selected_aggregator = "Claude"
        self.temperature: float = 0.7
        self.notifications_enabled: bool = True
        # Ephemeral runtime fields (not serialized)
        self._run_id: int = 0
        self._current_task = None


# --- Session persistence (robust across Gradio resets) ---
SESSION_FILE = project_root() / "Session.json"


def _serialize_session(s: SessionState) -> Dict[str, Any]:
    return {
        "chat_id": s.chat_id,
        "pdf_path": s.pdf_path,
        "chat_history": s.chat_history.entries(),
        "model_histories": {k: list(v) for k, v in s.model_histories.items()},
        "resubmissions_history": list(s.resubmissions_history),
        "notifications_enabled": s.notifications_enabled,
        "temperature": s.temperature,
        "selected_models": {
            "openai": s.selected_openai_model,
            "claude": s.selected_claude_model,
            "gemini": s.selected_gemini_model,
            "aggregator": s.selected_aggregator,
        },
        # Keep a light summary of costs only
        "cost_total": s.cost_tracker.total_cost,
        "cost_per_model": s.cost_tracker.model_costs,
    }


def save_session(s: SessionState) -> None:
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(_serialize_session(s), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save session: {e}")


def _apply_loaded_session(s: SessionState, data: Dict[str, Any]) -> SessionState:
    try:
        chat_id = data.get("chat_id")
        if isinstance(chat_id, str) and chat_id:
            s.chat_id = chat_id
        pdf_path = data.get("pdf_path")
        if isinstance(pdf_path, str) and pdf_path:
            s.pdf_path = pdf_path
        entries = data.get("chat_history")
        if isinstance(entries, list):
            s.chat_history._entries = []
            for e in entries:
                role = e.get("role")
                text = e.get("text")
                if role in ("user", "assistant") and isinstance(text, str):
                    s.chat_history._entries.append({"role": role, "text": text})
        mh = data.get("model_histories")
        if isinstance(mh, dict):
            for key in ["ChatGPT", "Claude", "Gemini"]:
                seq = mh.get(key)
                if isinstance(seq, list):
                    normalized = []
                    for item in seq:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            a, b = item
                            if isinstance(a, str) and isinstance(b, str):
                                normalized.append((a, b))
                    s.model_histories[key] = normalized
        rh = data.get("resubmissions_history")
        if isinstance(rh, list):
            normalized_rh = []
            for item in rh:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    a, b = item
                    if isinstance(a, str) and isinstance(b, str):
                        normalized_rh.append((a, b))
            s.resubmissions_history = normalized_rh
        notif = data.get("notifications_enabled")
        if isinstance(notif, bool):
            s.notifications_enabled = notif
        temp = data.get("temperature")
        if isinstance(temp, (int, float)):
            s.temperature = float(temp)
        # Do not override provider selections; those persist separately
    except Exception as e:
        print(f"[WARN] Failed to apply loaded session: {e}")
    return s


def load_session_from_disk() -> Dict[str, Any] | None:
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load session: {e}")
    return None


__all__ = [
    "SessionState",
    "SESSION_FILE",
    "save_session",
    "load_session_from_disk",
    "_apply_loaded_session",
    "_sanitize_pairs_for_display",
]


