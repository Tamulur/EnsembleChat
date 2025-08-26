from typing import List, Tuple, Any

from ensemble_chat.core.session_state import SessionState, _sanitize_pairs_for_display


def cost_line(s: SessionState, model_label: str) -> str:
    val = f"**Cost so far:** ${s.cost_tracker.get_model_cost(model_label):.4f}"
    print(f"[CORE][selectors] cost_line {model_label} -> {val}")
    return val


def model_display(s: SessionState, model_label: str) -> List[Tuple[str, str]]:
    def _ensure_text(x: Any) -> str:
        if isinstance(x, str):
            return x
        try:
            txt = getattr(x, "text", None)
            if isinstance(txt, str):
                return txt
        except Exception:
            pass
        return str(x)

    raw_pairs = s.model_histories.get(model_label, [])
    normalized: List[Tuple[str, str]] = []
    for left, right in raw_pairs:
        left_s = _ensure_text(left)
        right_s = _ensure_text(right)
        normalized.append((left_s, right_s))
    pairs = _sanitize_pairs_for_display(normalized)
    print(f"[CORE][selectors] model_display {model_label} -> {len(pairs)} pairs (normalized)")
    return pairs


def resubmissions_display(s: SessionState) -> List[Tuple[str, str]]:
    pairs = _sanitize_pairs_for_display(s.resubmissions_history)
    print(f"[CORE][selectors] resubmissions_display -> {len(pairs)} entries")
    return pairs


def notifications_flag(s: SessionState) -> str:
    flag = ("done" if s.notifications_enabled else "")
    print(f"[CORE][selectors] notifications_flag -> '{flag}'")
    return flag


__all__ = [
    "cost_line",
    "model_display",
    "resubmissions_display",
    "notifications_flag",
]


def active_button_label(s: SessionState) -> str:
    try:
        label = getattr(getattr(s, "_run_meta", None), "button_label", None)
        out = str(label) if label is not None else ""
        print(f"[CORE][selectors] active_button_label -> '{out}'")
        return out
    except Exception:
        return ""


def active_button_elem_id(s: SessionState) -> str:
    label = active_button_label(s)
    mapping = {
        "ChatGPT": "btn_chatgpt",
        "Claude": "btn_claude",
        "Gemini": "btn_gemini",
        "ChatGPT & Gemini": "btn_chatgpt_gemini",
        "All": "btn_all",
    }
    elem_id = mapping.get(label, "")
    print(f"[CORE][selectors] active_button_elem_id('{label}') -> '{elem_id}'")
    return elem_id


__all__.append("active_button_label")
__all__.append("active_button_elem_id")


