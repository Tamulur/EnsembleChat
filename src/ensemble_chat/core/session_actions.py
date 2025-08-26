from typing import Tuple

from ensemble_chat.core.session_state import SessionState, save_session
from ensemble_chat.core.settings_manager import APP_SETTINGS
from ensemble_chat.llm_providers import set_openai_model, set_claude_model, set_gemini_model


def set_pdf(path: str | None, s: SessionState) -> Tuple[SessionState, str]:
    print(f"[CORE][session_actions] set_pdf -> {path}")
    if path is not None:
        s.pdf_path = path
        save_session(s)
        return s, "switch_tab"
    return s, ""


def apply_provider_models(s: SessionState) -> SessionState:
    print(f"[CORE][session_actions] apply_provider_models: openai={s.selected_openai_model} claude={s.selected_claude_model} gemini={s.selected_gemini_model}")
    try:
        set_openai_model(str(s.selected_openai_model).lower())
    except Exception as e:
        print(f"[WARN] Failed to set initial OpenAI model to lowercase id: {e}. Falling back to raw selection.")
        set_openai_model(str(s.selected_openai_model))
    set_claude_model(s.selected_claude_model)
    set_gemini_model(s.selected_gemini_model)
    return s


def reset_session_state(s: SessionState) -> SessionState:
    print(f"[CORE][session_actions] reset_session_state")
    s = SessionState()
    # Rehydrate selections from persisted app settings
    try:
        s.selected_openai_model = APP_SETTINGS.get("openai_model", s.selected_openai_model)
        s.selected_claude_model = APP_SETTINGS.get("claude_model", s.selected_claude_model)
        s.selected_gemini_model = APP_SETTINGS.get("gemini_model", s.selected_gemini_model)
        s.selected_aggregator = APP_SETTINGS.get("aggregator", s.selected_aggregator)
        s.temperature = float(APP_SETTINGS.get("temperature", s.temperature))
        s.notifications_enabled = bool(APP_SETTINGS.get("notifications", s.notifications_enabled))
        apply_provider_models(s)
    except Exception as e:
        print(f"[WARN] Failed to rehydrate selections or apply provider models: {e}")
    save_session(s)
    return s


__all__ = [
    "set_pdf",
    "apply_provider_models",
    "reset_session_state",
]


