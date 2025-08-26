from typing import Any, Dict

from ensemble_chat.core.session_state import SessionState, save_session
from ensemble_chat.core.settings_manager import APP_SETTINGS, save_settings
from ensemble_chat.llm_providers import set_openai_model, set_claude_model, set_gemini_model


def set_openai(selection: str, s: SessionState) -> SessionState:
    print(f"[CORE][settings] set_openai -> {selection}")
    s.selected_openai_model = selection
    try:
        set_openai_model(str(selection).lower())
    except Exception as e:
        print(f"[WARN] Failed to set OpenAI model to lowercase id: {e}. Falling back to raw selection.")
        set_openai_model(str(selection))
    APP_SETTINGS["openai_model"] = selection
    save_settings(APP_SETTINGS)
    save_session(s)
    return s


def set_claude(selection: str, s: SessionState) -> SessionState:
    print(f"[CORE][settings] set_claude -> {selection}")
    s.selected_claude_model = selection
    set_claude_model(selection)
    APP_SETTINGS["claude_model"] = selection
    save_settings(APP_SETTINGS)
    save_session(s)
    return s


def set_gemini(selection: str, s: SessionState) -> SessionState:
    print(f"[CORE][settings] set_gemini -> {selection}")
    s.selected_gemini_model = selection
    set_gemini_model(selection)
    APP_SETTINGS["gemini_model"] = selection
    save_settings(APP_SETTINGS)
    save_session(s)
    return s


def set_aggregator(selection: str, s: SessionState) -> SessionState:
    print(f"[CORE][settings] set_aggregator -> {selection}")
    s.selected_aggregator = selection
    APP_SETTINGS["aggregator"] = selection
    save_settings(APP_SETTINGS)
    save_session(s)
    return s


def set_temperature(val: float, s: SessionState) -> SessionState:
    print(f"[CORE][settings] set_temperature -> {val}")
    try:
        s.temperature = float(val)
    except Exception as e:
        print(f"[WARN] Invalid temperature '{val}': {e}. Using default 0.7")
        s.temperature = 0.7
    APP_SETTINGS["temperature"] = s.temperature
    save_settings(APP_SETTINGS)
    save_session(s)
    return s


def set_notifications(enabled: bool, s: SessionState) -> SessionState:
    print(f"[CORE][settings] set_notifications -> {enabled}")
    s.notifications_enabled = bool(enabled)
    APP_SETTINGS["notifications"] = s.notifications_enabled
    save_settings(APP_SETTINGS)
    save_session(s)
    return s


__all__ = [
    "set_openai",
    "set_claude",
    "set_gemini",
    "set_aggregator",
    "set_temperature",
    "set_notifications",
]


