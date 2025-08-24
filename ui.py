import gradio as gr

from model_configs import MODEL_CONFIGS
from settings_manager import APP_SETTINGS
from session_state import SessionState, load_session_from_disk, _apply_loaded_session
from ui_layout import build_base_layout
from ui_handlers import wire_events


def build_ui():
    # Prepare initial state (load from disk if present)
    initial_state = SessionState()
    initial_state.selected_openai_model = APP_SETTINGS.get("openai_model", MODEL_CONFIGS["OpenAI"][0])
    initial_state.selected_claude_model = APP_SETTINGS.get("claude_model", MODEL_CONFIGS["Claude"][0])
    initial_state.selected_gemini_model = APP_SETTINGS.get("gemini_model", MODEL_CONFIGS["Gemini"][0])
    initial_state.selected_aggregator = APP_SETTINGS.get("aggregator", "Claude")
    initial_state.temperature = float(APP_SETTINGS.get("temperature", 0.7))
    initial_state.notifications_enabled = bool(APP_SETTINGS.get("notifications", True))

    loaded = load_session_from_disk()
    if isinstance(loaded, dict):
        initial_state = _apply_loaded_session(initial_state, loaded)
        # Bring back cost totals if present
        try:
            if isinstance(loaded.get("cost_total"), (int, float)):
                initial_state.cost_tracker.total_cost = float(loaded.get("cost_total"))
            if isinstance(loaded.get("cost_per_model"), dict):
                for k, v in loaded.get("cost_per_model", {}).items():
                    if isinstance(v, (int, float)):
                        initial_state.cost_tracker.model_costs[k] = float(v)
        except Exception:
            pass

    demo, ui = build_base_layout(initial_state, APP_SETTINGS, MODEL_CONFIGS, wire=wire_events)
    return demo


__all__ = [
    "build_ui",
]


