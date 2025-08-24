import asyncio
import os
import json

import gradio as gr
import warnings
from pathlib import Path
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning, message="You have not specified a value for the `type` parameter.*")
import html
import re
_ZWSP = "\u200B"

def _neutralize_angle_brackets_text(text: str):
    if not isinstance(text, str):
        return text
    decoded = html.unescape(text)
    decoded = re.sub(r"<(?=[A-Za-z/])", "<" + _ZWSP, decoded)
    decoded = re.sub(r"(?<=[A-Za-z0-9/])>", _ZWSP + ">", decoded)
    return decoded

from utils import CostTracker, save_chat, timestamp_id, BUDGET_LIMIT, create_user_friendly_error_message
from frontend_js import (
    JS_ALIGN_ON_CHANGE,
    JS_SCROLL_FIX_AFTER_EVENT,
    JS_PRESERVE_TAB_SCROLL,
    JS_NOTIFY_IF_FLAG,
    JS_PREPARE_NOTIFICATIONS,
    JS_SELECT_TAB_ATTACHMENTS_ON_LOAD,
    JS_SWITCH_TO_CHAT_TAB,
    JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL,
)
from history import ChatHistory
from proposer import call_proposer, call_synthesis
from aggregator import call_aggregator, first_non_empty_line, text_after_first_line, format_proposal_packet
from llm_providers import set_openai_model, set_claude_model, set_gemini_model

# Constants
ICON_DIR = Path(__file__).parent / "Icons"

BUTTONS = [
    "ChatGPT",
    "Claude",
    "Gemini",
    "ChatGPT & Gemini",
    "All",
]

# Mapping of button label to icon path
ICON_MAP = {
    "ChatGPT": str(ICON_DIR / "OpenAI.png"),
    "Claude": str(ICON_DIR / "Claude.png"),
    "Gemini": str(ICON_DIR / "Gemini.png"),
    "ChatGPT & Gemini": str(ICON_DIR / "ChatGPT_Gemini.png"),
    "All": str(ICON_DIR / "All.png"),
}


# --- Configuration loading for model dropdowns ---
CONFIG_DIR = Path(__file__).parent / "Configurations"

DEFAULT_MODELS = {
    "OpenAI": ["GPT-5", "GPT-5-mini", "o3", "GPT-4.1"],
    "Claude": ["claude-sonnet-4-0"],
    "Gemini": ["gemini-2.5-pro"],
}


def _read_models_from_file(filename: str, fallback: list[str]) -> list[str]:
    cfg_path = CONFIG_DIR / filename
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models")
        if isinstance(models, list) and all(isinstance(x, str) for x in models) and models:
            return models
        else:
            print(f"[WARN] '{filename}' missing valid 'models' list. Using defaults.")
    except Exception as e:
        print(f"[WARN] Failed to read '{filename}': {e}. Using defaults.")
    return fallback


def load_model_configurations() -> dict[str, list[str]]:
    return {
        "OpenAI": _read_models_from_file("OpenAI.json", DEFAULT_MODELS["OpenAI"]),
        "Claude": _read_models_from_file("Claude.json", DEFAULT_MODELS["Claude"]),
        "Gemini": _read_models_from_file("Gemini.json", DEFAULT_MODELS["Gemini"]),
    }


MODEL_CONFIGS = load_model_configurations()

MULTI_BUTTON_MODELS = {
    "ChatGPT & Gemini": ["ChatGPT", "Gemini"],
    "All": ["ChatGPT", "Claude", "Gemini"],
}

# Shared LaTeX delimiters configuration for all Chatbot components
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "[", "right": "]", "display": True},  # Support OpenAI's format
]

# --- Settings persistence (Settings.json) ---
SETTINGS_FILE = Path(__file__).parent / "Settings.json"


def _default_settings() -> dict:
    return {
        "openai_model": MODEL_CONFIGS["OpenAI"][0],
        "claude_model": MODEL_CONFIGS["Claude"][0],
        "gemini_model": MODEL_CONFIGS["Gemini"][0],
        "aggregator": "Claude",
        "temperature": 0.7,
        "notifications": True,
    }


def _validate_and_merge_settings(raw: dict) -> dict:
    defaults = _default_settings()
    merged = dict(defaults)

    try:
        if isinstance(raw.get("openai_model"), str) and raw["openai_model"] in MODEL_CONFIGS["OpenAI"]:
            merged["openai_model"] = raw["openai_model"]
        if isinstance(raw.get("claude_model"), str) and raw["claude_model"] in MODEL_CONFIGS["Claude"]:
            merged["claude_model"] = raw["claude_model"]
        if isinstance(raw.get("gemini_model"), str) and raw["gemini_model"] in MODEL_CONFIGS["Gemini"]:
            merged["gemini_model"] = raw["gemini_model"]
        if isinstance(raw.get("aggregator"), str) and raw["aggregator"] in ["ChatGPT", "Claude", "Gemini"]:
            merged["aggregator"] = raw["aggregator"]
        temp = raw.get("temperature")
        if isinstance(temp, (int, float)):
            merged["temperature"] = max(0.0, min(1.0, float(temp)))
        if isinstance(raw.get("notifications"), bool):
            merged["notifications"] = raw["notifications"]
    except Exception as e:
        print("[WARN] Settings validation error:", e)

    return merged


def load_settings() -> dict:
    defaults = _default_settings()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            settings = _validate_and_merge_settings(raw if isinstance(raw, dict) else {})
        except Exception as e:
            print(f"[WARN] Failed to read Settings.json: {e}. Using defaults.")
            settings = defaults
    else:
        settings = defaults
    # Ensure file exists and is normalized
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write Settings.json: {e}")
    return settings


def save_settings(settings: dict) -> None:
    normalized = _validate_and_merge_settings(settings if isinstance(settings, dict) else {})
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save Settings.json: {e}")


APP_SETTINGS = load_settings()


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
        # Settings
        self.selected_openai_model = APP_SETTINGS.get("openai_model", MODEL_CONFIGS["OpenAI"][0])
        self.selected_claude_model = APP_SETTINGS.get("claude_model", MODEL_CONFIGS["Claude"][0])
        self.selected_gemini_model = APP_SETTINGS.get("gemini_model", MODEL_CONFIGS["Gemini"][0])
        self.selected_aggregator = APP_SETTINGS.get("aggregator", "Claude")
        self.temperature: float = float(APP_SETTINGS.get("temperature", 0.7))
        self.notifications_enabled: bool = bool(APP_SETTINGS.get("notifications", True))


# --- Session persistence (robust across Gradio resets) ---
SESSION_FILE = Path(__file__).parent / "Session.json"


def _sanitize_pairs_for_display(pairs):
    sanitized = []
    for left, right in pairs:
        left_s = _neutralize_angle_brackets_text(left)
        right_s = _neutralize_angle_brackets_text(right)
        sanitized.append((left_s, right_s))
    return sanitized


def _serialize_session(s: SessionState) -> dict:
    return {
        "chat_id": s.chat_id,
        "pdf_path": s.pdf_path,
        "chat_history": s.chat_history.entries(),
        "model_histories": {
            k: list(v) for k, v in s.model_histories.items()
        },
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


def _apply_loaded_session(s: SessionState, data: dict) -> SessionState:
    try:
        chat_id = data.get("chat_id")
        if isinstance(chat_id, str) and chat_id:
            s.chat_id = chat_id
        pdf_path = data.get("pdf_path")
        if isinstance(pdf_path, str) and pdf_path:
            s.pdf_path = pdf_path
        entries = data.get("chat_history")
        if isinstance(entries, list):
            # Replace internal entries directly
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
        # Do not override provider selections from Settings.json; those persist separately
    except Exception as e:
        print(f"[WARN] Failed to apply loaded session: {e}")
    return s


def load_session_from_disk() -> dict | None:
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load session: {e}")
    return None


async def _handle_single(model_label: str, user_input: str, state: SessionState):
    try:
        reply_text = await call_proposer(
            model_label,
            user_input,
            state.chat_history.entries(),
            state.pdf_path,
            state.cost_tracker,
            retries=5,
            temperature=state.temperature,
        )
    except Exception as e:
        print("[ERROR] single LLM", model_label, e)
        reply_text = create_user_friendly_error_message(e, model_label)
    # Update chat history and per-model tab history
    state.chat_history.add_assistant(reply_text)
    state.model_histories[model_label].append((user_input, reply_text))
    save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
    return state.chat_history.as_display()


def build_ui():
    from frontend_css import CSS_GLOBAL
    # Prepare initial state (load from disk if present)
    initial_state = SessionState()
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

    with gr.Blocks(css=CSS_GLOBAL) as demo:
        # gr.Markdown("## EnsembleChat")
        state = gr.State(initial_state)
        # Top-right "New Chat" button (absolutely positioned via CSS, outside Tabs to avoid tab duplication)
        new_chat_btn = gr.Button(value="New Chat", elem_id="btn_new_chat")

        # ---------- TABS LAYOUT ----------
        with gr.Tabs(selected=4) as tabs:
            # ---- Main Chat tab ----
            with gr.Tab("Chat"):
                chat = gr.Chatbot(
                    height=630,
                    elem_id="chat_interface",
                    autoscroll=False,
                    show_label=False,
                    latex_delimiters=LATEX_DELIMITERS,
                    value=initial_state.chat_history.as_display(),
                )
                # Whenever the chat value changes, align the beginning of the newest
                # assistant message to the top of the visible chat area.
                chat.change(
                    None,
                    inputs=[],
                    outputs=[],
                    js=JS_ALIGN_ON_CHANGE,
                )
                status_display = gr.Markdown("", visible=False)
                user_box = gr.Textbox(label="You", value="Please explain this paper to me.")
                notify_flag = gr.Textbox(value="", visible=False)

                with gr.Row():
                    # Assign stable CSS ids so each button icon can be styled individually via CSS
                    BUTTON_ID_MAP = {
                        "ChatGPT": "btn_chatgpt",
                        "Claude": "btn_claude",
                        "Gemini": "btn_gemini",
                        "ChatGPT & Gemini": "btn_chatgpt_gemini",
                        "All": "btn_all",
                    }
                    btns = [
                        gr.Button(value=label, icon=ICON_MAP.get(label), elem_id=BUTTON_ID_MAP[label])
                        for label in BUTTONS
                    ]

            # ---- Per-model tabs ----
            MODEL_TAB_HEIGHT = 700

            with gr.Tab("ChatGPT"):
                chatgpt_cost = gr.Markdown(f"**Cost so far:** ${initial_state.cost_tracker.get_model_cost('ChatGPT'):.4f}", elem_id="chatgpt_cost")
                chatgpt_view = gr.Chatbot(
                    label="ChatGPT Output",
                    height=MODEL_TAB_HEIGHT,
                    value=_sanitize_pairs_for_display(initial_state.model_histories["ChatGPT"]),
                    autoscroll=False,
                    elem_id="chatgpt_view",
                    latex_delimiters=LATEX_DELIMITERS
                )

            with gr.Tab("Claude"):
                claude_cost = gr.Markdown(f"**Cost so far:** ${initial_state.cost_tracker.get_model_cost('Claude'):.4f}", elem_id="claude_cost")
                claude_view = gr.Chatbot(
                    label="Claude Output",
                    height=MODEL_TAB_HEIGHT,
                    value=_sanitize_pairs_for_display(initial_state.model_histories["Claude"]),
                    autoscroll=False,
                    elem_id="claude_view",
                    latex_delimiters=LATEX_DELIMITERS
                )

            with gr.Tab("Gemini"):
                gemini_cost = gr.Markdown(f"**Cost so far:** ${initial_state.cost_tracker.get_model_cost('Gemini'):.4f}", elem_id="gemini_cost")
                gemini_view = gr.Chatbot(
                    label="Gemini Output",
                    height=MODEL_TAB_HEIGHT,
                    value=_sanitize_pairs_for_display(initial_state.model_histories["Gemini"]),
                    autoscroll=False,
                    elem_id="gemini_view",
                    latex_delimiters=LATEX_DELIMITERS)

            # ---- Attachments tab ----
            with gr.Tab("Attachments"):
                with gr.Row():
                    initial_pdf_value = initial_state.pdf_path if (initial_state.pdf_path and os.path.isfile(initial_state.pdf_path)) else None
                    pdf_input = gr.File(label="Select PDF", file_types=[".pdf"], type="filepath", value=initial_pdf_value)
                # Hidden component to signal tab switch
                tab_switch_signal = gr.Textbox(value="", visible=False)

            # ---- Resubmissions tab ----
            with gr.Tab("Resubmissions"):
                resub_view = gr.Chatbot(label="Resubmissions", height=800, value=_sanitize_pairs_for_display(initial_state.resubmissions_history), autoscroll=False, elem_id="resub_view",
                                        latex_delimiters=LATEX_DELIMITERS)

            # ---- Settings tab ----
            with gr.Tab("Settings"):
                with gr.Row():
                    openai_model_dropdown = gr.Dropdown(
                        choices=MODEL_CONFIGS["OpenAI"],
                        value=(APP_SETTINGS.get("openai_model") if APP_SETTINGS.get("openai_model") in MODEL_CONFIGS["OpenAI"] else MODEL_CONFIGS["OpenAI"][0]),
                        label="OpenAI model",
                        interactive=True,
                    )
                with gr.Row():
                    claude_model_dropdown = gr.Dropdown(
                        choices=MODEL_CONFIGS["Claude"],
                        value=(APP_SETTINGS.get("claude_model") if APP_SETTINGS.get("claude_model") in MODEL_CONFIGS["Claude"] else MODEL_CONFIGS["Claude"][0]),
                        label="Claude model",
                        interactive=True,
                    )
                with gr.Row():
                    gemini_model_dropdown = gr.Dropdown(
                        choices=MODEL_CONFIGS["Gemini"],
                        value=(APP_SETTINGS.get("gemini_model") if APP_SETTINGS.get("gemini_model") in MODEL_CONFIGS["Gemini"] else MODEL_CONFIGS["Gemini"][0]),
                        label="Gemini model",
                        interactive=True,
                    )
                with gr.Row():
                    aggregator_dropdown = gr.Dropdown(
                        choices=["ChatGPT", "Claude", "Gemini"],
                        value=(APP_SETTINGS.get("aggregator") if APP_SETTINGS.get("aggregator") in ["ChatGPT", "Claude", "Gemini"] else "Claude"),
                        label="Aggregator",
                        interactive=True,
                    )
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=float(APP_SETTINGS.get("temperature", 0.7)),
                        label="Temperature",
                        interactive=True,
                    )
                with gr.Row():
                    notifications_checkbox = gr.Checkbox(
                        value=bool(APP_SETTINGS.get("notifications", True)),
                        label="Notifications",
                        interactive=True,
                    )

        # Update pdf path
        def _set_pdf(file, s: SessionState):
            if file is not None:
                s.pdf_path = file
                print(f"[DEBUG] PDF selected: {file}, sending switch_tab signal")
                save_session(s)
                return s, "switch_tab"  # Signal to switch tab
            print(f"[DEBUG] No PDF selected, file is: {file}")
            return s, ""  # No signal if no file

        pdf_change_evt = pdf_input.change(_set_pdf, inputs=[pdf_input, state], outputs=[state, tab_switch_signal])
        # After selecting a PDF, switch to the Chat tab only if a file was selected
        tab_switch_signal.change(
            None, 
            inputs=[tab_switch_signal], 
            outputs=None, 
            js=JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL
        )

        # --- Settings handlers ---
        def _set_openai_model(selection: str, s: SessionState):
            s.selected_openai_model = selection
            # Apply to provider layer immediately
            set_openai_model(selection)
            # Persist
            APP_SETTINGS["openai_model"] = selection
            save_settings(APP_SETTINGS)
            save_session(s)
            return s

        openai_model_dropdown.change(_set_openai_model, inputs=[openai_model_dropdown, state], outputs=state)

        def _set_claude_model(selection: str, s: SessionState):
            s.selected_claude_model = selection
            set_claude_model(selection)
            APP_SETTINGS["claude_model"] = selection
            save_settings(APP_SETTINGS)
            save_session(s)
            return s

        claude_model_dropdown.change(_set_claude_model, inputs=[claude_model_dropdown, state], outputs=state)

        def _set_gemini_model(selection: str, s: SessionState):
            s.selected_gemini_model = selection
            set_gemini_model(selection)
            APP_SETTINGS["gemini_model"] = selection
            save_settings(APP_SETTINGS)
            save_session(s)
            return s

        gemini_model_dropdown.change(_set_gemini_model, inputs=[gemini_model_dropdown, state], outputs=state)

        def _set_aggregator(selection: str, s: SessionState):
            s.selected_aggregator = selection
            APP_SETTINGS["aggregator"] = selection
            save_settings(APP_SETTINGS)
            save_session(s)
            return s

        aggregator_dropdown.change(_set_aggregator, inputs=[aggregator_dropdown, state], outputs=state)

        def _set_temperature(val: float, s: SessionState):
            try:
                s.temperature = float(val)
            except Exception:
                s.temperature = 0.7
            APP_SETTINGS["temperature"] = s.temperature
            save_settings(APP_SETTINGS)
            save_session(s)
            return s

        temperature_slider.change(_set_temperature, inputs=[temperature_slider, state], outputs=state)

        def _set_notifications(enabled: bool, s: SessionState):
            s.notifications_enabled = bool(enabled)
            APP_SETTINGS["notifications"] = s.notifications_enabled
            save_settings(APP_SETTINGS)
            save_session(s)
            return s

        notifications_checkbox.change(_set_notifications, inputs=[notifications_checkbox, state], outputs=state)

        # --- New Chat (reset session) ---
        def _reset_session(s: SessionState):
            try:
                if SESSION_FILE.exists():
                    SESSION_FILE.unlink(missing_ok=True)
            except Exception as e:
                print(f"[WARN] Failed to delete session file: {e}")
            # Fresh state
            s = SessionState()
            # Apply provider selections to backend
            try:
                set_openai_model(s.selected_openai_model)
                set_claude_model(s.selected_claude_model)
                set_gemini_model(s.selected_gemini_model)
            except Exception:
                pass
            # Persist empty fresh session
            save_session(s)
            # Compose outputs to clear all views
            def _cost_line(label: str) -> str:
                return f"**Cost so far:** ${s.cost_tracker.get_model_cost(label):.4f}"
            return (
                s.chat_history.as_display(),                 # chat
                gr.update(value="", visible=False),         # status_display
                gr.update(value=_cost_line("ChatGPT")),      # chatgpt_cost
                gr.update(value=[]),                         # chatgpt_view
                gr.update(value=_cost_line("Claude")),      # claude_cost
                gr.update(value=[]),                         # claude_view
                gr.update(value=_cost_line("Gemini")),      # gemini_cost
                gr.update(value=[]),                         # gemini_view
                gr.update(value=[]),                         # resub_view
                "",                                         # notify_flag
                s,                                           # state
                gr.update(value=None),                       # pdf_input (clear)
            )

        new_chat_btn.click(
            _reset_session,
            inputs=state,
            outputs=[
                chat,
                status_display,
                chatgpt_cost,
                chatgpt_view,
                claude_cost,
                claude_view,
                gemini_cost,
                gemini_view,
                resub_view,
                notify_flag,
                state,
                pdf_input,
            ],
        )

        # --- User message handling ---
        def _add_user_and_clear(user_input: str, s: SessionState):
            """Handle immediate chat updates and textbox clearing.

            1. If the user types a **non-empty** prompt → add it to history (pending assistant) and clear box.
            2. If the input is **empty** → treat as *redo last reply*: remove the most recent assistant
               response (if any) so the preceding user input becomes the active prompt again.
            """
            if not user_input:
                # Empty input → attempt redo (delete last assistant reply)
                s.chat_history.remove_last_assistant()
                save_session(s)
                return s.chat_history.as_display(), "", s

            # Normal new user input
            s.chat_history.add_user(user_input)
            save_session(s)
            return s.chat_history.as_display(), "", s


        # Attach handlers for each button (three-stage: instant update, status updates, then final processing)
        for btn in btns:
            # Stage 1: quick, add user message and clear textbox
            click_event = btn.click(
                _add_user_and_clear,
                inputs=[user_box, state],
                outputs=[chat, user_box, state],
                show_progress=False,
            )
            
            # Stage 2: long-running processing with status updates
            def _make_process(lbl):
                async def _handler(s: SessionState, current_file):
                    # Rebind PDF from client-side file input if present (survives browser UI)
                    try:
                        if current_file:
                            s.pdf_path = current_file
                    except Exception:
                        pass
                    # Helpers to build update objects for each model tab
                    def _cost_line(label: str) -> str:
                        return f"**Cost so far:** ${s.cost_tracker.get_model_cost(label):.4f}"

                    def _sanitize_chatbot_pairs(pairs):
                        sanitized = []
                        for left, right in pairs:
                            left_s = _neutralize_angle_brackets_text(left)
                            right_s = _neutralize_angle_brackets_text(right)
                            sanitized.append((left_s, right_s))
                        return sanitized

                    def _model_and_cost_updates():
                        return (
                            gr.update(value=_cost_line("ChatGPT")),
                            gr.update(value=_sanitize_chatbot_pairs(s.model_histories["ChatGPT"])),
                            gr.update(value=_cost_line("Claude")),
                            gr.update(value=_sanitize_chatbot_pairs(s.model_histories["Claude"])),
                            gr.update(value=_cost_line("Gemini")),
                            gr.update(value=_sanitize_chatbot_pairs(s.model_histories["Gemini"])),
                            gr.update(value=_sanitize_chatbot_pairs(s.resubmissions_history)),
                        )
                    # Retrieve last user message (just appended by _add_user_and_clear)
                    last_user = None
                    for entry in reversed(s.chat_history.entries()):
                        if entry["role"] == "user":
                            last_user = entry["text"]
                            break
                    if last_user is None:
                        yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates(), ""
                        return

                    # Budget guard (rough, before making calls we estimate slight cost)
                    if s.cost_tracker.will_exceed_budget(0.05):
                        warn = "Budget exceeded ($5). Start a new session or change selection."
                        disp = s.chat_history.as_display()
                        disp.append((None, warn))
                        yield disp, gr.update(value="", visible=False), *_model_and_cost_updates(), ""
                        return

                    if lbl in ["ChatGPT", "Claude", "Gemini"]:
                        yield s.chat_history.as_display(), gr.update(value="**Status:** Waiting for " + lbl + "…", visible=True), *_model_and_cost_updates(), ""
                        result = await _handle_single(lbl, last_user, s)
                        save_session(s)
                        yield result, gr.update(value="", visible=False), *_model_and_cost_updates(), ("done" if s.notifications_enabled else "")
                    else:
                        models = MULTI_BUTTON_MODELS[lbl]
                        
                        # Create a generator for status updates
                        async def status_generator():
                            # Proposer phase (concurrent)
                            async def proposer_task(model):
                                try:
                                    # Return model name with result for easier tracking
                                    result = await call_proposer(
                                        model,
                                        last_user,
                                        s.chat_history.entries(),
                                        s.pdf_path,
                                        s.cost_tracker,
                                        retries=5,
                                        temperature=s.temperature,
                                    )
                                    return model, result
                                except Exception as e:
                                    print("[ERROR] proposer", model, e)
                                    return model, create_user_friendly_error_message(e, model)

                            tasks = [proposer_task(m) for m in models]
                            proposals_by_model = {}
                            num_models = len(models)

                            yield s.chat_history.as_display(), gr.update(
                                value=f"**Status:** Collecting replies (0/{num_models})...", visible=True), *_model_and_cost_updates(), ""

                            for i, future in enumerate(asyncio.as_completed(tasks)):
                                model, proposal = await future
                                proposals_by_model[model] = proposal

                                # Update status message with progress
                                status_msg = f"**Status:** Collecting replies ({i + 1}/{num_models})..."
                                yield s.chat_history.as_display(), gr.update(value=status_msg,
                                                                              visible=True), *_model_and_cost_updates(), ""

                            # Re-order proposals to match original model list
                            proposals = [proposals_by_model[m] for m in models]
                            # ---- DEBUG: log each proposer reply ----
                            for m, p in zip(models, proposals):
                                # Truncate and replace newlines for logging
                                p_log = p.replace('\n', ' ')
                                if len(p_log) > 100:
                                    p_log = p_log[:97] + "..."
                                print("[PROPOSAL]", m, p_log)
                                s.model_histories[m].append((last_user, p))

                            # Aggregator iterations
                            for iteration in range(1, 6):
                                yield s.chat_history.as_display(), gr.update(value=f"**Status:** Aggregating replies, iteration {iteration}…", visible=True), *_model_and_cost_updates(), ""
                                
                                try:
                                    agg_out = await call_aggregator(
                                        proposals,
                                        last_user,
                                        s.chat_history.entries(),
                                        s.pdf_path,
                                        s.cost_tracker,
                                        iteration,
                                        s.selected_aggregator,
                                        temperature=s.temperature,
                                    )
                                except Exception as e:
                                    print(f"[ERROR] aggregator {s.selected_aggregator} iteration {iteration}: {e}")
                                    
                                    # On aggregator failure, fall back to showing proposals
                                    error_message = create_user_friendly_error_message(e, s.selected_aggregator)
                                    fallback_reply = f"**Aggregation failed:** {error_message}\n\n**Here are the individual proposals:**\n\n"
                                    for i, proposal in enumerate(proposals, 1):
                                        # Show all proposals, including error messages (which are now user-friendly)
                                        fallback_reply += f"**Proposal {i}:**\n{proposal}\n\n"
                                    
                                    s.chat_history.add_assistant(fallback_reply.strip())
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    save_session(s)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates(), ("done" if s.notifications_enabled else "")
                                    return

                                first = first_non_empty_line(agg_out).lower()
                                if "final" in first:
                                    final_reply = text_after_first_line(agg_out)
                                    s.chat_history.add_assistant(final_reply)
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    save_session(s)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates(), ("done" if s.notifications_enabled else "")
                                    return
                                elif "request synthesis from proposers" in first and iteration < 5:
                                    # Send aggregator notes to proposers for new synthesis round
                                    aggregator_notes = text_after_first_line(agg_out)
                                    # Log resubmission prompt (packet with proposals + remarks) into Resubmissions tab
                                    s.resubmissions_history.append(("", aggregator_notes))
                                    async def synth_task(model):
                                        try:
                                            return await call_synthesis(
                                                model,
                                                last_user,
                                                s.chat_history.entries(),
                                                s.pdf_path,
                                                aggregator_notes=aggregator_notes,
                                                cost_tracker=s.cost_tracker,
                                                retries=5,
                                                temperature=s.temperature,
                                            )
                                        except Exception as e:
                                            print("[ERROR] synthesis", model, e)
                                            return create_user_friendly_error_message(e, model)

                                    proposals = await asyncio.gather(*[synth_task(m) for m in models])
                                    # ---- DEBUG: log each synthesis proposal ----
                                    for m, p in zip(models, proposals):
                                        print("[SYNTHESIS PROPOSAL]", m, "\n", p, "\n---\n")
                                        s.model_histories[m].append((last_user, p))
                                    save_session(s)
                                    continue
                                else:
                                    # Fallback treat entire aggregator output as final
                                    s.chat_history.add_assistant(agg_out)
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    save_session(s)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates(), ("done" if s.notifications_enabled else "")
                                    return
                        
                        async for (
                            chat_display,
                            status_update,
                            chatgpt_cost_up,
                            chatgpt_up,
                            claude_cost_up,
                            claude_up,
                            gemini_cost_up,
                            gemini_up,
                            resub_up,
                            notify_update,
                        ) in status_generator():
                            yield (
                                chat_display,
                                status_update,
                                chatgpt_cost_up,
                                chatgpt_up,
                                claude_cost_up,
                                claude_up,
                                gemini_cost_up,
                                gemini_up,
                                resub_up,
                                notify_update,
                            )
                
                return _handler

            evt = click_event.then(
                _make_process(btn.value),
                inputs=[state, pdf_input],
                outputs=[
                    chat,
                    status_display,
                    chatgpt_cost,
                    chatgpt_view,
                    claude_cost,
                    claude_view,
                    gemini_cost,
                    gemini_view,
                    resub_view,
                    notify_flag,
                ],
            )

            # Also run a JS scroll fix after the server event completes
            evt.then(
                None,
                inputs=None,
                outputs=None,
                js=JS_SCROLL_FIX_AFTER_EVENT,
            )

            # Trigger a browser notification when final reply is done
            evt.then(
                None,
                inputs=[notify_flag],
                outputs=None,
                js=JS_NOTIFY_IF_FLAG,
            )

        # On initial load, switch to the Attachments tab
        demo.load(
            None,
            inputs=None,
            outputs=None,
            js=JS_SELECT_TAB_ATTACHMENTS_ON_LOAD,
        )

        # Preserve per-tab scroll position across tab switches
        demo.load(
            None,
            inputs=None,
            outputs=None,
            js=JS_PRESERVE_TAB_SCROLL,
        )

        # On initial load, set up a passive click-listener to request Notification permission
        demo.load(
            None,
            inputs=None,
            outputs=None,
            js=JS_PREPARE_NOTIFICATIONS,
        )

        # Apply initial provider models from loaded configuration once UI is ready
        def _apply_initial_models(s: SessionState):
            set_openai_model(s.selected_openai_model)
            set_claude_model(s.selected_claude_model)
            set_gemini_model(s.selected_gemini_model)
            return s

        demo.load(_apply_initial_models, inputs=state, outputs=state)

    return demo


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_ui()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
