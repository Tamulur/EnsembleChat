import asyncio
import os
import gradio as gr

from ensemble_chat.llm_providers import set_openai_model, set_claude_model, set_gemini_model
from ensemble_chat.ui.frontend_js import (
    JS_ALIGN_ON_CHANGE,
    JS_SCROLL_FIX_AFTER_EVENT,
    JS_PRESERVE_TAB_SCROLL,
    JS_NOTIFY_IF_FLAG,
    JS_PREPARE_NOTIFICATIONS,
    JS_SELECT_TAB_ATTACHMENTS_ON_LOAD,
    JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL,
)
from ensemble_chat.core.engine import run_multi, run_single
from ensemble_chat.core.model_configs import MODEL_CONFIGS
from ensemble_chat.core.settings_manager import APP_SETTINGS, save_settings
from ensemble_chat.core.session_state import SessionState, save_session, load_session_from_disk, _apply_loaded_session, _sanitize_pairs_for_display
from ensemble_chat.ui.ui_constants import MULTI_BUTTON_MODELS, LATEX_DELIMITERS
 


def wire_events(demo: gr.Blocks, ui: dict):
    state: gr.State = ui["state"]
    new_chat_btn: gr.Button = ui["new_chat_btn"]
    chat: gr.Chatbot = ui["chat"]
    status_display: gr.Markdown = ui["status_display"]
    user_box: gr.Textbox = ui["user_box"]
    notify_flag: gr.Textbox = ui["notify_flag"]
    btns: list[gr.Button] = ui["buttons"]
    chatgpt_cost: gr.Markdown = ui["chatgpt_cost"]
    chatgpt_view: gr.Chatbot = ui["chatgpt_view"]
    claude_cost: gr.Markdown = ui["claude_cost"]
    claude_view: gr.Chatbot = ui["claude_view"]
    gemini_cost: gr.Markdown = ui["gemini_cost"]
    gemini_view: gr.Chatbot = ui["gemini_view"]
    pdf_input: gr.File = ui["pdf_input"]
    tab_switch_signal: gr.Textbox = ui["tab_switch_signal"]
    resub_view: gr.Chatbot = ui["resub_view"]
    openai_model_dropdown: gr.Dropdown = ui["openai_model_dropdown"]
    claude_model_dropdown: gr.Dropdown = ui["claude_model_dropdown"]
    gemini_model_dropdown: gr.Dropdown = ui["gemini_model_dropdown"]
    aggregator_dropdown: gr.Dropdown = ui["aggregator_dropdown"]
    temperature_slider: gr.Slider = ui["temperature_slider"]
    notifications_checkbox: gr.Checkbox = ui["notifications_checkbox"]

    # --- Cooperative cancellation utilities ---
    def _cancel_inflight(s: SessionState):
        """Cancel any in-flight async tasks for this session and bump run id."""
        try:
            # Bump run id to invalidate any loops checking it
            if not hasattr(s, "_run_id"):
                s._run_id = 0
            s._run_id += 1
            current = getattr(s, "_current_task", None)
            if current is None:
                return
            # Support single task or list/tuple of tasks
            tasks = current if isinstance(current, (list, tuple)) else [current]
            for t in tasks:
                try:
                    if t is None:
                        continue
                    # Asyncio Task cancellation (ignore errors if already done)
                    if hasattr(t, "cancel"):
                        t.cancel()
                except Exception:
                    pass
            s._current_task = None
        except Exception:
            # Never let cancellation path throw
            pass

    # --- PDF selection ---
    def _set_pdf(file, s: SessionState):
        if file is not None:
            s.pdf_path = file
            print(f"[DEBUG] PDF selected: {file}, sending switch_tab signal")
            save_session(s)
            return s, "switch_tab"
        print(f"[DEBUG] No PDF selected, file is: {file}")
        return s, ""

    pdf_input.change(_set_pdf, inputs=[pdf_input, state], outputs=[state, tab_switch_signal])
    tab_switch_signal.change(None, inputs=[tab_switch_signal], outputs=None, js=JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL)

    # --- Settings handlers ---
    def _set_openai_model(selection: str, s: SessionState):
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
        except Exception as e:
            print(f"[WARN] Invalid temperature '{val}': {e}. Using default 0.7")
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
        # Cancel any in-flight work before resetting
        try:
            _cancel_inflight(s)
        except Exception:
            pass
        from ensemble_chat.core.session_state import SESSION_FILE
        try:
            if SESSION_FILE.exists():
                SESSION_FILE.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to delete session file: {e}")
        s = SessionState()
        # Rehydrate selections from persisted app settings
        try:
            s.selected_openai_model = APP_SETTINGS.get("openai_model", MODEL_CONFIGS["OpenAI"][0])
            s.selected_claude_model = APP_SETTINGS.get("claude_model", MODEL_CONFIGS["Claude"][0])
            s.selected_gemini_model = APP_SETTINGS.get("gemini_model", MODEL_CONFIGS["Gemini"][0])
            s.selected_aggregator = APP_SETTINGS.get("aggregator", "Claude")
            s.temperature = float(APP_SETTINGS.get("temperature", 0.7))
            s.notifications_enabled = bool(APP_SETTINGS.get("notifications", True))
            # Apply provider selections to backend
            set_openai_model(s.selected_openai_model)
            set_claude_model(s.selected_claude_model)
            set_gemini_model(s.selected_gemini_model)
        except Exception as e:
            print(f"[WARN] Failed to rehydrate selections or apply provider models: {e}")
        save_session(s)

        def _cost_line(label: str) -> str:
            return f"**Cost so far:** ${s.cost_tracker.get_model_cost(label):.4f}"

        return (
            s.chat_history.as_display(),
            gr.update(value="", visible=False),
            gr.update(value=_cost_line("ChatGPT")),
            gr.update(value=[]),
            gr.update(value=_cost_line("Claude")),
            gr.update(value=[]),
            gr.update(value=_cost_line("Gemini")),
            gr.update(value=[]),
            gr.update(value=[]),
            "",
            s,
            gr.update(value=None),
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
        queue=False,
    )

    # --- User message handling ---
    def _add_user_and_clear(user_input: str, s: SessionState):
        # Always cancel any in-flight work on new click
        _cancel_inflight(s)

        if not user_input:
            # Regenerate: keep last user message, ensure no dangling assistant
            s.chat_history.remove_last_assistant()
            save_session(s)
            return s.chat_history.as_display(), "", s

        # New query: remove any unanswered trailing user from an aborted run
        try:
            s.chat_history.remove_last_user()
        except Exception:
            pass
        s.chat_history.add_user(user_input)
        save_session(s)
        return s.chat_history.as_display(), "", s

    for btn in btns:
        click_event = btn.click(
            _add_user_and_clear,
            inputs=[user_box, state],
            outputs=[chat, user_box, state],
            show_progress=False,
            queue=False,
        )

        def _make_process(lbl):
            async def _handler(s: SessionState, current_file):
                # Track this event's asyncio task for cooperative cancellation
                try:
                    import asyncio as _asyncio
                    s._current_task = _asyncio.current_task()
                except Exception:
                    pass
                try:
                    if current_file:
                        s.pdf_path = current_file
                except Exception as e:
                    print(f"[WARN] Failed to set pdf_path from current_file: {e}")

                def _cost_line(label: str) -> str:
                    return f"**Cost so far:** ${s.cost_tracker.get_model_cost(label):.4f}"

                def _model_and_cost_updates():
                    return (
                        gr.update(value=_cost_line("ChatGPT")),
                        gr.update(value=_sanitize_pairs_for_display(s.model_histories["ChatGPT"])),
                        gr.update(value=_cost_line("Claude")),
                        gr.update(value=_sanitize_pairs_for_display(s.model_histories["Claude"])),
                        gr.update(value=_cost_line("Gemini")),
                        gr.update(value=_sanitize_pairs_for_display(s.model_histories["Gemini"])),
                        gr.update(value=_sanitize_pairs_for_display(s.resubmissions_history)),
                    )

                # Capture run id for this handler instance to detect invalidation
                if not hasattr(s, "_run_id"):
                    s._run_id = 0
                handler_run_id = s._run_id

                last_user = None
                for entry in reversed(s.chat_history.entries()):
                    if entry["role"] == "user":
                        last_user = entry["text"]
                        break
                if last_user is None:
                    yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates(), ""
                    return

                if s.cost_tracker.will_exceed_budget(0.05):
                    warn = "Budget exceeded ($5). Start a new session or change selection."
                    disp = s.chat_history.as_display()
                    disp.append((None, warn))
                    yield disp, gr.update(value="", visible=False), *_model_and_cost_updates(), ""
                    return

                if lbl in ["ChatGPT", "Claude", "Gemini"]:
                    async for event in run_single(lbl, last_user, s):
                        # If another click invalidated this run, stop immediately
                        if getattr(s, "_run_id", handler_run_id) != handler_run_id:
                            return
                        save_session(s)
                        if event.get("type") == "status":
                            yield (
                                s.chat_history.as_display(),
                                gr.update(value=event.get("text", ""), visible=True),
                                gr.update(value=_cost_line("ChatGPT")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["ChatGPT"])),
                                gr.update(value=_cost_line("Claude")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Claude"])),
                                gr.update(value=_cost_line("Gemini")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Gemini"])),
                                gr.update(value=_sanitize_pairs_for_display(s.resubmissions_history)),
                                "",
                            )
                        elif event.get("type") == "final":
                            yield (
                                s.chat_history.as_display(),
                                gr.update(value="", visible=False),
                                gr.update(value=_cost_line("ChatGPT")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["ChatGPT"])),
                                gr.update(value=_cost_line("Claude")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Claude"])),
                                gr.update(value=_cost_line("Gemini")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Gemini"])),
                                gr.update(value=_sanitize_pairs_for_display(s.resubmissions_history)),
                                ("done" if s.notifications_enabled else ""),
                            )
                            return
                else:
                    models = MULTI_BUTTON_MODELS[lbl]

                    async for event in run_multi(models, s.selected_aggregator, last_user, s):
                        # If another click invalidated this run, stop immediately
                        if getattr(s, "_run_id", handler_run_id) != handler_run_id:
                            return
                        # Persist state changes between events
                        save_session(s)

                        if event.get("type") == "status":
                            yield (
                                s.chat_history.as_display(),
                                gr.update(value=event.get("text", ""), visible=True),
                                gr.update(value=_cost_line("ChatGPT")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["ChatGPT"])),
                                gr.update(value=_cost_line("Claude")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Claude"])),
                                gr.update(value=_cost_line("Gemini")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Gemini"])),
                                gr.update(value=_sanitize_pairs_for_display(s.resubmissions_history)),
                                "",
                            )
                        elif event.get("type") == "final":
                            yield (
                                s.chat_history.as_display(),
                                gr.update(value="", visible=False),
                                gr.update(value=_cost_line("ChatGPT")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["ChatGPT"])),
                                gr.update(value=_cost_line("Claude")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Claude"])),
                                gr.update(value=_cost_line("Gemini")),
                                gr.update(value=_sanitize_pairs_for_display(s.model_histories["Gemini"])),
                                gr.update(value=_sanitize_pairs_for_display(s.resubmissions_history)),
                                ("done" if s.notifications_enabled else ""),
                            )
                            return

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

        evt.then(None, inputs=None, outputs=None, js=JS_SCROLL_FIX_AFTER_EVENT)
        evt.then(None, inputs=[notify_flag], outputs=None, js=JS_NOTIFY_IF_FLAG)

    # JS on load hooks
    demo.load(None, inputs=None, outputs=None, js=JS_SELECT_TAB_ATTACHMENTS_ON_LOAD)
    demo.load(None, inputs=None, outputs=None, js=JS_PRESERVE_TAB_SCROLL)
    demo.load(None, inputs=None, outputs=None, js=JS_PREPARE_NOTIFICATIONS)

    # Apply initial provider models on app load
    def _apply_initial_models(s: SessionState):
        try:
            set_openai_model(str(s.selected_openai_model).lower())
        except Exception as e:
            print(f"[WARN] Failed to set initial OpenAI model to lowercase id: {e}. Falling back to raw selection.")
            set_openai_model(str(s.selected_openai_model))
        set_claude_model(s.selected_claude_model)
        set_gemini_model(s.selected_gemini_model)
        return s

    demo.load(_apply_initial_models, inputs=state, outputs=state)


__all__ = [
    "wire_events",
]


