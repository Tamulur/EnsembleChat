import gradio as gr
import traceback
import asyncio

from ensemble_chat.ui.frontend_js import (
    JS_SCROLL_FIX_AFTER_EVENT,
    JS_PRESERVE_TAB_SCROLL,
    JS_NOTIFY_IF_FLAG,
    JS_PREPARE_NOTIFICATIONS,
    JS_SELECT_TAB_ATTACHMENTS_ON_LOAD,
    JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL,
)
from ensemble_chat.core.settings_actions import (
    set_openai as core_set_openai,
    set_claude as core_set_claude,
    set_gemini as core_set_gemini,
    set_aggregator as core_set_aggregator,
    set_temperature as core_set_temperature,
    set_notifications as core_set_notifications,
)
from ensemble_chat.core.session_state import SessionState
from ensemble_chat.core.interactions import prepare_user_and_state, resolve_last_user_text, budget_guard_message
from ensemble_chat.core.runtime import cancel_inflight
from ensemble_chat.core.orchestrator import run_from_label
from ensemble_chat.core.selectors import cost_line as sel_cost_line, model_display as sel_model_display, resubmissions_display as sel_resub_display, notifications_flag as sel_notify_flag
from ensemble_chat.core.session_actions import set_pdf as core_set_pdf, apply_provider_models as core_apply_models, reset_session_state as core_reset_session
from ensemble_chat.ui.ui_render import render_event, render_status_hidden, render_chat_with_message
from ensemble_chat.core.events import get_event_type
from ensemble_chat.ui.frontend_js import JS_TOGGLE_ACTIVE_BUTTON
from ensemble_chat.core.selectors import active_button_elem_id
 


def wire_events(demo: gr.Blocks, ui: dict):
    state: gr.State = ui["state"]
    new_chat_btn: gr.Button = ui["new_chat_btn"]
    chat: gr.Chatbot = ui["chat"]
    status_display: gr.Markdown = ui["status_display"]
    user_box: gr.Textbox = ui["user_box"]
    notify_flag: gr.Textbox = ui["notify_flag"]
    active_button_signal: gr.Textbox = ui["active_button_signal"]
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

    # --- Cooperative cancellation utilities (centralized in core.runtime) ---
    _cancel_inflight = cancel_inflight

    # --- PDF selection ---
    def _set_pdf(file, s: SessionState):
        print(f"[UI] pdf_input.change: file={file}")
        s, signal = core_set_pdf(file, s)
        if not file:
            print(f"[UI] No PDF selected")
        return s, signal

    pdf_input.change(_set_pdf, inputs=[pdf_input, state], outputs=[state, tab_switch_signal])
    tab_switch_signal.change(None, inputs=[tab_switch_signal], outputs=None, js=JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL)
    active_button_signal.change(None, inputs=[active_button_signal], outputs=None, js=JS_TOGGLE_ACTIVE_BUTTON)

    # --- Settings handlers ---
    def _set_openai_model(selection: str, s: SessionState):
        print(f"[UI] set_openai_model -> {selection}")
        return core_set_openai(selection, s)

    openai_model_dropdown.change(_set_openai_model, inputs=[openai_model_dropdown, state], outputs=state)

    def _set_claude_model(selection: str, s: SessionState):
        print(f"[UI] set_claude_model -> {selection}")
        return core_set_claude(selection, s)

    claude_model_dropdown.change(_set_claude_model, inputs=[claude_model_dropdown, state], outputs=state)

    def _set_gemini_model(selection: str, s: SessionState):
        print(f"[UI] set_gemini_model -> {selection}")
        return core_set_gemini(selection, s)

    gemini_model_dropdown.change(_set_gemini_model, inputs=[gemini_model_dropdown, state], outputs=state)

    def _set_aggregator(selection: str, s: SessionState):
        print(f"[UI] set_aggregator -> {selection}")
        return core_set_aggregator(selection, s)

    aggregator_dropdown.change(_set_aggregator, inputs=[aggregator_dropdown, state], outputs=state)

    def _set_temperature(val: float, s: SessionState):
        print(f"[UI] set_temperature -> {val}")
        return core_set_temperature(val, s)

    temperature_slider.change(_set_temperature, inputs=[temperature_slider, state], outputs=state)

    def _set_notifications(enabled: bool, s: SessionState):
        print(f"[UI] set_notifications -> {enabled}")
        return core_set_notifications(enabled, s)

    notifications_checkbox.change(_set_notifications, inputs=[notifications_checkbox, state], outputs=state)

    # --- New Chat (reset session) ---
    def _reset_session(s: SessionState):
        print(f"[UI] New Chat clicked: resetting session")
        # Cancel any in-flight work before resetting
        try:
            _cancel_inflight(s)
        except Exception as e:
            print(f"[UI] cancel_inflight error: {e}")
            traceback.print_exc()
        from ensemble_chat.core.session_state import SESSION_FILE
        try:
            if SESSION_FILE.exists():
                SESSION_FILE.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] Failed to delete session file: {e}")
        s = core_reset_session(s)

        return (
            s.chat_history.as_display(),
            gr.update(value="", visible=False),
            gr.update(value=sel_cost_line(s, "ChatGPT")),
            gr.update(value=sel_model_display(s, "ChatGPT")),
            gr.update(value=sel_cost_line(s, "Claude")),
            gr.update(value=sel_model_display(s, "Claude")),
            gr.update(value=sel_cost_line(s, "Gemini")),
            gr.update(value=sel_model_display(s, "Gemini")),
            gr.update(value=sel_resub_display(s)),
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
        print(f"[UI] action button clicked: input_len={len(user_input) if user_input else 0}")
        # Always cancel any in-flight work on new click
        _cancel_inflight(s)
        # Delegate history preparation to core interactions (no functional change)
        return prepare_user_and_state(user_input, s)

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
                print(f"[UI] process start label='{lbl}' current_file={bool(current_file)}")
                try:
                    if current_file:
                        s.pdf_path = current_file
                except Exception as e:
                    print(f"[WARN] Failed to set pdf_path from current_file: {e}")

                # Per-render selectors are used by the render adapter

                # Capture run id for this handler instance to detect invalidation
                if not hasattr(s, "_run_id"):
                    s._run_id = 0
                handler_run_id = s._run_id

                last_user = resolve_last_user_text(s)
                if last_user is None:
                    print(f"[UI] no last_user found; aborting run")
                    yield render_status_hidden(s)
                    return

                warn = budget_guard_message(s, 0.05)
                if warn:
                    print(f"[UI] budget guard: {warn}")
                    yield render_chat_with_message(s, warn)
                    return

                try:
                    async for event in run_from_label(lbl, last_user, s):
                            # If another click invalidated this run, stop immediately
                            if getattr(s, "_run_id", handler_run_id) != handler_run_id:
                                print(f"[UI] run invalidated in UI loop; stopping rendering")
                                return
                            ev_type = get_event_type(event)
                            # On run start lifecycle, emit active button signal (logging-only JS for now)
                            print(f"[UI] ev_type: {ev_type}")
                            if ev_type == "run_started":
                                try:
                                    elem_id = active_button_elem_id(s)
                                    print(f"[UI] active_signal(run_started) -> {elem_id}")
                                    base = render_status_hidden(s)
                                    yield (
                                        base[0], base[1], base[2], base[3], base[4], base[5], base[6], base[7], base[8], base[9],
                                        gr.update(value=elem_id),
                                    )
                                except Exception as e:
                                    print(f"[UI] active_signal(run_started) error: {e}")
                            if ev_type in ("status", "final", "error"):
                                updates = render_event(s, event)
                                # Append the active button signal as an 11th output to keep consistency
                                try:
                                    elem_id = active_button_elem_id(s) if ev_type != "final" else ""
                                    if elem_id:
                                        print(f"[UI] active_signal({ev_type}) -> {elem_id}")
                                    else:
                                        print(f"[UI] active_signal({ev_type}) -> (cleared)")
                                    yield (*updates, gr.update(value=elem_id))
                                except Exception as e:
                                    print(f"[UI] active_signal append error: {e}")
                                    yield updates
                                if ev_type == "final":
                                    return
                except asyncio.CancelledError:
                    print(f"[UI] process cancelled in _handler: label='{lbl}', handler_run_id={handler_run_id}, state_run_id={getattr(s, '_run_id', None)}")
                    return
                except Exception as e:
                    print(f"[UI] exception in _handler event loop: {e}")
                    traceback.print_exc()
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
                active_button_signal,
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
        return core_apply_models(s)

    demo.load(_apply_initial_models, inputs=state, outputs=state)


__all__ = [
    "wire_events",
]


