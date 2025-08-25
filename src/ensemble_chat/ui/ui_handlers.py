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
from ensemble_chat.core.history import ChatHistory
from ensemble_chat.core.aggregator import call_aggregator, first_non_empty_line, text_after_first_line, format_proposal_packet
from ensemble_chat.core.proposer import call_proposer, call_synthesis
from ensemble_chat.core.utils import CostTracker, save_chat, create_user_friendly_error_message
from ensemble_chat.core.model_configs import MODEL_CONFIGS
from ensemble_chat.core.settings_manager import APP_SETTINGS, save_settings
from ensemble_chat.core.session_state import SessionState, save_session, load_session_from_disk, _apply_loaded_session, _sanitize_pairs_for_display
from ensemble_chat.ui.ui_constants import MULTI_BUTTON_MODELS, LATEX_DELIMITERS
from ensemble_chat.core.conversation import handle_single


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
    )

    # --- User message handling ---
    def _add_user_and_clear(user_input: str, s: SessionState):
        if not user_input:
            s.chat_history.remove_last_assistant()
            save_session(s)
            return s.chat_history.as_display(), "", s
        s.chat_history.add_user(user_input)
        save_session(s)
        return s.chat_history.as_display(), "", s

    for btn in btns:
        click_event = btn.click(
            _add_user_and_clear,
            inputs=[user_box, state],
            outputs=[chat, user_box, state],
            show_progress=False,
        )

        def _make_process(lbl):
            async def _handler(s: SessionState, current_file):
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
                    yield s.chat_history.as_display(), gr.update(value="**Status:** Waiting for " + lbl + "…", visible=True), *_model_and_cost_updates(), ""
                    result = await handle_single(lbl, last_user, s)
                    save_session(s)
                    yield result, gr.update(value="", visible=False), *_model_and_cost_updates(), ("done" if s.notifications_enabled else "")
                else:
                    models = MULTI_BUTTON_MODELS[lbl]

                    async def status_generator():
                        # Initial proposer dispatch status
                        yield s.chat_history.as_display(), gr.update(value=f"**Status:** Sending requests for proposals…", visible=True), *_model_and_cost_updates(), ""

                        async def proposer_task(model):
                            try:
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

                        yield s.chat_history.as_display(), gr.update(value=f"**Status:** Collecting replies (0/{num_models})...", visible=True), *_model_and_cost_updates(), ""

                        for i, future in enumerate(asyncio.as_completed(tasks)):
                            model, proposal = await future
                            proposals_by_model[model] = proposal
                            status_msg = f"**Status:** Collecting replies ({i + 1}/{num_models})..."
                            yield s.chat_history.as_display(), gr.update(value=status_msg, visible=True), *_model_and_cost_updates(), ""

                        proposals = [proposals_by_model[m] for m in models]
                        for m, p in zip(models, proposals):
                            p_log = p.replace('\n', ' ')
                            if len(p_log) > 100:
                                p_log = p_log[:97] + "..."
                            print("[PROPOSAL]", m, p_log)
                            s.model_histories[m].append((last_user, p))

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
                                error_message = create_user_friendly_error_message(e, s.selected_aggregator)
                                fallback_reply = f"**Aggregation failed:** {error_message}\n\n**Here are the individual proposals:**\n\n"
                                for i, proposal in enumerate(proposals, 1):
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
                                aggregator_notes = text_after_first_line(agg_out)
                                # Build full packet for logging and for proposer synthesis
                                packet = format_proposal_packet(proposals)
                                full_prompt_for_logging = "[" + packet + "\n" + aggregator_notes + "]"
                                # Log full prompt into Resubmissions tab (user prompt containing proposals + remarks)
                                s.resubmissions_history.append(("", full_prompt_for_logging))

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
                                            proposals_packet=packet,
                                        )
                                    except Exception as e:
                                        print("[ERROR] synthesis", model, e)
                                        return create_user_friendly_error_message(e, model)

                                proposals = await asyncio.gather(*[synth_task(m) for m in models])
                                for m, p in zip(models, proposals):
                                    print("[SYNTHESIS PROPOSAL]", m, "\n", p, "\n---\n")
                                    s.model_histories[m].append((last_user, p))
                                save_session(s)
                                continue
                            else:
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


