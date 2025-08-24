import os
import asyncio
import gradio as gr

from llm_providers import set_openai_model, set_claude_model, set_gemini_model
from frontend_js import (
    JS_ALIGN_ON_CHANGE,
    JS_SCROLL_FIX_AFTER_EVENT,
    JS_PRESERVE_TAB_SCROLL,
    JS_NOTIFY_IF_FLAG,
    JS_PREPARE_NOTIFICATIONS,
    JS_SELECT_TAB_ATTACHMENTS_ON_LOAD,
    JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL,
)
from frontend_css import CSS_GLOBAL
from history import ChatHistory
from aggregator import call_aggregator, first_non_empty_line, text_after_first_line, format_proposal_packet
from proposer import call_proposer, call_synthesis
from utils import CostTracker, save_chat, create_user_friendly_error_message
from model_configs import MODEL_CONFIGS
from settings_manager import APP_SETTINGS, save_settings
from session_state import SessionState, save_session, load_session_from_disk, _apply_loaded_session, _sanitize_pairs_for_display
from ui_constants import ICON_MAP, BUTTONS, LATEX_DELIMITERS, MULTI_BUTTON_MODELS
from conversation import handle_single


 


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

    with gr.Blocks(css=CSS_GLOBAL) as demo:
        state = gr.State(initial_state)
        new_chat_btn = gr.Button(value="New Chat", elem_id="btn_new_chat")

        with gr.Tabs(selected=4) as tabs:
            with gr.Tab("Chat"):
                chat = gr.Chatbot(
                    height=630,
                    elem_id="chat_interface",
                    autoscroll=False,
                    show_label=False,
                    latex_delimiters=LATEX_DELIMITERS,
                    value=initial_state.chat_history.as_display(),
                )
                chat.change(None, inputs=[], outputs=[], js=JS_ALIGN_ON_CHANGE)
                status_display = gr.Markdown("", visible=False)
                user_box = gr.Textbox(label="You", value="")
                notify_flag = gr.Textbox(value="", visible=False)

                with gr.Row():
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

            MODEL_TAB_HEIGHT = 700
            with gr.Tab("ChatGPT"):
                chatgpt_cost = gr.Markdown(f"**Cost so far:** ${initial_state.cost_tracker.get_model_cost('ChatGPT'):.4f}", elem_id="chatgpt_cost")
                chatgpt_view = gr.Chatbot(
                    label="ChatGPT Output",
                    height=MODEL_TAB_HEIGHT,
                    value=_sanitize_pairs_for_display(initial_state.model_histories["ChatGPT"]),
                    autoscroll=False,
                    elem_id="chatgpt_view",
                    latex_delimiters=LATEX_DELIMITERS,
                )

            with gr.Tab("Claude"):
                claude_cost = gr.Markdown(f"**Cost so far:** ${initial_state.cost_tracker.get_model_cost('Claude'):.4f}", elem_id="claude_cost")
                claude_view = gr.Chatbot(
                    label="Claude Output",
                    height=MODEL_TAB_HEIGHT,
                    value=_sanitize_pairs_for_display(initial_state.model_histories["Claude"]),
                    autoscroll=False,
                    elem_id="claude_view",
                    latex_delimiters=LATEX_DELIMITERS,
                )

            with gr.Tab("Gemini"):
                gemini_cost = gr.Markdown(f"**Cost so far:** ${initial_state.cost_tracker.get_model_cost('Gemini'):.4f}", elem_id="gemini_cost")
                gemini_view = gr.Chatbot(
                    label="Gemini Output",
                    height=MODEL_TAB_HEIGHT,
                    value=_sanitize_pairs_for_display(initial_state.model_histories["Gemini"]),
                    autoscroll=False,
                    elem_id="gemini_view",
                    latex_delimiters=LATEX_DELIMITERS,
                )

            with gr.Tab("Attachments"):
                with gr.Row():
                    initial_pdf_value = initial_state.pdf_path if (initial_state.pdf_path and os.path.isfile(initial_state.pdf_path)) else None
                    pdf_input = gr.File(label="Select PDF", file_types=[".pdf"], type="filepath", value=initial_pdf_value)
                tab_switch_signal = gr.Textbox(value="", visible=False)

            with gr.Tab("Resubmissions"):
                resub_view = gr.Chatbot(label="Resubmissions", height=800, value=_sanitize_pairs_for_display(initial_state.resubmissions_history), autoscroll=False, elem_id="resub_view",
                                        latex_delimiters=LATEX_DELIMITERS)

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
                return s, "switch_tab"
            print(f"[DEBUG] No PDF selected, file is: {file}")
            return s, ""

        pdf_change_evt = pdf_input.change(_set_pdf, inputs=[pdf_input, state], outputs=[state, tab_switch_signal])
        tab_switch_signal.change(None, inputs=[tab_switch_signal], outputs=None, js=JS_SWITCH_TO_CHAT_TAB_IF_SIGNAL)

        # --- Settings handlers ---
        def _set_openai_model(selection: str, s: SessionState):
            s.selected_openai_model = selection
            try:
                set_openai_model(str(selection).lower())
            except Exception:
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
            from session_state import SESSION_FILE
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
            except Exception:
                pass
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
                    except Exception:
                        pass

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

        demo.load(None, inputs=None, outputs=None, js=JS_SELECT_TAB_ATTACHMENTS_ON_LOAD)
        demo.load(None, inputs=None, outputs=None, js=JS_PRESERVE_TAB_SCROLL)
        demo.load(None, inputs=None, outputs=None, js=JS_PREPARE_NOTIFICATIONS)

        def _apply_initial_models(s: SessionState):
            try:
                set_openai_model(str(s.selected_openai_model).lower())
            except Exception:
                set_openai_model(str(s.selected_openai_model))
            set_claude_model(s.selected_claude_model)
            set_gemini_model(s.selected_gemini_model)
            return s

        demo.load(_apply_initial_models, inputs=state, outputs=state)

    return demo


__all__ = [
    "build_ui",
]


