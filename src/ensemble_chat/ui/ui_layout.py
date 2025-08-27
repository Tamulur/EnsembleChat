import os
import gradio as gr

from ensemble_chat.ui.frontend_css import CSS_GLOBAL
from ensemble_chat.ui.frontend_js import JS_ALIGN_ON_CHANGE
from ensemble_chat.core.session_state import SessionState
from ensemble_chat.core.selectors import cost_line as sel_cost_line, model_display as sel_model_display, resubmissions_display as sel_resub_display
from ensemble_chat.ui.ui_constants import ICON_MAP, BUTTONS, LATEX_DELIMITERS, STOP_ICON


def build_base_layout(initial_state: SessionState, app_settings, model_configs, wire=None):
    with gr.Blocks(css=CSS_GLOBAL) as demo:
        state = gr.State(initial_state)
        new_chat_btn = gr.Button(value="New Chat", elem_id="btn_new_chat")

        with gr.Tabs(selected=4) as tabs:
            with gr.Tab("Chat"):
                chat = gr.Chatbot(
                    height=500,
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
                active_button_signal = gr.Textbox(value="", visible=False)
                # Hidden button to ensure Gradio serves the stop icon (visible but will be hidden with CSS)
                stop_icon_ref = gr.Button(value="Hidden", icon=STOP_ICON, visible=True, elem_id="hidden_stop_icon_btn")

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

            MODEL_TAB_HEIGHT = 600
            with gr.Tab("ChatGPT"):
                chatgpt_cost = gr.Markdown(
                    sel_cost_line(initial_state, "ChatGPT"),
                    elem_id="chatgpt_cost",
                )
                chatgpt_view = gr.Chatbot(
                    label="ChatGPT Output",
                    height=MODEL_TAB_HEIGHT,
                    value=sel_model_display(initial_state, "ChatGPT"),
                    autoscroll=False,
                    elem_id="chatgpt_view",
                    latex_delimiters=LATEX_DELIMITERS,
                )

            with gr.Tab("Claude"):
                claude_cost = gr.Markdown(
                    sel_cost_line(initial_state, "Claude"),
                    elem_id="claude_cost",
                )
                claude_view = gr.Chatbot(
                    label="Claude Output",
                    height=MODEL_TAB_HEIGHT,
                    value=sel_model_display(initial_state, "Claude"),
                    autoscroll=False,
                    elem_id="claude_view",
                    latex_delimiters=LATEX_DELIMITERS,
                )

            with gr.Tab("Gemini"):
                gemini_cost = gr.Markdown(
                    sel_cost_line(initial_state, "Gemini"),
                    elem_id="gemini_cost",
                )
                gemini_view = gr.Chatbot(
                    label="Gemini Output",
                    height=MODEL_TAB_HEIGHT,
                    value=sel_model_display(initial_state, "Gemini"),
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
                resub_view = gr.Chatbot(
                    label="Resubmissions",
                    height=800,
                    value=sel_resub_display(initial_state),
                    autoscroll=False,
                    elem_id="resub_view",
                    latex_delimiters=LATEX_DELIMITERS,
                )

            with gr.Tab("Settings"):
                with gr.Row():
                    openai_model_dropdown = gr.Dropdown(
                        choices=model_configs["OpenAI"],
                        value=(app_settings.get("openai_model") if app_settings.get("openai_model") in model_configs["OpenAI"] else model_configs["OpenAI"][0]),
                        label="OpenAI model",
                        interactive=True,
                    )
                with gr.Row():
                    claude_model_dropdown = gr.Dropdown(
                        choices=model_configs["Claude"],
                        value=(app_settings.get("claude_model") if app_settings.get("claude_model") in model_configs["Claude"] else model_configs["Claude"][0]),
                        label="Claude model",
                        interactive=True,
                    )
                with gr.Row():
                    gemini_model_dropdown = gr.Dropdown(
                        choices=model_configs["Gemini"],
                        value=(app_settings.get("gemini_model") if app_settings.get("gemini_model") in model_configs["Gemini"] else model_configs["Gemini"][0]),
                        label="Gemini model",
                        interactive=True,
                    )
                with gr.Row():
                    aggregator_dropdown = gr.Dropdown(
                        choices=["ChatGPT", "Claude", "Gemini"],
                        value=(app_settings.get("aggregator") if app_settings.get("aggregator") in ["ChatGPT", "Claude", "Gemini"] else "Claude"),
                        label="Aggregator",
                        interactive=True,
                    )
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=float(app_settings.get("temperature", 0.7)),
                        label="Temperature",
                        interactive=True,
                    )
                with gr.Row():
                    notifications_checkbox = gr.Checkbox(
                        value=bool(app_settings.get("notifications", True)),
                        label="Notifications",
                        interactive=True,
                    )

        ui = {
            "state": state,
            "new_chat_btn": new_chat_btn,
            "chat": chat,
            "status_display": status_display,
            "user_box": user_box,
            "notify_flag": notify_flag,
            "active_button_signal": active_button_signal,
            "buttons": btns,
            "chatgpt_cost": chatgpt_cost,
            "chatgpt_view": chatgpt_view,
            "claude_cost": claude_cost,
            "claude_view": claude_view,
            "gemini_cost": gemini_cost,
            "gemini_view": gemini_view,
            "pdf_input": pdf_input,
            "tab_switch_signal": tab_switch_signal,
            "resub_view": resub_view,
            "openai_model_dropdown": openai_model_dropdown,
            "claude_model_dropdown": claude_model_dropdown,
            "gemini_model_dropdown": gemini_model_dropdown,
            "aggregator_dropdown": aggregator_dropdown,
            "temperature_slider": temperature_slider,
            "notifications_checkbox": notifications_checkbox,
        }
        # Wire events within the Blocks context
        if callable(wire):
            wire(demo, ui)

    return demo, ui


__all__ = [
    "build_base_layout",
]


