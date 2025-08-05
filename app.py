import asyncio
import os

import gradio as gr

from utils import CostTracker, save_chat, timestamp_id, BUDGET_LIMIT
from history import ChatHistory
from proposer import call_proposer, call_synthesis
from aggregator import call_aggregator, first_non_empty_line, text_after_first_line, format_proposal_packet

# Constants
BUTTONS = [
    "o3",
    "Claude",
    "Gemini",
    "o3 & Claude",
    "All",
]

MULTI_BUTTON_MODELS = {
    "o3 & Claude": ["o3", "Claude"],
    "All": ["o3", "Claude", "Gemini"],
}


class SessionState:
    def __init__(self):
        self.pdf_path: str | None = None
        self.chat_history = ChatHistory()
        self.cost_tracker = CostTracker()
        self.chat_id = timestamp_id()


async def _handle_single(model_label: str, user_input: str, state: SessionState):
    try:
        reply_text = await call_proposer(model_label, user_input, state.chat_history.entries(), state.pdf_path,
                                         state.cost_tracker, retries=1)
    except Exception as e:
        print("[ERROR] single LLM", model_label, e)
        reply_text = "(error)"
    state.chat_history.add_assistant(reply_text)
    save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
    return state.chat_history.as_display()


async def _handle_multi(models: list[str], user_input: str, state: SessionState):
    # Proposer phase (concurrent)
    async def proposer_task(model):
        try:
            return await call_proposer(model, user_input, state.chat_history.entries(), state.pdf_path,
                                       state.cost_tracker, retries=5)
        except Exception as e:
            print("[ERROR] proposer", model, e)
            return "(error)"

    proposals = await asyncio.gather(*[proposer_task(m) for m in models])

    # Aggregator iterations
    for iteration in range(1, 6):
        stream_final = iteration == 5
        agg_out = await call_aggregator(proposals, user_input, state.chat_history.entries(), state.pdf_path,
                                        state.cost_tracker, iteration, stream_final=stream_final)

        first = first_non_empty_line(agg_out).lower()
        if "final" in first:
            final_reply = text_after_first_line(agg_out)
            state.chat_history.add_assistant(final_reply)
            save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
            return final_reply
        elif "request synthesis from proposers" in first and iteration < 5:
            # Send aggregator notes to proposers for new synthesis round
            aggregator_notes = text_after_first_line(agg_out)
            async def synth_task(model):
                try:
                    return await call_synthesis(model, user_input, state.chat_history.entries(), state.pdf_path,
                                                aggregator_notes=aggregator_notes, cost_tracker=state.cost_tracker, retries=5)
                except Exception as e:
                    print("[ERROR] synthesis", model, e)
                    return "(error)"

            proposals = await asyncio.gather(*[synth_task(m) for m in models])
            continue
        else:
            # Fallback treat entire aggregator output as final
            state.chat_history.add_assistant(agg_out)
            save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
            return agg_out


async def on_send(user_input: str, button: str, state: SessionState):
    if not user_input:
        return state.chat_history.as_display()

    state.chat_history.add_user(user_input)

    # Budget guard (rough, before making calls we estimate slight cost)
    if state.cost_tracker.will_exceed_budget(0.05):
        warn = "Budget exceeded ($5). Start a new session or change selection."
        disp = state.chat_history.as_display()
        disp.append((None, warn))
        return disp

    if button in ["o3", "Claude", "Gemini"]:
        return await _handle_single(button, user_input, state)
    else:
        models = MULTI_BUTTON_MODELS[button]
        await _handle_multi(models, user_input, state)
        return state.chat_history.as_display()


def build_ui():
    with gr.Blocks() as demo:
        # gr.Markdown("## EnsembleChat")
        state = gr.State(SessionState())

        with gr.Row():
            pdf_input = gr.File(label="Select PDF", file_types=[".pdf"], type="filepath")
        chat = gr.Chatbot(height=650)
        user_box = gr.Textbox(label="You")

        with gr.Row():
            btns = [gr.Button(value=b) for b in BUTTONS]

        # Update pdf path
        def _set_pdf(file, s: SessionState):
            if file is not None:
                s.pdf_path = file
            return s

        pdf_input.change(_set_pdf, inputs=[pdf_input, state], outputs=state)

        # Handlers for each button
        def _make_handler(btn_label):
            async def handler(user_input, s: SessionState):
                return await on_send(user_input, btn_label, s)
            return handler

        for btn in btns:
            btn.click(_make_handler(btn.value), inputs=[user_box, state], outputs=chat)

    return demo


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_ui()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
