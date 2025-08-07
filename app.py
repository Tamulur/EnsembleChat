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
        reply_text = await call_proposer(model_label, user_input, state.chat_history.entries(), state.pdf_path, state.cost_tracker, retries=1)
    except Exception as e:
        print("[ERROR] single LLM", model_label, e)
        reply_text = "(error)"
    state.chat_history.add_assistant(reply_text)
    save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
    return state.chat_history.as_display()





def build_ui():
    with gr.Blocks() as demo:
        # gr.Markdown("## EnsembleChat")
        state = gr.State(SessionState())

        with gr.Row():
            pdf_input = gr.File(label="Select PDF", file_types=[".pdf"], type="filepath")
        chat = gr.Chatbot(height=650)
        status_display = gr.Markdown("", visible=False)
        user_box = gr.Textbox(label="You", value="Does this pdf mention LLMs?")

        with gr.Row():
            btns = [gr.Button(value=b) for b in BUTTONS]

        # Update pdf path
        def _set_pdf(file, s: SessionState):
            if file is not None:
                s.pdf_path = file
            return s

        pdf_input.change(_set_pdf, inputs=[pdf_input, state], outputs=state)

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
                return s.chat_history.as_display(), "", s

            # Normal new user input
            s.chat_history.add_user(user_input)
            return s.chat_history.as_display(), "", s



        # Status callback for updating status display
        def update_status(message):
            return gr.update(value=f"**Status:** {message}", visible=True)
        
        def clear_status():
            return gr.update(value="", visible=False)

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
                async def _handler(s: SessionState):
                    # Status updates will be handled through progress tracking
                    yield s.chat_history.as_display(), gr.update(value="", visible=False)
                    
                    # Retrieve last user message (just appended by _add_user_and_clear)
                    last_user = None
                    for entry in reversed(s.chat_history.entries()):
                        if entry["role"] == "user":
                            last_user = entry["text"]
                            break
                    if last_user is None:
                        yield s.chat_history.as_display(), gr.update(value="", visible=False)
                        return

                    # Budget guard (rough, before making calls we estimate slight cost)
                    if s.cost_tracker.will_exceed_budget(0.05):
                        warn = "Budget exceeded ($5). Start a new session or change selection."
                        disp = s.chat_history.as_display()
                        disp.append((None, warn))
                        yield disp, gr.update(value="", visible=False)
                        return

                    if lbl in ["o3", "Claude", "Gemini"]:
                        result = await _handle_single(lbl, last_user, s)
                        yield result, gr.update(value="", visible=False)
                    else:
                        models = MULTI_BUTTON_MODELS[lbl]
                        
                        # Create a generator for status updates
                        async def status_generator():
                            yield s.chat_history.as_display(), gr.update(value="**Status:** Sending requests for proposals…", visible=True)
                            
                            # Proposer phase (concurrent)
                            async def proposer_task(model):
                                try:
                                    return await call_proposer(model, last_user, s.chat_history.entries(), s.pdf_path,
                                                               s.cost_tracker, retries=5)
                                except Exception as e:
                                    print("[ERROR] proposer", model, e)
                                    return "(error)"

                            yield s.chat_history.as_display(), gr.update(value="**Status:** Collecting replies…", visible=True)
                            proposals = await asyncio.gather(*[proposer_task(m) for m in models])

                            # ---- DEBUG: log each proposer reply ----
                            for m, p in zip(models, proposals):
                                print("[PROPOSAL]", m, "\n", p, "\n---\n")

                            # Aggregator iterations
                            for iteration in range(1, 6):
                                yield s.chat_history.as_display(), gr.update(value=f"**Status:** Aggregating replies, iteration {iteration}…", visible=True)
                                
                                agg_out = await call_aggregator(proposals, last_user, s.chat_history.entries(), s.pdf_path,
                                                                s.cost_tracker, iteration)

                                first = first_non_empty_line(agg_out).lower()
                                if "final" in first:
                                    final_reply = text_after_first_line(agg_out)
                                    s.chat_history.add_assistant(final_reply)
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False)
                                    return
                                elif "request synthesis from proposers" in first and iteration < 5:
                                    # Send aggregator notes to proposers for new synthesis round
                                    aggregator_notes = text_after_first_line(agg_out)
                                    async def synth_task(model):
                                        try:
                                            return await call_synthesis(model, last_user, s.chat_history.entries(), s.pdf_path,
                                                                        aggregator_notes=aggregator_notes, cost_tracker=s.cost_tracker, retries=5)
                                        except Exception as e:
                                            print("[ERROR] synthesis", model, e)
                                            return "(error)"

                                    proposals = await asyncio.gather(*[synth_task(m) for m in models])
                                    # ---- DEBUG: log each synthesis proposal ----
                                    for m, p in zip(models, proposals):
                                        print("[SYNTHESIS PROPOSAL]", m, "\n", p, "\n---\n")
                                    continue
                                else:
                                    # Fallback treat entire aggregator output as final
                                    s.chat_history.add_assistant(agg_out)
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False)
                                    return
                        
                        async for chat_display, status_update in status_generator():
                            yield chat_display, status_update
                
                return _handler

            click_event.then(
                _make_process(btn.value),
                inputs=state,
                outputs=[chat, status_display],
            )

    return demo


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_ui()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
