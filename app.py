import asyncio
import os

import gradio as gr
import warnings
from pathlib import Path
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning, message="You have not specified a value for the `type` parameter.*")

from utils import CostTracker, save_chat, timestamp_id, BUDGET_LIMIT
from frontend_js import (
    JS_ALIGN_ON_CHANGE,
    JS_SCROLL_FIX_AFTER_EVENT,
    JS_PRESERVE_TAB_SCROLL,
)
from history import ChatHistory
from proposer import call_proposer, call_synthesis
from aggregator import call_aggregator, first_non_empty_line, text_after_first_line, format_proposal_packet

# Constants
ICON_DIR = Path(__file__).parent / "Icons"

BUTTONS = [
    "o3",
    "Claude",
    "Gemini",
    "o3 & Claude",
    "All",
]

# Mapping of button label to icon path
ICON_MAP = {
    "o3": str(ICON_DIR / "OpenAI.png"),
    "Claude": str(ICON_DIR / "Claude.png"),
    "Gemini": str(ICON_DIR / "Gemini.png"),
    "o3 & Claude": str(ICON_DIR / "o3_claude.png"),
    "All": str(ICON_DIR / "all.png"),
}


MULTI_BUTTON_MODELS = {
    "o3 & Claude": ["o3", "Claude"],
    "All": ["o3", "Claude", "Gemini"],
}

# Shared LaTeX delimiters configuration for all Chatbot components
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "[", "right": "]", "display": True},  # Support OpenAI's format
]


class SessionState:
    def __init__(self):
        self.pdf_path: str | None = None
        self.chat_history = ChatHistory()
        self.cost_tracker = CostTracker()
        self.chat_id = timestamp_id()
        # Per-model chat history for tabs
        self.model_histories = {"o3": [], "Claude": [], "Gemini": []}


async def _handle_single(model_label: str, user_input: str, state: SessionState):
    try:
        reply_text = await call_proposer(model_label, user_input, state.chat_history.entries(), state.pdf_path, state.cost_tracker, retries=1)
    except Exception as e:
        print("[ERROR] single LLM", model_label, e)
        reply_text = "(error)"
    # Update chat history and per-model tab history
    state.chat_history.add_assistant(reply_text)
    state.model_histories[model_label].append(("", reply_text))
    save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
    return state.chat_history.as_display()


def build_ui():
    from frontend_css import CSS_GLOBAL
    with gr.Blocks(css=CSS_GLOBAL) as demo:
        # gr.Markdown("## EnsembleChat")
        state = gr.State(SessionState())

        # ---------- TABS LAYOUT ----------
        with gr.Tabs() as tabs:
            # ---- Main Chat tab ----
            with gr.Tab("Chat"):
                with gr.Row():
                    pdf_input = gr.File(label="Select PDF", file_types=[".pdf"], type="filepath")

                chat = gr.Chatbot(
                    height=600,
                    elem_id="chat_interface",
                    autoscroll=False,
                    show_label=False,
                    latex_delimiters=LATEX_DELIMITERS
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

                with gr.Row():
                    btns = [gr.Button(value=b, icon=ICON_MAP.get(b)) for b in BUTTONS]

            # ---- Per-model tabs ----
            with gr.Tab("o3"):
                o3_cost = gr.Markdown("**Cost so far:** $0.0000", elem_id="o3_cost")
                o3_view = gr.Chatbot(label="o3 Output", height=800, value=[], autoscroll=False, elem_id="o3_view",
                                      latex_delimiters=LATEX_DELIMITERS)

            with gr.Tab("Claude"):
                claude_cost = gr.Markdown("**Cost so far:** $0.0000", elem_id="claude_cost")
                claude_view = gr.Chatbot(label="Claude Output", height=800, value=[], autoscroll=False, elem_id="claude_view",
                                        latex_delimiters=LATEX_DELIMITERS)

            with gr.Tab("Gemini"):
                gemini_cost = gr.Markdown("**Cost so far:** $0.0000", elem_id="gemini_cost")
                gemini_view = gr.Chatbot(label="Gemini Output", height=800, value=[], autoscroll=False, elem_id="gemini_view",
                                       latex_delimiters=LATEX_DELIMITERS)

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
                    # Helpers to build update objects for each model tab
                    def _cost_line(label: str) -> str:
                        return f"**Cost so far:** ${s.cost_tracker.get_model_cost(label):.4f}"

                    def _model_and_cost_updates():
                        return (
                            gr.update(value=_cost_line("o3")),
                            gr.update(value=s.model_histories["o3"]),
                            gr.update(value=_cost_line("Claude")),
                            gr.update(value=s.model_histories["Claude"]),
                            gr.update(value=_cost_line("Gemini")),
                            gr.update(value=s.model_histories["Gemini"]),
                        )
                    # Retrieve last user message (just appended by _add_user_and_clear)
                    last_user = None
                    for entry in reversed(s.chat_history.entries()):
                        if entry["role"] == "user":
                            last_user = entry["text"]
                            break
                    if last_user is None:
                        yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates()
                        return

                    # Budget guard (rough, before making calls we estimate slight cost)
                    if s.cost_tracker.will_exceed_budget(0.05):
                        warn = "Budget exceeded ($5). Start a new session or change selection."
                        disp = s.chat_history.as_display()
                        disp.append((None, warn))
                        yield disp, gr.update(value="", visible=False), *_model_and_cost_updates()
                        return

                    if lbl in ["o3", "Claude", "Gemini"]:
                        yield s.chat_history.as_display(), gr.update(value="**Status:** Waiting for " + lbl + "…", visible=True), *_model_and_cost_updates()
                        result = await _handle_single(lbl, last_user, s)
                        yield result, gr.update(value="", visible=False), *_model_and_cost_updates()
                    else:
                        models = MULTI_BUTTON_MODELS[lbl]
                        
                        # Create a generator for status updates
                        async def status_generator():
                            # Proposer phase (concurrent)
                            async def proposer_task(model):
                                try:
                                    # Return model name with result for easier tracking
                                    result = await call_proposer(model, last_user, s.chat_history.entries(), s.pdf_path,
                                                               s.cost_tracker, retries=5)
                                    return model, result
                                except Exception as e:
                                    print("[ERROR] proposer", model, e)
                                    return model, "(error)"

                            tasks = [proposer_task(m) for m in models]
                            proposals_by_model = {}
                            num_models = len(models)

                            yield s.chat_history.as_display(), gr.update(
                                value=f"**Status:** Collecting replies (0/{num_models})...", visible=True), *_model_and_cost_updates()

                            for i, future in enumerate(asyncio.as_completed(tasks)):
                                model, proposal = await future
                                proposals_by_model[model] = proposal

                                # Update status message with progress
                                status_msg = f"**Status:** Collecting replies ({i + 1}/{num_models})..."
                                yield s.chat_history.as_display(), gr.update(value=status_msg,
                                                                              visible=True), *_model_and_cost_updates()

                            # Re-order proposals to match original model list
                            proposals = [proposals_by_model[m] for m in models]
                            # ---- DEBUG: log each proposer reply ----
                            for m, p in zip(models, proposals):
                                # Truncate and replace newlines for logging
                                p_log = p.replace('\n', ' ')
                                if len(p_log) > 100:
                                    p_log = p_log[:97] + "..."
                                print("[PROPOSAL]", m, p_log)
                                s.model_histories[m].append(("", p))

                            # Aggregator iterations
                            for iteration in range(1, 6):
                                yield s.chat_history.as_display(), gr.update(value=f"**Status:** Aggregating replies, iteration {iteration}…", visible=True), *_model_and_cost_updates()
                                
                                agg_out = await call_aggregator(proposals, last_user, s.chat_history.entries(), s.pdf_path,
                                                                s.cost_tracker, iteration)

                                first = first_non_empty_line(agg_out).lower()
                                if "final" in first:
                                    final_reply = text_after_first_line(agg_out)
                                    s.chat_history.add_assistant(final_reply)
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates()
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
                                        s.model_histories[m].append(("", p))
                                    continue
                                else:
                                    # Fallback treat entire aggregator output as final
                                    s.chat_history.add_assistant(agg_out)
                                    save_chat(s.chat_id, s.chat_history.entries(), s.pdf_path)
                                    yield s.chat_history.as_display(), gr.update(value="", visible=False), *_model_and_cost_updates()
                                    return
                        
                        async for (
                            chat_display,
                            status_update,
                            o3_cost_up,
                            o3_up,
                            claude_cost_up,
                            claude_up,
                            gemini_cost_up,
                            gemini_up,
                        ) in status_generator():
                            yield (
                                chat_display,
                                status_update,
                                o3_cost_up,
                                o3_up,
                                claude_cost_up,
                                claude_up,
                                gemini_cost_up,
                                gemini_up,
                            )
                
                return _handler

            evt = click_event.then(
                _make_process(btn.value),
                inputs=state,
                outputs=[
                    chat,
                    status_display,
                    o3_cost,
                    o3_view,
                    claude_cost,
                    claude_view,
                    gemini_cost,
                    gemini_view,
                ],
            )

            # Also run a JS scroll fix after the server event completes
            evt.then(
                None,
                inputs=None,
                outputs=None,
                js=JS_SCROLL_FIX_AFTER_EVENT,
            )

        # Preserve per-tab scroll position across tab switches
        demo.load(
            None,
            inputs=None,
            outputs=None,
            js=JS_PRESERVE_TAB_SCROLL,
        )

    return demo


def main():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_ui()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
