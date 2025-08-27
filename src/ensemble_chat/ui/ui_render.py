import gradio as gr
import traceback

from ensemble_chat.core.selectors import (
    cost_line as sel_cost_line,
    model_display as sel_model_display,
    resubmissions_display as sel_resub_display,
    notifications_flag as sel_notify_flag,
)
from ensemble_chat.core.events import StatusEvent, FinalEvent, get_event_type


def render_event(state, event):
    """Build the tuple of Gradio updates for a given logic event.

    Order must match ui_handlers outputs:
    [chat, status_display, chatgpt_cost, chatgpt_view, claude_cost, claude_view, gemini_cost, gemini_view, resub_view, notify_flag]
    """
    ev_type = get_event_type(event)
    text = getattr(event, "text", None)
    if text is None:
        try:
            text = (event or {}).get("text", "")
        except Exception:
            text = ""

    if ev_type == "status":
        print(f"[UI][render] status -> visible=True, text='{text[:80]}'")
        return (
            state.chat_history.as_display(),
            gr.update(value=text, visible=True),
            gr.update(value=sel_cost_line(state, "ChatGPT")),
            gr.update(value=sel_model_display(state, "ChatGPT")),
            gr.update(value=sel_cost_line(state, "Claude")),
            gr.update(value=sel_model_display(state, "Claude")),
            gr.update(value=sel_cost_line(state, "Gemini")),
            gr.update(value=sel_model_display(state, "Gemini")),
            gr.update(value=sel_resub_display(state)),
            "",
        )

    if ev_type == "final":
        print(f"[UI][render] final -> hide status, notify_flag={sel_notify_flag(state)!r}")
        return (
            state.chat_history.as_display(),
            gr.update(value="", visible=False),
            gr.update(value=sel_cost_line(state, "ChatGPT")),
            gr.update(value=sel_model_display(state, "ChatGPT")),
            gr.update(value=sel_cost_line(state, "Claude")),
            gr.update(value=sel_model_display(state, "Claude")),
            gr.update(value=sel_cost_line(state, "Gemini")),
            gr.update(value=sel_model_display(state, "Gemini")),
            gr.update(value=sel_resub_display(state)),
            sel_notify_flag(state),
        )

    if ev_type == "error":
        # Display a transient message appended to chat and keep status hidden
        msg = text or ""
        print(f"[UI][render] error -> append message, hide status")
        disp = state.chat_history.as_display()
        try:
            disp.append((None, msg))
        except Exception as e:
            print(f"[UI][render] error append failed: {e}")
            traceback.print_exc()
        return (
            disp,
            gr.update(value="", visible=False),
            gr.update(value=sel_cost_line(state, "ChatGPT")),
            gr.update(value=sel_model_display(state, "ChatGPT")),
            gr.update(value=sel_cost_line(state, "Claude")),
            gr.update(value=sel_model_display(state, "Claude")),
            gr.update(value=sel_cost_line(state, "Gemini")),
            gr.update(value=sel_model_display(state, "Gemini")),
            gr.update(value=sel_resub_display(state)),
            "",
        )

    # Default: treat unknown types as a status no-op (keep status hidden)
    print(f"[UI][render] unknown type={ev_type}, falling back to status-like render (hidden)")
    return (
        state.chat_history.as_display(),
        gr.update(value="", visible=False),
        gr.update(value=sel_cost_line(state, "ChatGPT")),
        gr.update(value=sel_model_display(state, "ChatGPT")),
        gr.update(value=sel_cost_line(state, "Claude")),
        gr.update(value=sel_model_display(state, "Claude")),
        gr.update(value=sel_cost_line(state, "Gemini")),
        gr.update(value=sel_model_display(state, "Gemini")),
        gr.update(value=sel_resub_display(state)),
        "",
    )


def render_status_hidden(state, chat=None):
    """Render with status hidden; optionally override chat display."""
    print(f"[UI][render] status_hidden")
    display = chat if chat is not None else state.chat_history.as_display()
    return (
        display,
        gr.update(value="", visible=False),
        gr.update(value=sel_cost_line(state, "ChatGPT")),
        gr.update(value=sel_model_display(state, "ChatGPT")),
        gr.update(value=sel_cost_line(state, "Claude")),
        gr.update(value=sel_model_display(state, "Claude")),
        gr.update(value=sel_cost_line(state, "Gemini")),
        gr.update(value=sel_model_display(state, "Gemini")),
        gr.update(value=sel_resub_display(state)),
        "",
    )


def render_chat_with_message(state, message: str):
    """Append a transient message to the chat display and render with status hidden."""
    print(f"[UI][render] chat_with_message -> '{message[:80]}'")
    disp = state.chat_history.as_display()
    try:
        disp.append((None, message))
    except Exception as e:
        print(f"[UI][render] chat_with_message append failed: {e}")
        traceback.print_exc()
    return render_status_hidden(state, chat=disp)


__all__ = [
    "render_event",
    "render_status_hidden",
    "render_chat_with_message",
]


