from typing import List

from ensemble_chat.core.proposer import call_proposer
from ensemble_chat.core.utils import save_chat, create_user_friendly_error_message
from ensemble_chat.core.session_state import SessionState


async def handle_single(model_label: str, user_input: str, state: SessionState) -> List:
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


__all__ = [
    "handle_single",
]


