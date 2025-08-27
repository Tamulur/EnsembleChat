from typing import Tuple, List, Optional
import traceback

from ensemble_chat.core.session_state import SessionState, save_session


def prepare_user_and_state(user_input: str, s: SessionState) -> Tuple[List, str, SessionState]:
    """Prepare chat history for a new run based on the user's input.

    - If user_input is empty: treat as redo → remove the most recent assistant reply
      so the last user message becomes active again.
    - If user_input is non-empty: remove any trailing unanswered user and add the new input.

    Returns a tuple compatible with the UI handler outputs:
      (chat_display, cleared_user_box_value, session_state)
    """
    if not user_input:
        print(f"[CORE][interactions] prepare_user_and_state: redo flow (empty input)")
        # Regenerate: keep last user message, ensure no dangling assistant
        try:
            s.chat_history.remove_last_assistant()
        except Exception as e:
            print(f"[CORE][interactions] remove_last_assistant error: {e}")
            traceback.print_exc()
        try:
            save_session(s)
        except Exception as e:
            print(f"[CORE][interactions] save_session error (redo): {e}")
        return s.chat_history.as_display(), "", s

    # New query: remove any unanswered trailing user from an aborted run
    try:
        s.chat_history.remove_last_user()
    except Exception as e:
        print(f"[CORE][interactions] remove_last_user error: {e}")
        traceback.print_exc()
    s.chat_history.add_user(user_input)
    print(f"[CORE][interactions] prepare_user_and_state: new input len={len(user_input)}")
    try:
        save_session(s)
    except Exception as e:
        print(f"[CORE][interactions] save_session error (new input): {e}")
    return s.chat_history.as_display(), "", s


def resolve_last_user_text(s: SessionState) -> Optional[str]:
    """Return the most recent user message text, or None if none present."""
    print(f"[CORE][interactions] resolve_last_user_text: entries={len(s.chat_history.entries())}")
    for entry in reversed(s.chat_history.entries()):
        if entry.get("role") == "user":
            return entry.get("text")
    return None


def budget_guard_message(s: SessionState, estimate: float = 0.05) -> Optional[str]:
    """Return a budget warning message if the next action would exceed the budget."""
    if s.cost_tracker.will_exceed_budget(estimate):
        print(f"[CORE][interactions] budget_guard triggered: total={s.cost_tracker.total_cost}")
        return "Budget exceeded ($5). Start a new session or change selection."
    return None


def pop_last_user_to_input(s: SessionState) -> Tuple[List, str, SessionState]:
    """Remove the trailing user message (if any) and return it for the input box.

    Returns a tuple compatible with the UI handler outputs:
      (chat_display, user_box_value, session_state)
    """
    try:
        entries = s.chat_history.entries()
        if entries and entries[-1].get("role") == "user":
            last_text = entries[-1].get("text", "")
            removed = s.chat_history.remove_last_user()
            if removed:
                try:
                    save_session(s)
                except Exception as e:
                    print(f"[CORE][interactions] save_session error (pop_last_user): {e}")
                return s.chat_history.as_display(), last_text, s
    except Exception as e:
        print(f"[CORE][interactions] pop_last_user_to_input error: {e}")
        traceback.print_exc()
    # Nothing to pop; keep input empty
    return s.chat_history.as_display(), "", s


__all__ = [
    "prepare_user_and_state",
    "resolve_last_user_text",
    "budget_guard_message",
    "pop_last_user_to_input",
]


