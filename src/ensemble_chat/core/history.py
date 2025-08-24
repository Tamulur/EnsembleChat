from typing import List, Dict
import html
import re
from ensemble_chat.core.sanitization import neutralize_angle_brackets

_ZWSP = "\u200B"


def _neutralize_angle_brackets(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Decode common entities first (e.g., &lt;planning&gt; -> <planning>)
    decoded = html.unescape(text)
    # Break potential HTML tags while preserving visible brackets
    decoded = re.sub(r"<(?=[A-Za-z/])", "<" + _ZWSP, decoded)
    decoded = re.sub(r"(?<=[A-Za-z0-9/])>", _ZWSP + ">", decoded)
    return decoded


class ChatHistory:
    """Maintains official chat history (user + final replies only)."""

    def __init__(self):
        self._entries: List[Dict] = []

    def add_user(self, text: str):
        self._entries.append({"role": "user", "text": text})

    def add_assistant(self, text: str):
        self._entries.append({"role": "assistant", "text": text})

    def entries(self):
        return list(self._entries)

    def remove_last_assistant(self):
        """Remove the most recent assistant reply if it directly follows a user message.
        Returns True if a reply was removed, False otherwise."""
        if len(self._entries) >= 1 and self._entries[-1]["role"] == "assistant":
            # Just pop the assistant message – keep the preceding user input
            self._entries.pop()
            return True
        return False

    def as_display(self):
        # Converts to list of tuples (user, assistant) for Gradio Chatbot
        display = []
        user_msg = None
        for entry in self._entries:
            if entry["role"] == "user":
                # start a new conversation pair with no assistant yet
                if user_msg is not None:
                    # Edge case: consecutive user messages without assistant reply
                    display.append((neutralize_angle_brackets(user_msg), None))
                user_msg = entry["text"]
            elif entry["role"] == "assistant":
                if user_msg is None:
                    # Assistant reply without preceding user (shouldn't happen), skip
                    continue
                display.append((neutralize_angle_brackets(user_msg), neutralize_angle_brackets(entry["text"])) )
                user_msg = None
        # If there's a trailing user message without assistant yet, show it
        if user_msg is not None:
            display.append((neutralize_angle_brackets(user_msg), None))
        return display


