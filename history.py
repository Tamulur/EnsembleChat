from typing import List, Dict


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

    def as_display(self):
        # Converts to list of tuples (user, assistant) for Gradio Chatbot
        display = []
        user_msg = None
        for entry in self._entries:
            if entry["role"] == "user":
                user_msg = entry["text"]
            elif entry["role"] == "assistant" and user_msg is not None:
                display.append((user_msg, entry["text"]))
                user_msg = None
        return display
