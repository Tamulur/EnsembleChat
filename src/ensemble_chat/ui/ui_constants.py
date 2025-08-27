from pathlib import Path
from ensemble_chat.core.paths import project_root


# Constants
ICON_DIR = project_root() / "Icons"

BUTTONS = [
    "ChatGPT",
    "Claude",
    "Gemini",
    "ChatGPT & Gemini",
    "All",
]

# Mapping of button label to icon path
ICON_MAP = {
    "ChatGPT": str(ICON_DIR / "OpenAI.png"),
    "Claude": str(ICON_DIR / "Claude.png"),
    "Gemini": str(ICON_DIR / "Gemini.png"),
    "ChatGPT & Gemini": str(ICON_DIR / "ChatGPT_Gemini.png"),
    "All": str(ICON_DIR / "All.png"),
    "_stop_internal": str(ICON_DIR / "Stop_Sign.png"),  # Internal reference to ensure Gradio serves this file
}

# Placeholder for stop icon; used later for visual swap
STOP_ICON = str(ICON_DIR / "Stop_Sign.png")


# Shared LaTeX delimiters configuration for all Chatbot components
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]


MULTI_BUTTON_MODELS = {
    "ChatGPT & Gemini": ["ChatGPT", "Gemini"],
    "All": ["ChatGPT", "Claude", "Gemini"],
}


