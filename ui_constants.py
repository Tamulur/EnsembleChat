from pathlib import Path


# Constants
ICON_DIR = Path(__file__).parent / "Icons"

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
}


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


