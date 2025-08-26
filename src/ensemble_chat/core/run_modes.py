from typing import List, Tuple


# Canonical labels for single-model runs
SINGLE_LABELS = {"ChatGPT", "Claude", "Gemini"}


# Mapping for multi-model buttons to proposer sets
MULTI_BUTTON_MODELS = {
    "ChatGPT & Gemini": ["ChatGPT", "Gemini"],
    "All": ["ChatGPT", "Claude", "Gemini"],
}


def resolve_run_mode(button_label: str) -> Tuple[str, List[str]]:
    """Resolve a button label to a run mode and involved models.

    Returns (mode, models):
      - ("single", [label]) for single-model runs
      - ("multi", [models...]) for multi-model runs
    """
    if button_label in SINGLE_LABELS:
        return "single", [button_label]
    models = MULTI_BUTTON_MODELS.get(button_label, [])
    return "multi", models


__all__ = [
    "resolve_run_mode",
]


