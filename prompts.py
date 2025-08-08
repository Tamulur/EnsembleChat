import os
from functools import lru_cache

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

@lru_cache(maxsize=32)
def _read(path: str) -> str:
    full = os.path.join(ROOT_DIR, path)
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


# Cached read of ExampleExplanations.txt
@lru_cache(maxsize=1)
def _examples() -> str:
    try:
        return _read("ExampleExplanations.txt")
    except FileNotFoundError:
        # If the examples file is missing, return empty string to avoid errors
        return ""


def _with_examples(text: str) -> str:
    """Replace the {examples} placeholder in prompts with the actual examples."""
    examples = _examples()
    if not examples:
        return text
    return text.replace("{examples}", examples)


def proposer_system(model: str) -> str:
    return _with_examples(_read(os.path.join("ProposerSystemPrompts", f"{model}.txt")))


def proposer_synthesis_prompt(model: str) -> str:
    return _read(os.path.join("SynthesizeFromProposalsPrompts", f"{model}.txt"))


def aggregator_system() -> str:
    return _with_examples(_read("AggregatorSystemPrompt.txt"))


def aggregator_user() -> str:
    return _read("AggregatorUserPrompt.txt")


def aggregator_force_user() -> str:
    return _read("AggregatorForceReplyUserPrompt.txt")
