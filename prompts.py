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
        return _read(os.path.join("Prompts", "ExampleExplanations.txt"))
    except FileNotFoundError:
        # If the examples file is missing, return empty string to avoid errors
        return ""


# Cached read of SystemPromptCommon.txt
@lru_cache(maxsize=1)
def _system_prompt_common() -> str:
    try:
        return _read(os.path.join("Prompts", "SystemPromptCommon.txt"))
    except FileNotFoundError:
        # If the system prompt common file is missing, return empty string to avoid errors
        return ""


@lru_cache(maxsize=1)
def _synthesize_prompt_common() -> str:
    """Cached read of SynthesizePromptCommon.txt for user prompts."""
    try:
        return _read(os.path.join("Prompts", "SynthesizePromptCommon.txt"))
    except FileNotFoundError:
        return ""


def _with_placeholders(text: str) -> str:
    """Replace both {SystemPromptCommon} and {examples} placeholders in prompts."""
    # Replace SystemPromptCommon first
    system_common = _system_prompt_common()
    text = text.replace("{SystemPromptCommon}", system_common if system_common else "")
    
    # Then replace examples
    examples = _examples()
    text = text.replace("{examples}", examples if examples else "")
    
    return text


def _with_user_placeholders(text: str) -> str:
    """Replace user-prompt placeholders such as {SynthesizePromptCommon}."""
    synth_common = _synthesize_prompt_common()
    return text.replace("{SynthesizePromptCommon}", synth_common if synth_common else "")


def proposer_system(model: str) -> str:
    # Normalize to file names (ChatGPT maps to its prompt file)
    filename = f"{model}.txt"
    return _with_placeholders(_read(os.path.join("Prompts", "ProposerSystemPrompts", filename)))


def proposer_synthesis_prompt(model: str) -> str:
    filename = f"{model}.txt"
    # Apply both general and user-specific placeholder replacements
    return _with_user_placeholders(
        _with_placeholders(_read(os.path.join("Prompts", "SynthesizeFromProposalsPrompts", filename)))
    )


def aggregator_system() -> str:
    return _with_placeholders(_read(os.path.join("Prompts", "AggregatorSystemPrompt.txt")))


def aggregator_user() -> str:
    return _with_user_placeholders(_read(os.path.join("Prompts", "AggregatorUserPrompt.txt")))


def aggregator_force_user() -> str:
    return _with_user_placeholders(_read(os.path.join("Prompts", "AggregatorForceReplyUserPrompt.txt")))
