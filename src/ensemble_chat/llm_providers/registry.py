from typing import Dict, Optional

from .base import Provider, ModuleProvider
from .shared import LLMError

# Import concrete provider modules to wrap them
from . import openai_provider as _openai
from . import anthropic_provider as _anthropic
from . import gemini_provider as _gemini


_REGISTRY: Dict[str, Provider] = {}


def _canonical_label(label: str) -> str:
    low = str(label).strip().lower()
    if low in ("chatgpt", "openai", "gpt", "gpt-5", "gpt5"):
        return "ChatGPT"
    if low in ("claude", "anthropic"):
        return "Claude"
    if low in ("gemini", "google", "google-ai"):
        return "Gemini"
    # Preserve original on unknown; lookups will fail cleanly
    return label


def register(label: str, provider: Provider) -> None:
    _REGISTRY[_canonical_label(label)] = provider


def get_provider(label: str) -> Provider:
    key = _canonical_label(label)
    if key not in _REGISTRY:
        raise LLMError(f"Unknown provider label: {label}")
    return _REGISTRY[key]


def set_provider_model(label: str, model_id: str) -> str:
    provider = get_provider(label)
    return provider.set_model(model_id)


def register_defaults() -> None:
    # Wrap module-level call/set_model functions
    if "ChatGPT" not in _REGISTRY:
        register("ChatGPT", ModuleProvider(_openai.call, _openai.set_model))
    if "Claude" not in _REGISTRY:
        register("Claude", ModuleProvider(_anthropic.call, _anthropic.set_model))
    if "Gemini" not in _REGISTRY:
        register("Gemini", ModuleProvider(_gemini.call, _gemini.set_model))


__all__ = [
    "register",
    "register_defaults",
    "get_provider",
    "set_provider_model",
]


