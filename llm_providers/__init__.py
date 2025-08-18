from typing import Dict, List, Optional, Tuple

from .shared import LLMError, retry
from . import openai_provider as _openai
from . import anthropic_provider as _anthropic
from . import gemini_provider as _gemini


def set_openai_model(label_or_id: str) -> str:
    return _openai.set_model(label_or_id)


def set_claude_model(label_or_id: str) -> str:
    return _anthropic.set_model(label_or_id)


def set_gemini_model(label_or_id: str) -> str:
    return _gemini.set_model(label_or_id)


async def call_llm(
    model_label: str,
    messages: List[Dict[str, str]],
    *,
    pdf_path: Optional[str] = None,
    retries: int = 1,
    temperature: float = 0.7,
) -> Tuple[str, int, int]:
    async def _inner():
        ml = model_label.lower()
        if ml == "chatgpt":
            return await _openai.call(messages, pdf_path, temperature=temperature)
        elif ml == "claude":
            return await _anthropic.call(messages, pdf_path, temperature=temperature)
        elif ml == "gemini":
            return await _gemini.call(messages, pdf_path, temperature=temperature)
        else:
            raise ValueError(f"Unknown model label: {model_label}")

    try:
        return await retry(_inner, retries=retries, context=f"model={model_label}")
    except Exception as e:
        raise LLMError(str(e))

__all__ = [
    "LLMError",
    "call_llm",
    "set_openai_model",
    "set_claude_model",
    "set_gemini_model",
]


