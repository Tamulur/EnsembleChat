from .shared import LLMError, retry
from .registry import register_defaults, get_provider, set_provider_model


def set_openai_model(label_or_id: str) -> str:
    register_defaults()
    return set_provider_model("ChatGPT", label_or_id)


def set_claude_model(label_or_id: str) -> str:
    register_defaults()
    return set_provider_model("Claude", label_or_id)


def set_gemini_model(label_or_id: str) -> str:
    register_defaults()
    return set_provider_model("Gemini", label_or_id)


async def call_llm(
    model_label: str,
    messages: list[dict[str, str]],
    *,
    pdf_path: str | None = None,
    retries: int = 1,
    temperature: float = 0.7,
) -> tuple[str, int, int, str]:
    async def _inner():
        register_defaults()
        provider = get_provider(model_label)
        return await provider.call(messages, pdf_path, temperature=temperature)

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


