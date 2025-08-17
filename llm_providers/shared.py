import asyncio
from typing import Any


TIMEOUT = 180  # seconds


class LLMError(Exception):
    """Raised when an LLM provider call fails."""


def print_messages(model_label: str, idx: int, role: str, content: Any) -> None:
    """Utility for debug logging of messages sent to providers.

    Safely handles content that can be either a string or a list (Anthropic-style
    message blocks). Long text is truncated for readability.
    """
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text_part = block.get("text") or block.get("type") or str(block)
                parts.append(str(text_part))
            else:
                parts.append(str(block))
        rendered = " | ".join(parts)
    else:
        rendered = str(content)

    if len(rendered) > 100:
        rendered = rendered[:97] + "..."
    rendered = rendered.replace('\n', ' ')

    print(f"Message to {model_label} [{idx}, {role}]: '{rendered}'")


async def retry(func, retries: int = 5, context: str | None = None):
    delay = 1
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            attempt_num = attempt + 1
            is_last = attempt == retries - 1
            prefix = f"[RETRY]{' ' + context if context else ''}"
            if is_last:
                print(f"{prefix} Attempt {attempt_num}/{retries} failed: {e}. No more retries.")
                raise
            else:
                print(f"{prefix} Attempt {attempt_num}/{retries} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2


