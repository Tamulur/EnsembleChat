import os
from typing import Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

from .shared import LLMError, print_messages


MODEL_ID = "claude-sonnet-4-0"


def set_model(label_or_id: str) -> str:
    global MODEL_ID
    MODEL_ID = label_or_id
    print(f"[CONFIG] Claude model set to: {MODEL_ID}")
    return MODEL_ID


_anthropic_file_cache: Dict[str, str] = {}


async def _get_anthropic_file_content(pdf_path: str) -> dict:
    if anthropic is None:
        raise LLMError("anthropic package not installed")

    if pdf_path in _anthropic_file_cache:
        file_id = _anthropic_file_cache[pdf_path]
    else:
        client = anthropic.AsyncAnthropic(default_headers={"anthropic-beta": "files-api-2025-04-14"})
        try:
            with open(pdf_path, "rb") as f:
                file_obj = await client.beta.files.upload(file=(os.path.basename(pdf_path), f, "application/pdf"))
        except Exception as exc:
            raise LLMError(f"Failed to upload PDF to Anthropic Files API: {exc}") from exc

        file_id = getattr(file_obj, "id", None)
        if file_id is None and isinstance(file_obj, dict):
            file_id = file_obj.get("id")
        if not file_id:
            raise LLMError("Anthropic file upload did not return a file_id")

        _anthropic_file_cache[pdf_path] = file_id

    document_content = {
        "type": "document",
        "source": {"type": "file", "file_id": file_id},
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    return document_content


async def call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, temperature: float = 0.7) -> Tuple[str, int, int]:
    if anthropic is None:
        raise LLMError("anthropic package not installed")

    client = anthropic.AsyncAnthropic(default_headers={"anthropic-beta": "files-api-2025-04-14"})

    system_messages: List[str] = []
    non_system_messages: List[Dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_messages.append(msg.get("content", ""))
        else:
            non_system_messages.append(msg)

    system_prompt = "\n\n".join(system_messages) if system_messages else None

    processed_messages: List[Dict[str, object]] = []
    for i, msg in enumerate(non_system_messages):
        processed_msg = msg.copy()

        if pdf_path and msg.get("role") == "user" and i == 0:
            pdf_content = await _get_anthropic_file_content(pdf_path)
            if isinstance(processed_msg["content"], str):
                text_content = processed_msg["content"]
                processed_msg["content"] = [pdf_content, {"type": "text", "text": text_content}]
            else:
                processed_msg["content"] = [pdf_content] + processed_msg["content"]

        processed_messages.append(processed_msg)

    for i, msg in enumerate(messages):
        print_messages("Anthropic", i, msg.get("role"), msg.get("content"))

    api_params: Dict[str, object] = {
        "model": MODEL_ID,
        "messages": processed_messages,
        "max_tokens": 20000,
        "thinking": {"type": "enabled", "budget_tokens": 8192},
    }
    if system_prompt:
        api_params["system"] = system_prompt

    resp = await client.messages.create(**api_params)
    answer = next((blk.text for blk in resp.content if blk.type == "text"), "")
    return answer, resp.usage.input_tokens, resp.usage.output_tokens


