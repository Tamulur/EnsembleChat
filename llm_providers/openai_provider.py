import os
import json
from typing import Dict, List, Optional, Tuple

try:
    import openai
    try:
        _openai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
    except AttributeError:
        _openai_client = None
except ImportError:  # pragma: no cover
    openai = None
    _openai_client = None

from .shared import TIMEOUT, LLMError, print_messages


# Default model (can be changed via set_model)
MODEL_ID = "gpt-5"


def set_model(label_or_id: str) -> str:
    global MODEL_ID
    MODEL_ID = label_or_id
    print(f"[CONFIG] OpenAI model set to: {MODEL_ID}")
    return MODEL_ID


_openai_file_cache: Dict[str, str] = {}
_openai_vector_store_cache: Dict[str, str] = {}


async def _get_openai_vector_store_id(pdf_path: str) -> str:
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")
    if not os.path.isfile(pdf_path):
        raise LLMError(f"PDF not found: {pdf_path}")

    if pdf_path in _openai_vector_store_cache:
        return _openai_vector_store_cache[pdf_path]

    try:
        vs = await _openai_client.vector_stores.create(name=f"EnsembleChat:{os.path.basename(pdf_path)}")
        vector_store_id = getattr(vs, "id", None) or (vs.get("id") if isinstance(vs, dict) else None)
        if not vector_store_id:
            raise LLMError("Failed to create OpenAI vector store (missing id)")

        with open(pdf_path, "rb") as f:
            await _openai_client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=[f],
            )

        _openai_vector_store_cache[pdf_path] = vector_store_id
        return vector_store_id
    except Exception as exc:
        raise LLMError(f"Failed to prepare OpenAI vector store: {exc}") from exc


async def call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, temperature: float = 0.7) -> Tuple[str, int, int, str]:
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")

    vector_store_id: Optional[str] = None
    if pdf_path:
        vector_store_id = await _get_openai_vector_store_id(pdf_path)

    input_payload: List[Dict] = []
    system_instructions = ""
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        text_content = msg.get("content", "")
        if role == "system":
            system_instructions = text_content
            continue
        print_messages("OpenAI", idx, role, text_content)

        content_type = "output_text" if role == "assistant" else "input_text"
        content_blocks = [{"type": content_type, "text": text_content}]
        input_payload.append({"role": role, "content": content_blocks})

    tools = [{"type": "web_search"}]
    if vector_store_id:
        tools.insert(0, {"type": "file_search", "vector_store_ids": [vector_store_id]})

    create_kwargs = {
        "model": MODEL_ID,
        "tools": tools,
        "input": input_payload,
        "instructions": system_instructions,
        "prompt_cache_key": "cache-demo-1",
        "timeout": TIMEOUT,
    }
    # Temperature is supported by OpenAI Responses API
    try:
        if isinstance(temperature, (int, float)):
            create_kwargs["temperature"] = float(temperature)
    except Exception:
        pass

    try:
        resp = await _openai_client.responses.create(**create_kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        if ("temperature" in msg) and ("unsupported" in msg or "not supported" in msg):
            try:
                if "temperature" in create_kwargs:
                    del create_kwargs["temperature"]
            except Exception:
                pass
            resp = await _openai_client.responses.create(**create_kwargs)
        else:
            raise

    text = ""
    if hasattr(resp, "output") and resp.output:
        try:
            for output_item in resp.output:
                content_blocks = getattr(output_item, "content", None)
                if not content_blocks:
                    continue
                for block in content_blocks:
                    block_text = getattr(block, "text", None)
                    if block_text:
                        text += block_text
        except Exception as exc:
            print("[WARN] Failed to read OpenAI Responses output:", exc)

    if not text and hasattr(resp, "choices") and getattr(resp, "choices"):
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = ""

    if not text:
        text = str(resp)

    prompt_tokens = 0
    completion_tokens = 0
    raw_usage = getattr(resp, "usage", None)
    if raw_usage is not None:
        prompt_tokens = getattr(raw_usage, "input_tokens", getattr(raw_usage, "prompt_tokens", 0)) or 0
        completion_tokens = getattr(raw_usage, "output_tokens", getattr(raw_usage, "completion_tokens", 0)) or 0

    cached_tokens = -1
    if raw_usage is not None:
        if isinstance(raw_usage, dict):
            cached_tokens = ((raw_usage.get("input_tokens_details") or {}).get("cached_tokens", -1) or -1)
        else:
            input_details = getattr(raw_usage, "input_tokens_details", None)
            if input_details is not None:
                cached_tokens = getattr(input_details, "cached_tokens", -1) or -1
    if cached_tokens >= 0:
        print("cached_tokens for OpenAI:", cached_tokens)
    else:
        print("information about cached_tokens was not available for OpenAI")

    # Try to serialize the full raw response for logging
    raw_text = ""
    try:
        if hasattr(resp, "model_dump_json") and callable(getattr(resp, "model_dump_json")):
            raw_text = resp.model_dump_json(indent=2)  # type: ignore[attr-defined]
        elif hasattr(resp, "to_dict") and callable(getattr(resp, "to_dict")):
            raw_text = json.dumps(resp.to_dict(), indent=2, default=str)
        elif isinstance(resp, dict):
            raw_text = json.dumps(resp, indent=2, default=str)
        else:
            # Fallback string representation
            raw_text = str(resp)
    except Exception:
        try:
            raw_text = str(resp)
        except Exception:
            raw_text = "[unserializable OpenAI response]"

    return text, prompt_tokens, completion_tokens, raw_text


