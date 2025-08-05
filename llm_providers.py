import os
import asyncio
from typing import Dict, List, Tuple, Optional

# Optional third-party clients
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

# Model identifiers (override with env vars / config as desired)
OPENAI_MODEL = "o3"
CLAUDE_MODEL = "claude-sonnet-4"
GEMINI_MODEL = "gemini-2.5-pro"

TIMEOUT = 120  # seconds


class LLMError(Exception):
    """Raised when an LLM provider call fails."""


# ---------------------------------------------------------------------------
# File-upload caching so we only upload the PDF once per provider per session
# ---------------------------------------------------------------------------
_openai_file_cache: Dict[str, str] = {}
_anthropic_file_cache: Dict[str, str] = {}
_gemini_file_cache: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Provider-specific helpers
# ---------------------------------------------------------------------------

async def _get_openai_file_id(pdf_path: str) -> str:
    if openai is None:
        raise LLMError("openai package not installed")
    if pdf_path in _openai_file_cache:
        return _openai_file_cache[pdf_path]

    resp = await openai.files.acreate(file=open(pdf_path, "rb"), purpose="assistants")
    file_id = resp.id if hasattr(resp, "id") else resp["id"]
    _openai_file_cache[pdf_path] = file_id
    return file_id


async def _openai_call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, stream: bool = False) -> Tuple[str, int, int]:
    if openai is None:
        raise LLMError("openai package not installed")

    extra = {}
    if pdf_path:
        file_id = await _get_openai_file_id(pdf_path)
        extra["file_ids"] = [file_id]

    resp = await openai.ChatCompletion.acreate(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        stream=stream,
        timeout=TIMEOUT,
        **extra,
    )

    if stream:
        collected: List[str] = []
        async for chunk in resp:
            delta = chunk.choices[0].delta.get("content", "")
            collected.append(delta)
        text = "".join(collected)
        # Streaming usage currently not returned
        return text, 0, 0
    else:
        text = resp.choices[0].message.content
        usage = resp.usage or {}
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


async def _get_anthropic_file_id(pdf_path: str) -> str:
    if anthropic is None:
        raise LLMError("anthropic package not installed")
    if pdf_path in _anthropic_file_cache:
        return _anthropic_file_cache[pdf_path]

    client = anthropic.Anthropic()
    resp = await client.files.create(file=open(pdf_path, "rb"))
    file_id = resp.id if hasattr(resp, "id") else resp["id"]
    _anthropic_file_cache[pdf_path] = file_id
    return file_id


async def _anthropic_call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, stream: bool = False) -> Tuple[str, int, int]:
    if anthropic is None:
        raise LLMError("anthropic package not installed")

    client = anthropic.Anthropic()
    extra = {}
    if pdf_path:
        file_id = await _get_anthropic_file_id(pdf_path)
        extra["attachments"] = [{"file_id": file_id}]

    if stream:
        stream_resp = await client.messages.create(
            model=CLAUDE_MODEL,
            messages=messages,
            temperature=0.7,
            stream=True,
            max_tokens=4096,
            **extra,
        )
        collected: List[str] = []
        async for chunk in stream_resp:
            if chunk.type == "content_block_delta":
                collected.append(chunk.delta.text)
        return "".join(collected), 0, 0
    else:
        resp = await client.messages.create(
            model=CLAUDE_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            **extra,
        )
        text = resp.content[0].text if resp.content else ""
        return text, 0, 0


def _get_gemini_file_resource(pdf_path: str):
    if genai is None:
        raise LLMError("google-generativeai package not installed")
    if pdf_path in _gemini_file_cache:
        return _gemini_file_cache[pdf_path]
    file_resource = genai.upload_file(path=pdf_path)
    _gemini_file_cache[pdf_path] = file_resource
    return file_resource


async def _gemini_call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, stream: bool = False) -> Tuple[str, int, int]:
    if genai is None:
        raise LLMError("google-generativeai package not installed")

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Concatenate messages into a single prompt; Gemini's Python SDK doesn't yet support role syntax.
    content = "\n".join([(m["role"].upper() + ": " + m["content"]) for m in messages])

    inputs: List = []
    if pdf_path:
        file_res = _get_gemini_file_resource(pdf_path)
        inputs.append(file_res)
    inputs.append(content)

    if stream:
        resp = model.generate_content(inputs, stream=True, generation_config={"temperature": 0.7})
        text_parts: List[str] = []
        for chunk in resp:
            if chunk.candidates and chunk.candidates[0].content.parts:
                text_parts.append(chunk.candidates[0].content.parts[0].text)
        return "".join(text_parts), 0, 0
    else:
        resp = model.generate_content(inputs, generation_config={"temperature": 0.7})
        return resp.text, 0, 0


# ---------------------------------------------------------------------------
# Unified public helper with retry/backoff
# ---------------------------------------------------------------------------

async def _retry(func, retries: int = 5):
    delay = 1
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


async def call_llm(
    model_label: str,
    messages: List[Dict[str, str]],
    *,
    pdf_path: Optional[str] = None,
    stream: bool = False,
    retries: int = 1,
) -> Tuple[str, int, int]:
    """Unified async call that routes to the appropriate provider wrapper.

    Parameters
    ----------
    model_label : str
        "o3", "claude", or "gemini" (case-insensitive).
    messages : list
        OpenAI-style messages list.
    pdf_path : str | None
        Local path to PDF to attach to the request (will be uploaded once per
        provider and reused).
    stream : bool
        Whether to stream tokens back (only used in certain cases).
    retries : int
        Automatic retry count for proposer calls.
    """

    async def _inner():
        ml = model_label.lower()
        if ml == "o3":
            return await _openai_call(messages, pdf_path, stream=stream)
        elif ml == "claude":
            return await _anthropic_call(messages, pdf_path, stream=stream)
        elif ml == "gemini":
            return await _gemini_call(messages, pdf_path, stream=stream)
        else:
            raise ValueError(f"Unknown model label: {model_label}")

    try:
        return await _retry(_inner, retries=retries)
    except Exception as e:
        raise LLMError(str(e))
