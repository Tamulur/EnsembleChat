import os
import asyncio
from typing import Dict, List, Tuple, Optional

# Optional third-party clients
try:
    import openai
    # Initialize async client (OpenAI Python SDK v1+)
    try:
        _openai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
       base_url="https://api.openai.com/v1")
    except AttributeError:
        # Fallback if AsyncOpenAI is not available (older SDK)
        _openai_client = None
except ImportError:  # pragma: no cover
    openai = None
    _openai_client = None

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

# Model identifiers (override with env vars / config as desired)
OPENAI_MODEL = "gpt-4.1"
CLAUDE_MODEL = "claude-sonnet-4-0"
GEMINI_MODEL = "gemini-2.5-pro"

TIMEOUT = 120  # seconds


class LLMError(Exception):
    """Raised when an LLM provider call fails."""


# ---------------------------------------------------------------------------
# File-upload caching so we only upload the PDF once per provider per session
# ---------------------------------------------------------------------------
_openai_file_cache: Dict[str, str] = {}
_anthropic_file_cache: Dict[str, dict] = {}  # Changed to dict since we store content objects now
_gemini_file_cache: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Provider-specific helpers
# ---------------------------------------------------------------------------

async def _get_openai_file_id(pdf_path: str) -> str:
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")
    if pdf_path in _openai_file_cache:
        return _openai_file_cache[pdf_path]

    resp = await _openai_client.files.create(file=open(pdf_path, "rb"), purpose="assistants")
    # `files.create` may return either a File object or a plain string ID depending on SDK version
    if isinstance(resp, str):
        file_id = resp
    else:
        # OpenAIObject or dataclass with attr access
        file_id = getattr(resp, "id", None) or resp.get("id")
    _openai_file_cache[pdf_path] = file_id
    if not file_id:
        print("Failed to get OpenAI file_id") # DEBUG
    return file_id


async def _openai_call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, stream: bool = False) -> Tuple[str, int, int]:
    """Call the OpenAI Responses API (v1+ SDK) with optional PDF attachment."""
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")

    file_id: Optional[str] = None
    if pdf_path:
        file_id = await _get_openai_file_id(pdf_path)
    else:
        print("No PDF path provided") # DEBUG
    if not file_id:
        print("Not using OpenAI file_id") # DEBUG

    # Build the `input` payload for the Responses API
    input_payload: List[Dict] = []
    hasAppendedPDF = False
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        text_content = msg.get("content", "")

        print(f"Message {idx}: role={role}, content={text_content}")  # DEBUG

        # Each content entry must be a list of blocks
        content_blocks = [{"type": "input_text", "text": text_content}]
        # Attach the PDF to the first user message only
        if file_id and role == "user" and not hasAppendedPDF:
            print("Using OpenAI file_id:", file_id) # DEBUG
            content_blocks.insert(0, {"type": "input_file", "file_id": file_id})
            hasAppendedPDF = True

        input_payload.append({"role": role, "content": content_blocks})
    resp = await _openai_client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type":"web_search"}],  # optional
        input=input_payload,
        temperature=0.7,
        stream=stream,
        timeout=TIMEOUT,
    )

    # -------------------------------
    # Parse response (stream / non-stream)
    # -------------------------------
    if stream:
        collected: List[str] = []
        async for chunk in resp:
            # Streaming may yield plain text or structured chunks
            if isinstance(chunk, str):
                collected.append(chunk)
                continue

            # Structured chunk with choices list
            if hasattr(chunk, "choices") and chunk.choices:
                # OpenAI SDK objects expose .choices, dicts expose ["choices"]
                first_choice = chunk.choices[0] if isinstance(chunk.choices, list) else chunk["choices"][0]
                if isinstance(first_choice, dict):
                    delta = first_choice.get("delta")
                else:
                    delta = getattr(first_choice, "delta", None)
                if delta:
                    if isinstance(delta, dict):
                        message_part = delta.get("message")
                    else:
                        message_part = getattr(delta, "message", None)
                    if message_part:
                        if isinstance(message_part, dict):
                            collected.append(message_part.get("content", ""))
                        else:
                            collected.append(getattr(message_part, "content", ""))
            else:
                # Fallback: attempt to cast to str
                collected.append(str(chunk))
        return "".join(collected), 0, 0
    else:
        if isinstance(resp, str):
            text = resp
            print("OpenAI text from string:", text) # DEBUG
            usage_dict = {}
        else:
            # ---------------- Extract text depending on schema ----------------
            text = ""
            # 1) Responses API (new): resp.output -> list of messages -> content blocks
            if hasattr(resp, "output") and resp.output:
                try:
                    first_msg = resp.output[0]
                    parts = []
                    for block in getattr(first_msg, "content", []):
                        block_text = getattr(block, "text", None)
                        if block_text:
                            parts.append(block_text)
                    text = "".join(parts)
                except Exception as e:
                    print("[WARN] Failed to parse Response.output:", e)
            # 2) Legacy chat/completions style with .choices
            if not text and hasattr(resp, "choices") and resp.choices:
                text = resp.choices[0].message.content
            # 3) Fallback for dict representations
            if not text and isinstance(resp, dict):
                if "output" in resp and resp["output"]:
                    blocks = resp["output"][0].get("content", [])
                    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
                elif "choices" in resp:
                    text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            # 4) Ultimate fallback
            if not text:
                text = str(resp)
                print("OpenAI text fallback str:", text)  # DEBUG
            # ---------------- Normalize usage object ----------------
            raw_usage = getattr(resp, "usage", None)
            if raw_usage is None and isinstance(resp, dict):
                raw_usage = resp.get("usage")
            if raw_usage is None:
                usage_dict = {}
            elif isinstance(raw_usage, dict):
                usage_dict = raw_usage
            else:
                # ResponseUsage or similar object from SDK
                usage_dict = {
                    "prompt_tokens": getattr(raw_usage, "input_tokens", getattr(raw_usage, "prompt_tokens", 0)),
                    "completion_tokens": getattr(raw_usage, "output_tokens", getattr(raw_usage, "completion_tokens", 0)),
                }
        return text, usage_dict.get("prompt_tokens", 0), usage_dict.get("completion_tokens", 0)


async def _get_anthropic_file_content(pdf_path: str) -> dict:
    """Get PDF content for Anthropic API (files are embedded directly, not uploaded separately)."""
    if anthropic is None:
        raise LLMError("anthropic package not installed")
    if pdf_path in _anthropic_file_cache:
        return _anthropic_file_cache[pdf_path]

    # Read PDF file and encode as base64
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    import base64
    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
    
    # Create document content block for modern Anthropic API
    document_content = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": pdf_base64
        }
    }
    
    _anthropic_file_cache[pdf_path] = document_content
    return document_content


async def _anthropic_call(messages: List[Dict[str, str]], pdf_path: Optional[str], *, stream: bool = False) -> Tuple[str, int, int]:
    if anthropic is None:
        raise LLMError("anthropic package not installed")

    client = anthropic.AsyncAnthropic()
    
    # Separate system messages from other messages
    system_messages = []
    non_system_messages = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system_messages.append(msg.get("content", ""))
        else:
            non_system_messages.append(msg)
    
    # Combine system messages into a single system prompt
    system_prompt = "\n\n".join(system_messages) if system_messages else None
    
    # Process non-system messages and embed PDF content directly if provided
    processed_messages = []
    for i, msg in enumerate(non_system_messages):
        processed_msg = msg.copy()
        
        # Add PDF content to the first user message
        if pdf_path and msg.get("role") == "user" and i == 0:
            pdf_content = await _get_anthropic_file_content(pdf_path)
            
            # Convert content to list format if it's a string
            if isinstance(processed_msg["content"], str):
                text_content = processed_msg["content"]
                processed_msg["content"] = [
                    pdf_content,
                    {"type": "text", "text": text_content}
                ]
            else:
                # If content is already a list, prepend the PDF
                processed_msg["content"] = [pdf_content] + processed_msg["content"]
        
        processed_messages.append(processed_msg)

    # Prepare API call parameters
    api_params = {
        "model": CLAUDE_MODEL,
        "messages": processed_messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    
    # Add system prompt if we have one
    if system_prompt:
        api_params["system"] = system_prompt

    if stream:
        api_params["stream"] = True
        stream_resp = await client.messages.create(**api_params)
        collected: List[str] = []
        async for chunk in stream_resp:
            if chunk.type == "content_block_delta":
                collected.append(chunk.delta.text)
        return "".join(collected), 0, 0
    else:
        resp = await client.messages.create(**api_params)
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
