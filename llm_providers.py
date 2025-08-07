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
    import google.genai as genai
    from google.genai import types  # Updated to use new google-genai package
    _gemini_client = genai.Client()
except ImportError:  # pragma: no cover
    genai = None
    _gemini_client = None

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


def print_messages(model_label: str, idx: int, role: str, content):
    """Utility for debug logging of messages sent to providers.

    Safely handles content that can be either a string or a list (Anthropic-style
    message blocks). Long text is truncated for readability.
    """
    # Convert complex content structures to a readable string for logging
    if isinstance(content, list):
        # Extract text fields from common content block formats
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

    # Truncate to 120 characters for concise output
    if len(rendered) > 120:
        rendered = rendered[:117] + "..."

    print(f"Message to {model_label} [{idx}, {role}]: '{rendered}'")



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


async def _openai_call(messages: List[Dict[str, str]], pdf_path: Optional[str]) -> Tuple[str, int, int]:
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

        print_messages("OpenAI", idx, role, text_content)

        # Each content entry must be a list of blocks
        # Use correct content type based on role: "output_text" for assistant, "input_text" for others
        content_type = "output_text" if role == "assistant" else "input_text"
        content_blocks = [{"type": content_type, "text": text_content}]
        # Attach the PDF to the first user message only
        if file_id and role == "user" and not hasAppendedPDF:
            content_blocks.insert(0, {"type": "input_file", "file_id": file_id})
            hasAppendedPDF = True

        input_payload.append({"role": role, "content": content_blocks})
    resp = await _openai_client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type":"web_search"}],
        input=input_payload,
        temperature=0.7,
        timeout=TIMEOUT,
    )

    # -------------------------------
    # Parse response
    # -------------------------------
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
                # Iterate through all output items to find message content
                # (resp.output[0] might be web search results, actual message could be later)
                parts = []
                for output_item in resp.output:
                    # Look for items that have content blocks (actual messages)
                    content_blocks = getattr(output_item, "content", [])
                    if content_blocks:
                        for block in content_blocks:
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
                # Iterate through all output items to find content blocks
                parts = []
                for output_item in resp["output"]:
                    if isinstance(output_item, dict) and "content" in output_item:
                        blocks = output_item.get("content", [])
                        for block in blocks:
                            if isinstance(block, dict) and "text" in block:
                                parts.append(block.get("text", ""))
                text = "".join(parts)
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


async def _anthropic_call(messages: List[Dict[str, str]], pdf_path: Optional[str]) -> Tuple[str, int, int]:
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

    for i, msg in enumerate(messages):
        print_messages("Anthropic", i, msg.get("role"), msg.get("content"))

    # Prepare API call parameters
    api_params = {
        "model": CLAUDE_MODEL,
        "messages": processed_messages,
        "max_tokens": 20000,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 8192
        },
        # For some reason the output is truncated when using the search tool. Maybe use streaming and higher max_tokens?
        # "tools": [{
        #     "type": "web_search_20250305",
        #     "name": "web_search",
        #     "max_uses": 3
        # }]
    }
    
    # Add system prompt if we have one
    if system_prompt:
        api_params["system"] = system_prompt

    resp = await client.messages.create(**api_params)

    answer = next(
        (blk.text for blk in resp.content if blk.type == "text"), ""
    )
    return answer, 0, 0



def dicts_to_gemini_history(msgs: List[Dict[str, str]]) -> List[types.Content]:
    """Convert [{'role': str, 'content': str}, …] to SDK-ready history."""
    return [
        types.Content(
            role="model" if m["role"] == "assistant" else m["role"],
            parts=[types.Part.from_text(text=m["content"])]
        )
        for m in msgs
    ]

def _get_gemini_file_resource(pdf_path: str):
    if genai is None:
        raise LLMError("google-genai package not installed")
    if pdf_path in _gemini_file_cache:
        return _gemini_file_cache[pdf_path]
    file_resource = _gemini_client.files.upload( file=pdf_path)
    _gemini_file_cache[pdf_path] = file_resource
    return file_resource

async def _gemini_call(messages: List[Dict[str, str]], pdf_path: Optional[str]) -> Tuple[str, int, int]:
    if genai is None:
        raise LLMError("google-genai package not installed")

    # Debug: print all messages
    for idx, msg in enumerate(messages):
        print_messages("Gemini", idx, msg.get("role"), msg.get("content"))

    system_instruction = messages.pop(0)["content"]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("`messages` must end with a user message in this workflow.")

    last_user_content = messages[-1]["content"]   # newest user prompt
    prior_history    = dicts_to_gemini_history(messages[:-1])  # everything before it

    pdf = _get_gemini_file_resource(pdf_path)
    pdf_part = types.Part.from_uri(
        file_uri=pdf.uri,          # ← the URI returned by files.upload()
        mime_type=pdf.mime_type    # ← "application/pdf"
    )     
    pdf_content = types.Content(role="user", parts=[pdf_part])   # PDF first, no text
    prior_history.insert(0, pdf_content)
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[grounding_tool]
    )
    chat = _gemini_client.chats.create(
            model=GEMINI_MODEL,
            config=config,
            history=prior_history)

    resp = chat.send_message(last_user_content)
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
    retries : int
        Automatic retry count for proposer calls.
    """

    async def _inner():
        ml = model_label.lower()
        if ml == "o3":
            return await _openai_call(messages, pdf_path)
        elif ml == "claude":
            return await _anthropic_call(messages, pdf_path)
        elif ml == "gemini":
            return await _gemini_call(messages, pdf_path)
        else:
            raise ValueError(f"Unknown model label: {model_label}")

    try:
        return await _retry(_inner, retries=retries)
    except Exception as e:
        raise LLMError(str(e))
