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
OPENAI_MODEL = "o3" #"gpt-4.1"
CLAUDE_MODEL = "claude-sonnet-4-0"
GEMINI_MODEL = "gemini-2.5-pro"

TIMEOUT = 180  # seconds


class LLMError(Exception):
    """Raised when an LLM provider call fails."""


# ---------------------------------------------------------------------------
# File-upload caching so we only upload the PDF once per provider per session
# ---------------------------------------------------------------------------
_openai_file_cache: Dict[str, str] = {}
# Cache pdf_path -> vector_store_id so we only build the index once per session
_openai_vector_store_cache: Dict[str, str] = {}
_anthropic_file_cache: Dict[str, str] = {}  # Cache pdf_path -> file_id for Anthropic Files API
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

    # Truncate to 100 characters for concise output
    if len(rendered) > 100:
        rendered = rendered[:97] + "..."
    rendered = rendered.replace('\n', ' ')

    print(f"Message to {model_label} [{idx}, {role}]: '{rendered}'")



# ---------------------------------------------------------------------------
# Provider-specific helpers
# ---------------------------------------------------------------------------

async def _get_openai_file_id(pdf_path: str) -> str:
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")
    if pdf_path in _openai_file_cache:
        return _openai_file_cache[pdf_path]

    resp = await _openai_client.files.create(file=open(pdf_path, "rb"), purpose="user_data")
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


async def _get_openai_vector_store_id(pdf_path: str) -> str:
    """Create (once) a vector store for the given PDF and return its id.

    Uses OpenAI File Search (Vector Stores). Uploads the local PDF directly into
    the vector store and waits for indexing to complete.
    """
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")
    if not os.path.isfile(pdf_path):
        raise LLMError(f"PDF not found: {pdf_path}")

    if pdf_path in _openai_vector_store_cache:
        return _openai_vector_store_cache[pdf_path]

    try:
        # Create a new vector store
        vs = await _openai_client.vector_stores.create(
            name=f"EnsembleChat:{os.path.basename(pdf_path)}"
        )
        vector_store_id = getattr(vs, "id", None) or (vs.get("id") if isinstance(vs, dict) else None)
        if not vector_store_id:
            raise LLMError("Failed to create OpenAI vector store (missing id)")

        # Upload the file and wait for indexing to complete
        # The SDK supports uploading file-like objects directly
        with open(pdf_path, "rb") as f:
            await _openai_client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=[f],
            )

        _openai_vector_store_cache[pdf_path] = vector_store_id
        return vector_store_id
    except Exception as exc:
        raise LLMError(f"Failed to prepare OpenAI vector store: {exc}") from exc


async def _openai_call(messages: List[Dict[str, str]], pdf_path: Optional[str]) -> Tuple[str, int, int]:
    """Call the OpenAI Responses API (v1+ SDK) with optional PDF attachment."""
    if _openai_client is None:
        raise LLMError("openai package not installed or AsyncOpenAI not available")

    vector_store_id: Optional[str] = None
    if pdf_path:
        vector_store_id = await _get_openai_vector_store_id(pdf_path)
    else:
        print("No PDF path provided")  # DEBUG

    # Build the `input` payload for the Responses API
    input_payload: List[Dict] = []
    system_instructions = ""
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        text_content = msg.get("content", "")
        if role == "system":
            system_instructions = text_content
            continue
        print_messages("OpenAI", idx, role, text_content)

        # Each content entry must be a list of blocks
        # Use correct content type based on role: "output_text" for assistant, "input_text" for others
        content_type = "output_text" if role == "assistant" else "input_text"
        content_blocks = [{"type": content_type, "text": text_content}]

        input_payload.append({"role": role, "content": content_blocks})

    # Enable web search and file search tools; attach the vector store if present
    tools = [{"type": "web_search"}]
    if vector_store_id:
        tools.insert(0, {"type": "file_search",
        "vector_store_ids": [vector_store_id]})

    create_kwargs = {
        "model": OPENAI_MODEL,
        "tools": tools,
        "input": input_payload,
        "instructions": system_instructions,
        "prompt_cache_key": "cache-demo-1",
        "timeout": TIMEOUT,
    }

    resp = await _openai_client.responses.create(**create_kwargs)

    print("Full OpenAI response:", resp)
    
    # -------------------------------
    # Parse response (streamlined for Responses API)
    # -------------------------------
    text = ""
    # Prefer Responses API shape: resp.output -> items -> content[].text
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

    # Minimal legacy fallback for chat/completions
    if not text and hasattr(resp, "choices") and getattr(resp, "choices"):
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = ""

    if not text:
        text = str(resp)

    # Usage tokens
    prompt_tokens = 0
    completion_tokens = 0
    raw_usage = getattr(resp, "usage", None)
    if raw_usage is not None:
        prompt_tokens = getattr(raw_usage, "input_tokens", getattr(raw_usage, "prompt_tokens", 0)) or 0
        completion_tokens = getattr(raw_usage, "output_tokens", getattr(raw_usage, "completion_tokens", 0)) or 0

    # Log cached tokens (Responses API: usage.input_tokens_details.cached_tokens)
    cached_tokens = -1
    if raw_usage is not None:
        if isinstance(raw_usage, dict):
            cached_tokens = (
                (raw_usage.get("input_tokens_details") or {}).get("cached_tokens", -1) or -1
            )
        else:
            input_details = getattr(raw_usage, "input_tokens_details", None)
            if input_details is not None:
                cached_tokens = getattr(input_details, "cached_tokens", -1) or -1
    if cached_tokens >= 0:
        print("cached_tokens for o3:", cached_tokens)
    else:
        print("information about cached_tokens was not available for o3")

    return text, prompt_tokens, completion_tokens


async def _get_anthropic_file_content(pdf_path: str) -> dict:
    """Upload PDF once via Anthropic Files API and return a document block that references the resulting file_id."""
    if anthropic is None:
        raise LLMError("anthropic package not installed")

    # Re-use cached file_id if we've already uploaded this PDF in the current session
    if pdf_path in _anthropic_file_cache:
        file_id = _anthropic_file_cache[pdf_path]
    else:
        # Initialise a client that has the required beta header so we can use the Files API
        client = anthropic.AsyncAnthropic(
            default_headers={"anthropic-beta": "files-api-2025-04-14"}
        )
        try:
            # Upload via Files API (beta)
            with open(pdf_path, "rb") as f:
                file_obj = await client.beta.files.upload(
                    file=(os.path.basename(pdf_path), f, "application/pdf")
                )
        except Exception as exc:
            raise LLMError(f"Failed to upload PDF to Anthropic Files API: {exc}") from exc

        # The returned object may be a dict or an SDK object – handle both.
        file_id = getattr(file_obj, "id", None)
        if file_id is None and isinstance(file_obj, dict):
            file_id = file_obj.get("id")
        if not file_id:
            raise LLMError("Anthropic file upload did not return a file_id")

        _anthropic_file_cache[pdf_path] = file_id

    # Build the document content block that references the uploaded file
    document_content = {
        "type": "document",
        "source": {
            "type": "file",
            "file_id": file_id
        },
        "cache_control": {"type": "ephemeral"}
    }
    return document_content


async def _anthropic_call(messages: List[Dict[str, str]], pdf_path: Optional[str]) -> Tuple[str, int, int]:
    if anthropic is None:
        raise LLMError("anthropic package not installed")

    client = anthropic.AsyncAnthropic(default_headers={"anthropic-beta": "files-api-2025-04-14"})
    
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
    return answer, resp.usage.input_tokens, resp.usage.output_tokens



def dicts_to_gemini_history(msgs: List[Dict[str, str]]) -> List[types.Content]:
    """Convert [{'role': str, 'content': str}, …] to SDK-ready history."""
    return [
        types.Content(
            role="model" if m["role"] == "assistant" else m["role"],
            parts=[types.Part.from_text(text=m["content"])]
        )
        for m in msgs
    ]

async def _get_gemini_file_resource(pdf_path: str):
    if genai is None:
        raise LLMError("google-genai package not installed")
    if pdf_path in _gemini_file_cache:
        return _gemini_file_cache[pdf_path]
    # google-genai SDK is synchronous; run upload off the event loop
    def _upload():
        return _gemini_client.files.upload(file=pdf_path)
    file_resource = await asyncio.to_thread(_upload)
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

    pdf = await _get_gemini_file_resource(pdf_path)
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
    # The chats.create and send_message methods are synchronous; run them in a thread
    def _send_message():
        chat = _gemini_client.chats.create(
            model=GEMINI_MODEL,
            config=config,
            history=prior_history,
        )
        resp = chat.send_message(last_user_content)

        # Extract response text
        text = getattr(resp, "text", None)
        if not text:
            # fall back to str for safety
            text = str(resp)
        # Extract token usage from usage_metadata
        prompt_tokens = resp.usage_metadata.prompt_token_count
        completion_tokens = resp.usage_metadata.candidates_token_count
        
        return text, prompt_tokens, completion_tokens

    text, prompt_tokens, completion_tokens = await asyncio.to_thread(_send_message)
    return text, prompt_tokens, completion_tokens

# ---------------------------------------------------------------------------
# Unified public helper with retry/backoff
# ---------------------------------------------------------------------------

async def _retry(func, retries: int = 5, context: str | None = None):
    delay = 1
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            attempt_num = attempt + 1
            is_last = attempt == retries - 1
            if is_last:
                prefix = f"[RETRY]{' ' + context if context else ''}"
                print(f"{prefix} Attempt {attempt_num}/{retries} failed: {e}. No more retries.")
                raise
            else:
                prefix = f"[RETRY]{' ' + context if context else ''}"
                print(f"{prefix} Attempt {attempt_num}/{retries} failed: {e}. Retrying in {delay}s...")
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
        return await _retry(_inner, retries=retries, context=f"model={model_label}")
    except Exception as e:
        raise LLMError(str(e))
