import asyncio
from typing import Dict, List, Optional, Tuple

try:
    import google.genai as genai
    from google.genai import types
    _gemini_client = genai.Client()
except ImportError:  # pragma: no cover
    genai = None
    types = None  # type: ignore
    _gemini_client = None

from .shared import LLMError, print_messages


MODEL_ID = "gemini-2.5-pro"


def set_model(label_or_id: str) -> str:
    global MODEL_ID
    MODEL_ID = label_or_id
    print(f"[CONFIG] Gemini model set to: {MODEL_ID}")
    return MODEL_ID


_gemini_file_cache: Dict[str, object] = {}


def _dicts_to_gemini_history(msgs: List[Dict[str, str]]):
    return [
        types.Content(
            role="model" if m["role"] == "assistant" else m["role"],
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in msgs
    ]


async def _get_gemini_file_resource(pdf_path: str):
    if genai is None:
        raise LLMError("google-genai package not installed")
    if pdf_path in _gemini_file_cache:
        return _gemini_file_cache[pdf_path]

    def _upload():
        return _gemini_client.files.upload(file=pdf_path)

    file_resource = await asyncio.to_thread(_upload)
    _gemini_file_cache[pdf_path] = file_resource
    return file_resource


async def call(messages: List[Dict[str, str]], pdf_path: Optional[str]) -> Tuple[str, int, int]:
    if genai is None:
        raise LLMError("google-genai package not installed")

    for idx, msg in enumerate(messages):
        print_messages("Gemini", idx, msg.get("role"), msg.get("content"))

    system_instruction = messages.pop(0)["content"]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("`messages` must end with a user message in this workflow.")

    last_user_content = messages[-1]["content"]
    prior_history = _dicts_to_gemini_history(messages[:-1])

    # Attach PDF only if provided; allow running without a PDF
    if pdf_path:
        pdf = await _get_gemini_file_resource(pdf_path)
        pdf_part = types.Part.from_uri(file_uri=pdf.uri, mime_type=pdf.mime_type)
        pdf_content = types.Content(role="user", parts=[pdf_part])
        prior_history.insert(0, pdf_content)

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(system_instruction=system_instruction, tools=[grounding_tool])

    def _send_message():
        chat = _gemini_client.chats.create(model=MODEL_ID, config=config, history=prior_history)
        resp = chat.send_message(last_user_content)
        text = getattr(resp, "text", None) or str(resp)
        prompt_tokens = resp.usage_metadata.prompt_token_count
        completion_tokens = resp.usage_metadata.candidates_token_count
        return text, prompt_tokens, completion_tokens

    text, prompt_tokens, completion_tokens = await asyncio.to_thread(_send_message)
    return text, prompt_tokens, completion_tokens


