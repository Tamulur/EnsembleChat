import asyncio
from typing import List, Dict, Optional

from ensemble_chat.core import prompts
from ensemble_chat.llm_providers import call_llm, LLMError
from ensemble_chat.core.utils import CostTracker, write_last_raw_response


async def call_proposer(model_label: str, user_input: str, chat_history: List[Dict], pdf_path: Optional[str],
                        cost_tracker: CostTracker, retries: int = 5, temperature: float = 0.7) -> str:
    messages = []
    messages.append({"role": "system", "content": prompts.proposer_system(model_label)})

    # attach history
    for entry in chat_history:
        role = entry.get("role")
        content = entry.get("text")
        messages.append({"role": role, "content": content})

    text, pt, ct, raw = await call_llm(model_label, messages, pdf_path=pdf_path, retries=retries, temperature=temperature)
    cost_tracker.add_usage(model_label, pt, ct)
    try:
        write_last_raw_response(model_label, raw)
    except Exception:
        pass
    return text


async def call_synthesis(
    model_label: str,
    user_input: str,
    chat_history: List[Dict],
    pdf_path: Optional[str],
    aggregator_notes: str,
    cost_tracker: CostTracker,
    retries: int = 5,
    temperature: float = 0.7,
    proposals_packet: str | None = None,
) -> str:
    messages = []
    messages.append({"role": "system", "content": prompts.proposer_system(model_label)})

    for entry in chat_history:
        messages.append({"role": entry.get("role"), "content": entry.get("text")})

    # Build full synthesis prompt including proposals packet and aggregator remarks
    base = prompts.proposer_synthesis_prompt(model_label)
    content_parts = [base]
    if proposals_packet:
        content_parts.append(proposals_packet)
    if aggregator_notes:
        content_parts.append(aggregator_notes)
    synth_prompt = "[" + "\n".join(content_parts) + "]"
    messages.append({"role": "user", "content": synth_prompt})

    text, pt, ct, raw = await call_llm(model_label, messages, pdf_path=pdf_path, retries=retries, temperature=temperature)
    cost_tracker.add_usage(model_label, pt, ct)
    try:
        write_last_raw_response(model_label, raw)
    except Exception:
        pass
    return text


