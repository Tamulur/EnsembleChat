import asyncio
from typing import List, Dict

import prompts
from llm_providers import call_llm, LLMError
from utils import CostTracker


def format_proposal_packet(proposals: List[str]) -> str:
    parts = []
    for idx, text in enumerate(proposals, 1):
        parts.append(f"# Proposed Reply {idx}:\n{text}\n")
    return "\n".join(parts)


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def text_after_first_line(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip()


async def call_aggregator(proposals: List[str], user_input: str, chat_history: List[Dict], pdf_path: str,
                          cost_tracker: CostTracker, iteration: int, stream_final: bool = False):
    agg_messages = []
    agg_messages.append({"role": "system", "content": prompts.aggregator_system()})

    # Attach chat history
    for entry in chat_history:
        role = entry.get("role")
        content = entry.get("text")
        agg_messages.append({"role": role, "content": content})


    # Aggregator user prompt file depending on forced final
    if iteration == 5:
        agg_user_prompt = prompts.aggregator_force_user()
    else:
        agg_user_prompt = prompts.aggregator_user()

    packet = format_proposal_packet(proposals)
    agg_messages.append({"role": "user", "content": "[" + agg_user_prompt + "\n" + packet + "]"})

    # Claude aggregator label is fixed
    retries = 1  # aggregator retry once on failure per spec
    try:
        response_text, pt, ct = await call_llm("claude", agg_messages, pdf_path=pdf_path, stream=stream_final, retries=retries)
        cost_tracker.add_usage("claude", pt, ct)
        return response_text
    except LLMError as e:
        if retries == 1:
            # already retried once
            raise
        else:
            raise
