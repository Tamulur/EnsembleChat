import asyncio
from typing import List, Dict, Optional

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


async def call_aggregator(
    proposals: List[str],
    user_input: str,
    chat_history: List[Dict],
    pdf_path: Optional[str],
    cost_tracker: CostTracker,
    iteration: int,
    aggregator_label: str,
    temperature: float = 0.7,
):
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

    # Use selected provider as aggregator
    # Use more retries for overload-prone scenarios, but keep it simple
    try:
        response_text, pt, ct = await call_llm(
            aggregator_label,
            agg_messages,
            pdf_path=pdf_path,
            retries=1,  # Start with normal retry count
            temperature=temperature,
        )
        cost_tracker.add_usage(aggregator_label, pt, ct)
        return response_text
    except LLMError as e:
        # Check if this is a 529 overload error and give it one more chance with more retries
        error_str = str(e).lower()
        if ("529" in error_str or "overloaded" in error_str) and iteration == 1:
            print(f"[AGGREGATOR] Detected overload error on first attempt, retrying with more attempts...")
            # Only retry with more attempts on the first iteration to avoid endless loops
            response_text, pt, ct = await call_llm(
                aggregator_label,
                agg_messages,
                pdf_path=pdf_path,
                retries=3,  # More retries only for overload errors on first iteration
                temperature=temperature,
            )
            cost_tracker.add_usage(aggregator_label, pt, ct)
            return response_text
        else:
            # For non-overload errors or later iterations, just fail
            raise
