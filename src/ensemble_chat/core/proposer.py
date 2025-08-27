import asyncio
import traceback
from typing import List, Dict, Optional

from ensemble_chat.core import prompts
from ensemble_chat.llm_providers import call_llm, LLMError
from ensemble_chat.core.utils import CostTracker, write_last_raw_response


async def call_proposer(model_label: str, user_input: str, chat_history: List[Dict], pdf_path: Optional[str],
                        cost_tracker: CostTracker, retries: int = 5, temperature: float = 0.7) -> str:
    print(f"[CORE][proposer] call_proposer: model={model_label} retries={retries} temp={temperature}")
    base_messages = []
    base_messages.append({"role": "system", "content": prompts.proposer_system(model_label)})

    # attach history
    for entry in chat_history:
        role = entry.get("role")
        content = entry.get("text")
        base_messages.append({"role": role, "content": content})

    last_exc = None
    for attempt in range(1, int(max(1, retries)) + 1):
        try:
            print(f"[CORE][proposer] attempt {attempt}/{retries} -> {model_label}")
            # Defensive: provider implementations may mutate the messages list, so pass a fresh copy per attempt
            messages = [dict(m) for m in base_messages]
            text, pt, ct, raw = await call_llm(model_label, messages, pdf_path=pdf_path, retries=1, temperature=temperature)
            cost_tracker.add_usage(model_label, pt, ct)
            try:
                write_last_raw_response(model_label, raw)
            except Exception as e:
                print(f"[WARN] Failed to write last raw response for {model_label}: {e}")
            return text
        except Exception as e:
            last_exc = e
            print(f"[CORE][proposer] attempt {attempt} failed for {model_label}: {repr(e)}")
            if attempt < retries:
                delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
                try:
                    await asyncio.sleep(delay)
                except Exception as sleep_exc:
                    print(f"[CORE][proposer] sleep failed after attempt {attempt} for {model_label}: {sleep_exc}")
                    traceback.print_exc()
            else:
                raise last_exc


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
    print(f"[CORE][proposer] call_synthesis: model={model_label} retries={retries} temp={temperature}")
    base_messages = []
    base_messages.append({"role": "system", "content": prompts.proposer_system(model_label)})

    for entry in chat_history:
        base_messages.append({"role": entry.get("role"), "content": entry.get("text")})

    # Build full synthesis prompt including proposals packet and aggregator remarks
    base = prompts.proposer_synthesis_prompt(model_label)
    content_parts = [base]
    if proposals_packet:
        content_parts.append(proposals_packet)
    if aggregator_notes:
        content_parts.append(aggregator_notes)
    synth_prompt = "[" + "\n".join(content_parts) + "]"
    base_messages.append({"role": "user", "content": synth_prompt})

    last_exc = None
    for attempt in range(1, int(max(1, retries)) + 1):
        try:
            print(f"[CORE][proposer] synth attempt {attempt}/{retries} -> {model_label}")
            messages = [dict(m) for m in base_messages]
            text, pt, ct, raw = await call_llm(model_label, messages, pdf_path=pdf_path, retries=1, temperature=temperature)
            cost_tracker.add_usage(model_label, pt, ct)
            try:
                write_last_raw_response(model_label, raw)
            except Exception as e:
                print(f"[WARN] Failed to write last raw synthesis response for {model_label}: {e}")
            return text
        except Exception as e:
            last_exc = e
            print(f"[CORE][proposer] synth attempt {attempt} failed for {model_label}: {repr(e)}")
            if attempt < retries:
                delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
                try:
                    await asyncio.sleep(delay)
                except Exception as sleep_exc:
                    print(f"[CORE][proposer] synth sleep failed after attempt {attempt} for {model_label}: {sleep_exc}")
                    traceback.print_exc()
            else:
                raise last_exc


