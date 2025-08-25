import asyncio
from typing import AsyncIterator, Dict, List

from ensemble_chat.core.session_state import SessionState
from ensemble_chat.core.proposer import call_proposer, call_synthesis
from ensemble_chat.core.aggregator import call_aggregator, first_non_empty_line, text_after_first_line, format_proposal_packet
from ensemble_chat.core.utils import save_chat, create_user_friendly_error_message


async def run_single(
    model_label: str,
    last_user: str,
    state: SessionState,
) -> AsyncIterator[Dict]:
    """Drive a single-LLM call and yield status/final events.

    Events:
      - {'type': 'status', 'text': str}
      - {'type': 'final', 'text': str}
    """

    # Capture run id to support cooperative cancellation
    run_id_at_start = getattr(state, "_run_id", 0)
    yield {"type": "status", "text": f"**Status:** Waiting for {model_label}…"}

    try:
        reply_text = await call_proposer(
            model_label,
            last_user,
            state.chat_history.entries(),
            state.pdf_path,
            state.cost_tracker,
            retries=5,
            temperature=state.temperature,
        )
    except asyncio.CancelledError:
        # Cooperative cancellation: do not mutate state
        raise
    except Exception as e:
        reply_text = create_user_friendly_error_message(e, model_label)

    # If invalidated by a newer click, stop without mutating state
    if getattr(state, "_run_id", run_id_at_start) != run_id_at_start:
        raise asyncio.CancelledError()

    # Update state
    state.chat_history.add_assistant(reply_text)
    state.model_histories[model_label].append((last_user, reply_text))
    save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)

    yield {"type": "final", "text": reply_text}


async def run_multi(
    models: List[str],
    aggregator_label: str,
    last_user: str,
    state: SessionState,
) -> AsyncIterator[Dict]:
    """Drive the multi-LLM proposer + aggregator flow and yield UI events.

    Events yielded are dicts with a 'type' key:
      - {'type': 'status', 'text': str}
      - {'type': 'final', 'text': str}
    """

    # Capture run id to support cooperative cancellation
    run_id_at_start = getattr(state, "_run_id", 0)

    # Status: dispatching
    yield {"type": "status", "text": "**Status:** Sending requests for proposals…"}

    # Fan out proposer calls
    async def proposer_task(model: str):
        try:
            result = await call_proposer(
                model,
                last_user,
                state.chat_history.entries(),
                state.pdf_path,
                state.cost_tracker,
                retries=5,
                temperature=state.temperature,
            )
            return model, result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return model, create_user_friendly_error_message(e, model)

    tasks = [proposer_task(m) for m in models]
    proposals_by_model: Dict[str, str] = {}
    num_models = len(models)

    # Initial collecting status
    yield {"type": "status", "text": f"**Status:** Collecting replies (0/{num_models})..."}

    pending = set(map(asyncio.create_task, tasks))
    try:
        i = 0
        while pending:
            # Check invalidation before waiting
            if getattr(state, "_run_id", run_id_at_start) != run_id_at_start:
                raise asyncio.CancelledError()
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                model, proposal = await d
                proposals_by_model[model] = proposal
                i += 1
                yield {"type": "status", "text": f"**Status:** Collecting replies ({i}/{num_models})..."}
    except asyncio.CancelledError:
        for t in pending:
            t.cancel()
        # Propagate to allow upstream to stop without mutating state
        raise

    proposals = [proposals_by_model[m] for m in models]

    # If invalidated, stop before mutating any state below
    if getattr(state, "_run_id", run_id_at_start) != run_id_at_start:
        raise asyncio.CancelledError()

    # Append initial proposals to model histories for tabs
    for m, p in zip(models, proposals):
        state.model_histories[m].append((last_user, p))

    # Aggregation loop up to 5 iterations
    for iteration in range(1, 6):
        # If invalidated, stop before doing more work
        if getattr(state, "_run_id", run_id_at_start) != run_id_at_start:
            raise asyncio.CancelledError()
        yield {"type": "status", "text": f"**Status:** Aggregating replies, iteration {iteration}…"}

        try:
            agg_out = await call_aggregator(
                proposals,
                last_user,
                state.chat_history.entries(),
                state.pdf_path,
                state.cost_tracker,
                iteration,
                aggregator_label,
                temperature=state.temperature,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_message = create_user_friendly_error_message(e, aggregator_label)
            fallback_reply = f"**Aggregation failed:** {error_message}\n\n**Here are the individual proposals:**\n\n"
            for i, proposal in enumerate(proposals, 1):
                fallback_reply += f"**Proposal {i}:**\n{proposal}\n\n"
            state.chat_history.add_assistant(fallback_reply.strip())
            save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
            yield {"type": "final", "text": state.chat_history._entries[-1]["text"]}
            return

        # If invalidated, stop before mutating state with aggregator output
        if getattr(state, "_run_id", run_id_at_start) != run_id_at_start:
            raise asyncio.CancelledError()
        first = first_non_empty_line(agg_out).lower()
        if "final" in first:
            final_reply = text_after_first_line(agg_out)
            state.chat_history.add_assistant(final_reply)
            save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
            yield {"type": "final", "text": final_reply}
            return
        elif "request synthesis from proposers" in first and iteration < 5:
            aggregator_notes = text_after_first_line(agg_out)
            # Build proposals packet for synthesis and logging in resubmissions
            packet = format_proposal_packet(proposals)
            full_prompt_for_logging = "[" + packet + "\n" + aggregator_notes + "]"
            state.resubmissions_history.append(("", full_prompt_for_logging))

            async def synth_task(model: str):
                try:
                    return await call_synthesis(
                        model,
                        last_user,
                        state.chat_history.entries(),
                        state.pdf_path,
                        aggregator_notes=aggregator_notes,
                        cost_tracker=state.cost_tracker,
                        retries=5,
                        temperature=state.temperature,
                        proposals_packet=packet,
                    )
                except Exception as e:
                    return create_user_friendly_error_message(e, model)

            proposals = await asyncio.gather(*[synth_task(m) for m in models])
            for m, p in zip(models, proposals):
                state.model_histories[m].append((last_user, p))
            # Continue to next aggregation iteration
            continue
        else:
            # Treat as final reply if neither explicit FINAL nor synthesis request
            state.chat_history.add_assistant(agg_out)
            save_chat(state.chat_id, state.chat_history.entries(), state.pdf_path)
            yield {"type": "final", "text": agg_out}
            return


