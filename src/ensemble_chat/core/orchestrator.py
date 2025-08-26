from typing import AsyncIterator, Dict
import asyncio as _asyncio

from ensemble_chat.core.run_modes import resolve_run_mode
from ensemble_chat.core.engine import run_single, run_multi
from ensemble_chat.core.session_state import SessionState, save_session
from ensemble_chat.core.run_state import start_single, start_multi, mark_final, mark_cancelled, clear, snapshot as run_snapshot
from ensemble_chat.core.selectors import active_button_label, active_button_elem_id
from ensemble_chat.core.events import event_from_raw, LogicEvent, FinalEvent, RunStartedEvent, RunCancelledEvent, RunCompletedEvent


async def run_from_label(button_label: str, last_user: str, state: SessionState) -> AsyncIterator[Dict]:
    """Dispatch to single or multi run based on the button label.

    Passes through events from the underlying engine without modification.
    """
    mode, models_or_single = resolve_run_mode(button_label)
    print(f"[CORE][orchestrator] run_from_label: label='{button_label}', mode='{mode}', models={models_or_single}, run_id={getattr(state, '_run_id', None)}")
    try:
        run_snapshot(state)
    except Exception:
        pass

    # Track current task centrally so cancellation can be handled in core
    try:
        state._current_task = _asyncio.current_task()
        print(f"[CORE][orchestrator] set current_task={state._current_task}")
    except Exception:
        pass

    try:
        if mode == "single":
            model = models_or_single[0]
            start_single(state, model)
            # Log active button info via selectors
            try:
                _ = active_button_elem_id(state)
            except Exception:
                pass
            # Emit run started lifecycle event
            print(f"[CORE][orchestrator] lifecycle -> RunStartedEvent({model})")
            yield RunStartedEvent(label=model)
            async for raw in run_single(model, last_user, state):
                # Persist state between events so UI remains read-only w.r.t. logic
                save_session(state)
                event: LogicEvent = event_from_raw(raw)
                print(f"[CORE][orchestrator] event(single {model}): {event.__class__.__name__}")
                if isinstance(event, FinalEvent):
                    mark_final(state)
                    # Log completion before yielding final so it appears even if consumer stops after final
                    print(f"[CORE][orchestrator] lifecycle -> RunCompletedEvent({model})")
                    yield event
                    # Emit run completed lifecycle event (may not be consumed if UI returns after final)
                    yield RunCompletedEvent(label=model)
                else:
                    yield event
        else:
            models = models_or_single
            # Pass the originating button label so selectors can resolve an element id
            start_multi(state, models, state.selected_aggregator, button_label=button_label)
            try:
                _ = active_button_elem_id(state)
            except Exception:
                pass
            print(f"[CORE][orchestrator] lifecycle -> RunStartedEvent(Multi)")
            yield RunStartedEvent(label="Multi")
            async for raw in run_multi(models, state.selected_aggregator, last_user, state):
                save_session(state)
                event: LogicEvent = event_from_raw(raw)
                print(f"[CORE][orchestrator] event(multi {models}): {event.__class__.__name__}")
                if isinstance(event, FinalEvent):
                    mark_final(state)
                    print(f"[CORE][orchestrator] lifecycle -> RunCompletedEvent(Multi)")
                    yield event
                    yield RunCompletedEvent(label="Multi")
                else:
                    yield event
    finally:
        try:
            print(f"[CORE][orchestrator] clearing current_task; was={getattr(state, '_current_task', None)}")
            state._current_task = None
            # Move to IDLE after completion or cancellation
            if getattr(state, "_run_phase", None) in ("completed", "cancelled"):
                clear(state)
        except Exception:
            pass


__all__ = [
    "run_from_label",
]


