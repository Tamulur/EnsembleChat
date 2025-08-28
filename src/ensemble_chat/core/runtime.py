from ensemble_chat.core.session_state import SessionState
from ensemble_chat.core.run_state import mark_cancelled
import traceback


def cancel_inflight(s: SessionState) -> None:
    """Cancel any in-flight async tasks for this session and bump run id.

    Centralized cooperative cancellation to be used by UI handlers and core flows.
    """
    try:
        old_run_id = getattr(s, "_run_id", 0)
        # Bump run id to invalidate any loops checking it
        if not hasattr(s, "_run_id"):
            s._run_id = 0
        s._run_id += 1
        current = getattr(s, "_current_task", None)
        print(f"[CORE][cancel] Bumping run_id {old_run_id} -> {s._run_id}; has_current_task={current is not None}")
        if current is None:
            mark_cancelled(s)
            print("[CORE][cancel] No current task; marked as cancelled")
            return
        # Support single task or list/tuple of tasks
        tasks = current if isinstance(current, (list, tuple)) else [current]
        for t in tasks:
            try:
                if t is None:
                    continue
                # Asyncio Task cancellation (ignore errors if already done)
                if hasattr(t, "cancel"):
                    print(f"[CORE][cancel] Cancelling task={t}")
                    t.cancel()
            except Exception as e:
                print(f"[ERROR][CORE][cancel] Error cancelling task: {e}")
                traceback.print_exc()
        s._current_task = None
        mark_cancelled(s)
        print("[CORE][cancel] Cancellation signal sent; marked as cancelled")
    except Exception as e:
        # Never let cancellation path throw, but log the error
        print(f"[ERROR][CORE][cancel] cancel_inflight error: {e}")
        traceback.print_exc()


__all__ = [
    "cancel_inflight",
]


