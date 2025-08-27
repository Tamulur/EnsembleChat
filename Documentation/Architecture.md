## EnsembleChat Architecture

This document explains the structure and execution flow of EnsembleChat so that developers and LLM agents can quickly locate responsibilities and make safe changes. It complements Functionality.md by focusing on where things live in code and how they interact.

### High-level overview

- **UI layer (Gradio)**: Builds the interface, wires event handlers, renders state updates. Files under `src/ensemble_chat/ui/`.
- **Core logic**: Orchestrates runs, manages session and run state, builds prompts, aggregates results, tracks cost, saves history. Files under `src/ensemble_chat/core/`.
- **LLM providers**: Thin async adapters over OpenAI, Anthropic (Claude), and Google (Gemini) SDKs, including PDF attachment logic. Files under `src/ensemble_chat/llm_providers/`.
- **Entry points**: `app.py` and `src/ensemble_chat/__main__.py` both launch the Gradio Blocks app.

The UI initiates actions and renders pure “events” from core; the core calls providers; providers return text and usage which the core persists and summarizes for UI.

### Directory map and roles

- `app.py`, `src/ensemble_chat/__main__.py`: Start the Gradio app by calling `ui.build_ui()`.
- `src/ensemble_chat/ui/`
  - `ui.py`: Creates initial `SessionState` from persisted settings, restores prior session if any, builds the base layout via `ui_layout.build_base_layout`, injects `ui_handlers.wire_events`.
  - `ui_layout.py`: Declares all Gradio components and tabs (“Chat”, model tabs, “Attachments”, “Resubmissions”, “Settings”). Returns both the Blocks instance and references to the widgets.
  - `ui_handlers.py`: Wires UI events to core logic. Key responsibilities:
    - Handles PDF selection, settings changes, and “New Chat”.
    - On button click, prepares history (`core.interactions.prepare_user_and_state`) and triggers an async processing generator (`core.orchestrator.run_from_label`).
    - Streams logic events to the UI via `ui_render.render_event` and maintains active-button visual state via small JS helpers.
  - `ui_render.py`: Converts core events into Gradio updates. Enforces separation: UI renders what core emits; it does not run logic.
  - `ui_constants.py`, `frontend_css.py`, `frontend_js.py`: Static assets and small JS/CSS helpers for scroll, tab switching, notifications, and active-button highlighting.

- `src/ensemble_chat/core/`
  - `orchestrator.py`: Single source of truth for starting runs from a button label. Resolves run mode (`single` vs `multi`), emits lifecycle events (`RunStartedEvent`, `RunCompletedEvent`), and forwards events from `engine.py`. Tracks the current asyncio task for cooperative cancellation and persists session snapshots between events.
  - `engine.py`: Implements execution flows as async generators that yield UI-friendly events:
    - `run_single(model_label, last_user, state)`: Calls a single proposer (`proposer.call_proposer`), updates official chat history, per-model history, saves transcript, and yields `StatusEvent` then `FinalEvent`.
    - `run_multi(models, aggregator_label, last_user, state)`: Fan-outs proposer calls concurrently, collects proposals, runs iterative aggregation via `aggregator.call_aggregator`, optionally triggers proposer synthesis rounds, updates histories and resubmissions log, saves transcript, and yields `StatusEvent`s and a final `FinalEvent`.
  - `proposer.py`: Builds proposer messages (system + chat history + user), handles retries/backoff, records raw model responses for debugging, and returns text plus token usage to cost tracker. Also exposes `call_synthesis` for re-synthesis prompts with aggregator notes and proposals packet.
  - `aggregator.py`: Constructs aggregator messages (system + chat history + user prompt), formats proposal packets, parses the aggregator’s first line to decide "FINAL" vs "REQUEST SYNTHESIS FROM PROPOSERS", and calls the chosen provider to obtain the aggregator output.
  - `events.py`: Typed event classes (`StatusEvent`, `FinalEvent`, `ErrorEvent`, lifecycle events). Also includes `event_from_raw` and `get_event_type` to normalize and inspect events.
  - `run_modes.py`: Maps UI button labels to run modes and proposer sets.
  - `run_state.py`: Ephemeral run state machine (IDLE/RUNNING/CANCELLED/COMPLETED) and metadata (`RunMetadata`: current button, models, aggregator, iteration). Used by orchestrator and UI selectors.
  - `runtime.py`: Central cooperative cancellation—bumps a session `_run_id` and cancels any live asyncio tasks.
  - `session_state.py`: Persistent session container: `pdf_path`, `chat_history`, per-model histories (for tabs), resubmissions history, cost tracker, selections and flags (temperature, notifications), and ephemeral runtime fields. Also handles save/load of `Session.json`.
  - `session_actions.py`: Utilities to set the PDF, apply selected provider models, and reset the session to defaults from `settings_manager`.
  - `settings_manager.py`: Loads/validates/saves app-wide settings in `Configurations/Settings.json`. Exposes singleton `APP_SETTINGS`.
  - `settings_actions.py`: UI-facing mutations that update both `SessionState` and `APP_SETTINGS`, and propagate model selection into provider adapters.
  - `selectors.py`: Pure read adapters to derive UI-facing values (cost lines, model tab displays, resubmissions display, notification flag, active button element id). These keep UI rendering logic simple and testable.
  - `history.py`: `ChatHistory` class for official chat history (only user + final replies). Also renders history as Gradio chat pairs.
  - `conversation.py`: Legacy single-run helper (calls proposer directly) used in some flows; core engine supersedes it for the main UI.
  - `interactions.py`: High-level helpers for preparing chat state on user actions (new query vs redo), resolving the last user message, and budget guard checks.
  - `prompts.py`: Reads prompt files and performs placeholder substitution. Provides `proposer_system`, `proposer_synthesis_prompt`, `aggregator_system`, `aggregator_user`, and `aggregator_force_user`.
  - `paths.py`: Locates project root so prompts, configurations, and session files can be read consistently.
  - `utils.py`: `CostTracker` with per-model totals and a budget cap, transcript saving to `Chats/`, human-friendly error mapping, raw-response logging (`RawProposerLogs/`), and timestamp helpers.
  - `model_configs.py`: Loads model lists for dropdowns from `Configurations/*.json` with fallbacks.
  - `sanitization.py`: Escapes potentially dangerous angle brackets and unwraps inline-math inside code spans for safe display.

- `src/ensemble_chat/llm_providers/`
  - `base.py`: Provider interface and a `ModuleProvider` adapter that wraps module-level functions (`call`, `set_model`).
  - `registry.py`: Registers and retrieves providers; normalizes labels ("ChatGPT", "Claude", "Gemini").
  - `shared.py`: Provider-agnostic utilities: retry helper, `LLMError`, logging of messages, and a 180s timeout constant.
  - `openai_provider.py`: OpenAI Responses API integration: builds `input` blocks, attaches web search tool; attaches vector-store file search when a PDF is selected; supports temperature when available.
  - `anthropic_provider.py`: Anthropic Messages API with Files API for PDFs, supports "thinking" mode; serializes raw responses for logs.
  - `gemini_provider.py`: Google GenAI client with `files.upload` for PDFs and Google Search grounding tool; maps dialog history to Gemini types.

### Key classes and responsibilities

- `SessionState` (core.session_state): In-memory session data, plus persistence to `Session.json`.
- `ChatHistory` (core.history): Manages official conversation (user + final replies only).
- `CostTracker` (core.utils): Tracks per-model spend, enforces a budget guard via `interactions.budget_guard_message` in UI.
- `RunMetadata`/`RunPhase` (core.run_state): Ephemeral state for current run; validators for active button and UI highlighting.
- Event classes (core.events): Strongly-typed communication channel from core to UI; UI rendering never inspects core internals directly.
- Provider adapters (llm_providers.*): Each exposes `set_model(model_id)` and async `call(messages, pdf_path, temperature=...)` returning `(text, prompt_tokens, completion_tokens, raw_text)`.

### Execution flow

1) User clicks a button in the Chat tab (e.g., "All").
   - UI handler calls `interactions.prepare_user_and_state` to modify `ChatHistory` for new input or redo and clears the input box.
   - A generator handler begins: `orchestrator.run_from_label(button_label, last_user, state)`.

2) Orchestration determines run mode.
   - `run_modes.resolve_run_mode` maps the label to either a single model or a list of proposers.
   - `run_state.start_single` or `start_multi` sets ephemeral metadata; `orchestrator` yields `RunStartedEvent`.

3) Engine drives the run and emits events.
   - Single mode: `engine.run_single` yields a waiting status, calls `proposer.call_proposer`, updates official and per-model histories, saves transcript, then yields a `FinalEvent`.
   - Multi mode: `engine.run_multi` yields dispatch/collecting statuses, fans out proposer calls concurrently, appends proposals to per-model tabs, then iterates aggregation using `aggregator.call_aggregator`:
     - If aggregator’s first non-empty line contains "FINAL": commit final reply to `ChatHistory`, save transcript, yield `FinalEvent`.
     - If line contains "REQUEST SYNTHESIS FROM PROPOSERS" (and iteration < 5): build proposals packet, log the full prompt and aggregator notes in `resubmissions_history`, call `proposer.call_synthesis` per model, then continue next aggregation iteration.
     - On 5th iteration, `prompts.aggregator_force_user()` is used to force a final.

4) UI rendering loop.
   - `ui_handlers` consumes the async event stream; for each event it calls `ui_render.render_event`, which updates `chat`, per-model tabs, resubmissions tab, status banner, and notification flag.
   - Active button highlighting is handled via `selectors.active_button_elem_id` and a small JS snippet.

5) Cancellation and redo.
   - Any button click triggers `runtime.cancel_inflight`, which cancels the current asyncio task and bumps `state._run_id`. Engine checks `_run_id` between awaits and exits early when invalidated. Redo leaves last user message active and discards the last assistant reply.

### UI vs logic separation

- UI never calls providers directly. All model calls live in core via `engine` → `proposer`/`aggregator` → `llm_providers`.
- UI mutations go through `session_actions.py` and `settings_actions.py` to ensure persistence and provider model application.
- UI only renders what `events.py` emits via `ui_render.py` and derives read-only projections via `selectors.py`.

### Where to change things

- **Add a new provider**: implement `llm_providers/<name>_provider.py` with `set_model` and async `call`; register it in `llm_providers/registry.py`; update `ui_constants.BUTTONS` and `core/run_modes.py` if exposing new buttons; extend `Configurations/<Provider>.json` and `core/model_configs.py` for dropdowns.
- **Change prompts**: edit files in `Prompts/`; placeholder replacement is centralized in `core/prompts.py`.
- **Tweak aggregation logic**: update `core/aggregator.py` parsing/packing or `engine.run_multi` loop.
- **Adjust cost/budget**: change constants in `core/utils.py` and guards in `core/interactions.py`.
- **Session persistence shape**: `core/session_state.py` and `core/settings_manager.py`.
- **UI behavior or layout**: `ui_layout.py` and `ui_handlers.py`; rendering rules in `ui_render.py`; visuals in `frontend_css.py` and `frontend_js.py`.

### Data and persistence

- Official chat transcripts: `Chats/<chat_id>.md` via `utils.save_chat` (only user + final assistant).
- Raw proposer responses: `RawProposerLogs/<proposer>.txt` via `utils.write_last_raw_response`.
- Last session: `Session.json` (auto-loaded on startup by `ui.build_ui`).
- App settings: `Configurations/Settings.json` (auto-normalized).

### Notable conventions

- No streaming; all provider calls return complete text chunks.
- One PDF per session, attached using each provider’s native API when present.
- Aggregator anonymity: proposals do not reveal authoring model identities.
- Iteration cap: force final on the 5th aggregator pass.
- Cooperative cancellation: `_run_id` invalidation and `asyncio.Task.cancel()` ensure immediate stop and clean UI state.

### Minimal call graph (essentials)

```text
UI button click → interactions.prepare_user_and_state → orchestrator.run_from_label
  single → engine.run_single → proposer.call_proposer → llm_providers.call
  multi  → engine.run_multi → proposer.call_proposer (fan-out) → aggregator.call_aggregator → [optional] proposer.call_synthesis → aggregator.call_aggregator …
→ engine updates ChatHistory/Model tabs/Resubmissions → utils.save_chat → events.FinalEvent → ui_render.render_event
```

### Quick pointers for LLM agents

- To change model dropdown values or defaults: `Configurations/*.json`, `core/model_configs.py`, `core/settings_manager.py`.
- To add a new button combining models: `ui_constants.BUTTONS`, `ui_constants.MULTI_BUTTON_MODELS`, `core/run_modes.py`.
- To adjust statuses or text shown during runs: `core/engine.py` (StatusEvent texts) and `ui_render.py`.
- To change notification behavior: `selectors.notifications_flag`, `ui/frontend_js.py` (JS_NOTIFY_IF_FLAG), Settings checkbox in `ui_layout.py` + `ui_handlers.py`.


