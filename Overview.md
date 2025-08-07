# Overview\.md

## Goal

A local Gradio chat app for **personal use** that can, per user turn, send the same question (plus chat history and the selected PDF) to multiple LLMs (“proposers”), then pass their proposals to an **aggregator** LLM that either (a) issues the final reply or (b) requests a re-synthesis round from the proposers. Final replies (only) are added to the official chat history.

* **Proposer models:** OpenAI o3, Claude Sonnet 4, Gemini 2.5 Pro (fixed set).
* **Aggregator model:** Claude Sonnet 4 (fixed).
* **One PDF per session**, always attached to every model call using each provider’s native PDF/file integration. No local parsing/OCR; assume normal-length arXiv-style PDFs.

## Frontend (Gradio)

* At top: **Select PDF** button (store path for session; do **not** copy or persist the file).
* Chat area shows **user messages** and **final AI replies** (only).
* Under input: fixed action buttons:

  1. **o3**
  2. **Claude**
  3. **Gemini**
  4. **o3 & Claude**
  5. **All** (o3 + Claude + Gemini)
     These map to either single-LLM mode (no aggregation) or proposer+aggregator mode.

  * **Redo last reply:** If the most recent LLM response is unsatisfactory, leave the input box empty and click any action button. The application will
    1. Remove that last AI reply from the visible chat (and persistence).
    2. Re-process the **same user input as before** according to the button pressed.  
       *Example:* clicking **All** resends the prior user input to all proposers and aggregates their new replies.

### UI status messages (non-stream)

* “Sending requests for proposals…” when dispatching proposer calls.
* “Collecting replies…” while waiting for proposers.
* “Aggregating replies, iteration *N*…” during each aggregator pass.

### Streaming

* **Stream tokens** when we know a final will be produced:

  * Single-LLM buttons (o3 / Claude / Gemini).
  * Aggregator **forced final** pass (on the 5th iteration for that user turn).
* Otherwise, aggregator output is fetched non-streaming and then displayed.

## Data/State

* **Official chat history** (persisted): only **user inputs** and **final replies** (either from a single LLM call or from the aggregator). Proposals and aggregator deliberations are not stored.
* Save chats under `Chats/` (one file per session, e.g., timestamped JSON). The app only remembers **the selected PDF path** during the active session.

## Model Calls

### Common

* Attach the **same PDF** to every request using each provider’s file APIs. Do not check sizes; assume typical research PDFs.
* For models that support it, set the model to use web search automatically. For example, for OpenAI, use the Responses API to let it automatically use the web search tool.
* Temperature: **0.7** for all calls. Other parameters at provider defaults.
* **Timeout per request:** 120 seconds.
* **Retries (proposers only):** up to 5 on error (exponential backoff is fine). If still failing, **proceed with remaining proposals**.

### Single-LLM mode (buttons: o3 / Claude / Gemini)

* Send: model’s **system prompt** (from `ProposerSystemPrompts/<Model>.txt`), the **PDF**, full **chat history** (final replies only), and the **new user input** as the user message.
* Display the model’s reply **streaming** and **add to history** as the final reply.

### Multi-LLM mode (buttons: o3 & Claude / All)

1. **Proposer phase**

   * For each selected proposer, send: its system prompt (`ProposerSystemPrompts/<Model>.txt`), the **PDF**, the **chat history** (final replies only), and the **new user input** as user message.
   * Collect replies (do **not** show). Status: “Sending requests for proposals…”, then “Collecting replies…”. 
2. **Aggregator phase** (iteration count starts at 1)

   * Send to aggregator:

     * `AggregatorSystemPrompt.txt` (system).
     * The **PDF**.
     * **Chat history** (final replies only) including the **latest user input**.
     * `AggregatorUserPrompt.txt` as user message, followed by the proposals packet:

       ```
       # Proposed Reply 1:
       <text>

       # Proposed Reply 2:
       <text>
       ...
       ```

       **Do not** reveal which model wrote which proposal.

   * Status: “Aggregating replies, iteration N…”.

   * **Aggregator control via first line (case-insensitive):**

     * If the first non-empty line contains **“FINAL”** (e.g., `FINAL REPLY:`), treat everything after that line as the **final reply**. Show it (non-stream unless this is the forced final), and **add to history**.
     * If the first non-empty line contains **“REQUEST SYNTHESIS FROM PROPOSERS”**, trigger a **re-synthesis round**:

       * For each proposer, send:

         * Its system prompt (`ProposerSystemPrompts/<Model>.txt`)
         * The **PDF**
         * The **chat history up to and including the last user input**
         * Its **synthesis prompt** from `SynthesizeFromProposalsPrompts/<Model>.txt` as user message, **appending the aggregator’s notes** (everything after the sentinel line).
       * Collect the new proposals and go back to **Aggregator phase** with iteration `N+1`. Status: “Aggregating replies, iteration N+1…”.

   * **Iteration cap & forced final:** On the **5th** aggregator pass for the same user input, **replace** `AggregatorUserPrompt.txt` with `AggregatorForceReplyUserPrompt.txt` to force a final. Display **streaming**, then **add to history**.

## Prompts & Files

```
/ProposerSystemPrompts/
  o3.txt
  Claude.txt
  Gemini.txt

/SynthesizeFromProposalsPrompts/
  o3.txt
  Claude.txt
  Gemini.txt

AggregatorSystemPrompt.txt
AggregatorUserPrompt.txt
AggregatorForceReplyUserPrompt.txt
```

* All prompt files are **plain text** (no JSON schemas).
* Proposer prompts: normal role behavior for answering based on PDF + chat context.
* Synthesis prompts: instruct the proposer to revise/defend its answer given **all** proposals and the aggregator’s remarks.
* Aggregator prompts: instruct to either **FINAL REPLY** or **REQUEST SYNTHESIS FROM PROPOSERS**, starting the output with the chosen sentinel line on the **first line**.

## History & Anonymity Rules

* **Only final replies** go into history; proposals and aggregator notes are not stored or shown.
* When constructing proposal packets for the aggregator (and re-synthesis prompts for proposers), **do not** include model identities—only “Proposed Reply 1/2/…”.
* Later turns can safely include past final replies; they are anonymous with respect to whether they came from a single model or aggregation.

## Budget Guard

* Maintain a running **session cost estimate** (simple token\*price approximation per model call using hardcoded per-model prices). If the estimate would exceed **\$5** for the current session, **stop** and display a brief message offering to start a new session or change the button selection.

## Errors & Retries

* **Proposers:** retry up to **5** times on transport/5xx/timeouts; otherwise skip and continue with remaining proposals.
* **Aggregator:** on error/timeout, retry once; if still failing, show a concise error and keep the turn open for user re-send.
* Surface a concise, user-facing error string; log full details to console.

## Minimal Logging

* Write chat transcripts (user inputs + final replies) to `Chats/<session_id>.json`.
* Track per-turn timing (proposers total, aggregator per iteration) and the cost estimate counters.
