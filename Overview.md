# Overview\.md

## Goal

A local Gradio chat app for **personal use** that can, per user turn, send the same question (plus chat history and the selected PDF) to multiple LLMs (“proposers”), then pass their proposals to an **aggregator** LLM that either (a) issues the final reply or (b) requests a re-synthesis round from the proposers. Final replies (only) are added to the official chat history.

* **Proposer models:** ChatGPT, Claude, Gemini.
* **Aggregator model:** Either ChatGPT, or Claude, or Gemini.
* **One PDF per session**, always attached to every model call using each provider’s native PDF/file integration. No local parsing/OCR; assume normal-length arXiv-style PDFs.

## Frontend (Gradio)

Seven tabs: Chat, ChatGPT, Claude, Gemini, Attachments, Resubmissions, and Settings.

### Main tab: Chat
* Chat area shows **user messages** and **final AI replies** (only).
* Under input: fixed action buttons:

  1. **ChatGPT**
  2. **Claude**
  3. **Gemini**
  4. **ChatGPT & Gemini**
  5. **All** (ChatGPT + Claude + Gemini)
     These map to either single-LLM mode (no aggregation) or proposer+aggregator mode.

  * **Redo last reply:** If the most recent LLM response is unsatisfactory, leave the input box empty and click any action button. The application will
    1. Remove that last AI reply from the visible chat (and persistence).
    2. Re-process the **same user input as before** according to the button pressed.  
       *Example:* clicking **All** resends the prior user input to all proposers and aggregates their new replies.

### Model tabs: last outputs of each LLM
The next three tabs show the last output of each LLM that is produced when directly queried for an answer (with the ChatGPT, Claude or Gemini button), or when asked for a proposal (with the All button for example). The user query that was used to generate that LLM output precedes the output in the tab's chat window. The Claude tab contains only the output that was produced in these cases, it does not show the output that Claude produced in the aggregator role (if Claude is set to be the aggregator).

### Tab Attachments
In this tab the user can select the pdf with a **Select PDF** button (store path for session; do **not** copy or persist the file).

### Resubmissions Tab
The Resubmissions tab shows a chat window with a history of resubmission requests that the aggregator LLM sent so far. Every time the aggregator LLM decides that because something is off, it needs to ask the other LLMs for another round of proposals (so for every iteration after the first) by sending """REQUEST SYNTHESIS FROM PROPOSERS""", the user prompt it sends (containing the proposals from the previous iteration and the remarks for each) is logged as a separate message entry into this tab's chat window.

### Settings Tab
In the Settings tab the user can select which model to use for each provider: For each provider, there is a dropdown with possible models. The model selected here will be used for the respective provider. Which models are available in the dropdowns for each provider is determined in the configuration files. In the folder "Configurations", there is a file for each provider: "OpenAI.json", "Claude.json" and "Gemini.json". Each configuration JSON file contains a field "models" that has a list of model names for that provider. For OpenAI the list should be "GPT-5", "GPT-5-mini", "o3", and "GPT-4.1". For Claude the list should be "claude-sonnet-4-0". For Gemini the list should be "gemini-2.5-pro". Currently the lists for Claude and Gemini only have one possible option each. Another dropdown labelled "Aggregator" lets the user select one of the providers (ChatGPT, Claude, or Gemini) as the aggregator.
There is a field to set the temperature that should be used for any models whose API allows this, default 0.7.
There is a "Notifications" checkbox, that sets whether to show system notifications when the final reply has finished. Default it on.
All settings are stored in a file Settings.json. If that file doesn't exist, it is created. At start, it is read and the settings applied. When the user changes a setting, the file is updated.


### UI status messages

* “Sending requests for proposals…” when dispatching proposer calls.
* “Collecting replies…” while waiting for proposers.
* “Aggregating replies, iteration *N*…” during each aggregator pass.

### Notifications
* Show a system notification "Reply complete" when the final reply is finished and if the "Notifications" checkbox in the Settings tab is checked.

## No Streaming

* Never stream the output, just let it generate in one go.

## Data/State

* **Official chat history** (persisted): only **user inputs** and **final replies** (either from a single LLM call or from the aggregator). Proposals and aggregator deliberations are not stored.
* Save chats under `Chats/` (one file per session, timestamped filenames). The app only remembers **the selected PDF path** during the active session. Save chats in Markdown format.

## Model Calls

### Common

* Attach the **same PDF** to every request using each provider’s file APIs. Do not check sizes; assume typical research PDFs.
* For models that support it, set the model to use web search automatically. For example, for OpenAI, use the Responses API to let it automatically use the web search tool.
* Temperature: **0.7** for all calls. Other parameters at provider defaults.
* **Timeout per request:** 180 seconds.
* **Retries (proposers only):** up to 5 on error (exponential backoff is fine). If still failing, **proceed with remaining proposals**.

### Single-LLM mode (buttons: ChatGPT / Claude / Gemini)

* Send: model’s **system prompt** (from `Prompts/ProposerSystemPrompts/<Model>.txt`), the **PDF**, full **chat history** (final replies only), and the **new user input** as the user message.
* Display the model's reply and **add to history** as the final reply.

### Multi-LLM mode (buttons: ChatGPT & Gemini / All)

1. **Proposer phase**

   * For each selected proposer, send: its system prompt (`Prompts/ProposerSystemPrompts/<Model>.txt`), the **PDF**, the **chat history** (final replies only), and the **new user input** as user message.
   * Collect replies (do **not** show). Status: “Sending requests for proposals…”, then “Collecting replies…”. 
2. **Aggregator phase** (iteration count starts at 1)

   * Send to aggregator:

     * `Prompts/AggregatorSystemPrompt.txt` (system).
     * The **PDF**.
     * **Chat history** (final replies only) including the **latest user input**.
     * `Prompts/AggregatorUserPrompt.txt` as user message, followed by the proposals packet:

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

     * If the first non-empty line contains **"FINAL"** (e.g., `FINAL REPLY:`), treat everything after that line as the **final reply**. Show it and **add to history**.
     * If the first non-empty line contains **“REQUEST SYNTHESIS FROM PROPOSERS”**, trigger a **re-synthesis round**:

       * For each proposer, send:

         * Its system prompt (`Prompts/ProposerSystemPrompts/<Model>.txt`)
         * The **PDF**
         * The **chat history up to and including the last user input**
         * Its **synthesis prompt** from `Prompts/SynthesizeFromProposalsPrompts/<Model>.txt` as user message, **appending the aggregator’s notes** (everything after the sentinel line).
       * Collect the new proposals and go back to **Aggregator phase** with iteration `N+1`. Status: “Aggregating replies, iteration N+1…”.

   * **Iteration cap & forced final:** On the **5th** aggregator pass for the same user input, **replace** `Prompts/AggregatorUserPrompt.txt` with `Prompts/AggregatorForceReplyUserPrompt.txt` to force a final. Display the result, then **add to history**.

## Prompts & Files

```
Prompts/
  ProposerSystemPrompts/
    ChatGPT.txt
    Claude.txt
    Gemini.txt

  SynthesizeFromProposalsPrompts/
    ChatGPT.txt
    Claude.txt
    Gemini.txt

  AggregatorSystemPrompt.txt
  AggregatorUserPrompt.txt
  AggregatorForceReplyUserPrompt.txt
  SynthesizePromptCommon.txt
  SystemPromptCommon.txt
  ExampleExplanations.txt
```

* All prompt files are **plain text** (no JSON schemas).
* Proposer prompts: normal role behavior for answering based on PDF + chat context.
* Synthesis prompts: instruct the proposer to revise/defend its answer given **all** proposals and the aggregator’s remarks.
* Aggregator prompts: instruct to either **FINAL REPLY** or **REQUEST SYNTHESIS FROM PROPOSERS**, starting the output with the chosen sentinel line on the **first line**.
* `Prompts/SynthesizePromptCommon.txt` should replace the string {SynthesizePromptCommon} in all user prompts.
* `Prompts/SystemPromptCommon.txt` should replace the string {SystemPromptCommon} in all system prompts (so in what is read from `Prompts/AggregatorSystemPrompt.txt` or from the proposer system prompts).
* `Prompts/ExampleExplanations.txt` contains examples of good explanations. All system prompts have a placeholder {examples} that should be replaced with the contents of this file.

## History & Anonymity Rules

* **Only final replies** go into history; proposals and aggregator notes are not stored or shown.
* When constructing proposal packets for the aggregator (and re-synthesis prompts for proposers), **do not** include model identities—only “Proposed Reply 1/2/…”.
* Later turns can safely include past final replies; they are anonymous with respect to whether they came from a single model or aggregation.

## Budget Guard

* Maintain a running **session cost estimate** (simple token\*price approximation per model call using hardcoded per-model prices). If the estimate would exceed **\$5** for the current session, **stop** and display a brief message offering to start a new session or change the button selection.

## Errors & Retries

* **Single LLM mode:** retry up to **5** times on transport/5xx/timeouts
* **Proposers:** retry up to **5** times on transport/5xx/timeouts; otherwise skip and continue with remaining proposals.
* **Aggregator:** on error/timeout, retry once, or three times on 529 (overloaded) messages; if still failing, show a concise error and keep the turn open for user re-send.
* Surface a concise, user-facing error string; log full details to console.

## Minimal Logging

* Write chat transcripts (user inputs + final replies) to `Chats/<session_id>.md`.
* Track per-turn timing (proposers total, aggregator per iteration) and the cost estimate counters.
