# Multi-LLM PDF Chat (Gradio)

Chat with any PDF, querying LLMs individually or let them work out a response among themselves.

<img width="1875" height="756" alt="Screenshot" src="https://github.com/user-attachments/assets/9f888619-a6b1-4636-9098-74e14735adbd" />

_Note: this is my experiment in vibe-coding. Don't expect everything to be air-tight._

## ✨ Features
- **Single or multi-model replies, decide per reply** – choose OpenAI ChatGPT, Claude Sonnet 4, Gemini 2.5 Pro, or an automatic aggregation of all three.
- **One-click controls** – five fixed buttons (ChatGPT · Claude · Gemini · ChatGPT + Claude · All)
- **Quick Redo** - Leave input field empty and click a button to redo the last response
- **Aggregator logic** – Claude Sonnet 4 coordinates proposer replies and may iterate up to five times before producing a final answer.
- **Cost & timeout guardrails** – 120 s per request, exponential-back-off retries, and a \$5 session budget cap.
- **Lean persistence** – only user inputs & final replies are stored (`Chats/`).
- **Individual model tabs** – separate tabs for ChatGPT, Claude, and Gemini show their last outputs, including proposals that don't appear in the main chat.
- **Notification** - shows a system notification once the final answer is ready.


## 🚀 Quick Start
```bash
git clone https://github.com/Tamulur/EnsembleChat.git
cd <your-repo>
pip install -r requirements.txt          # or poetry install
export OPENAI_API_KEY=...                # plus ANTHROPIC_API_KEY, GOOGLE_API_KEY
python app.py
```

Open the Gradio URL, click Select PDF, choose a model button, and start chatting.
