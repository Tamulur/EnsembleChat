import os
import json
import time
from datetime import datetime
from typing import List, Dict

PRICES_PER_1K_TOKENS = {
    "o3": 0.005,         # $/1k tokens – dummy values, replace with real
    "claude": 0.008,
    "gemini": 0.006,
}

# Very rough character->token conversion multiplier if API usage info missing
CHARS_PER_TOKEN = 4

BUDGET_LIMIT = 5.0  # $5 per session


class CostTracker:
    """Tracks and estimates token usage cost during a chat session."""

    def __init__(self):
        self.total_cost = 0.0
        self.debug_info: List[Dict] = []  # for console logging only

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    def add_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        total_tokens = prompt_tokens + completion_tokens
        price_per_1k = PRICES_PER_1K_TOKENS.get(model.lower())
        if price_per_1k is None:
            price_per_1k = 0.005  # fallback dummy
        cost = (total_tokens / 1000) * price_per_1k
        self.total_cost += cost
        self.debug_info.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "running_total": self.total_cost,
            }
        )
        print("[COST] Model %s | prompt %d | completion %d | cost %.4f | total %.4f"
              % (model, prompt_tokens, completion_tokens, cost, self.total_cost))

    def will_exceed_budget(self, estimated_cost: float) -> bool:
        return (self.total_cost + estimated_cost) > BUDGET_LIMIT

    def summary(self):
        return {
            "total_cost": self.total_cost,
            "details": self.debug_info,
        }


def ensure_chats_dir():
    if not os.path.exists("Chats"):
        os.makedirs("Chats", exist_ok=True)


def save_chat(chat_id: str, history: List[Dict], pdf_path: str | None = None):
    """Save chat transcript to Chats directory.

    Filename format: <pdf_title>_<timestamp>.json when pdf_path provided,
    otherwise <timestamp>.json.
    """
    ensure_chats_dir()
    if pdf_path:
        title = os.path.splitext(os.path.basename(pdf_path))[0]
        filename = f"{title}_{chat_id}.json"
    else:
        filename = f"{chat_id}.json"
    path = os.path.join("Chats", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def timestamp_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
