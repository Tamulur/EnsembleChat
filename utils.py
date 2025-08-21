import os
import time
from datetime import datetime
from typing import List, Dict

PRICES_PER_1K_TOKENS = {
    # Prices per 1k tokens; separate input vs output to allow asymmetric pricing
    "chatgpt": {"input": 0.002, "output": 0.008},
    "claude": {"input": 0.003, "output": 0.015},
    "gemini": {"input": 0.00125, "output": 0.01},
}

# Very rough character->token conversion multiplier if API usage info missing
CHARS_PER_TOKEN = 4

BUDGET_LIMIT = 5.0  # $5 per session


class CostTracker:
    """Tracks and estimates token usage cost during a chat session."""

    def __init__(self):
        self.total_cost = 0.0
        # Track cumulative spend per model (keys are lowercased model labels)
        self.model_costs = {"chatgpt": 0.0, "claude": 0.0, "gemini": 0.0}
        self.debug_info: List[Dict] = []  # for console logging only

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    def add_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        model_key = model.lower()
        pricing = PRICES_PER_1K_TOKENS.get(model_key)
        if pricing is None or not isinstance(pricing, dict):
            pricing = {"input": 0.005, "output": 0.005}  # fallback dummy

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        cost = input_cost + output_cost

        self.total_cost += cost
        # Update per-model totals
        self.model_costs[model_key] = self.model_costs.get(model_key, 0.0) + cost
        self.debug_info.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "input_price_per_1k": pricing["input"],
                "output_price_per_1k": pricing["output"],
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cost": cost,
                "running_total": self.total_cost,
            }
        )
        print(
            "[COST] Model %s | prompt %d | completion %d | cost %.4f$ | total %.4f$"
            % (model, prompt_tokens, completion_tokens, cost, self.total_cost)
        )

    def will_exceed_budget(self, estimated_cost: float) -> bool:
        return (self.total_cost + estimated_cost) > BUDGET_LIMIT

    def summary(self):
        return {
            "total_cost": self.total_cost,
            "details": self.debug_info,
        }

    def get_model_cost(self, model: str) -> float:
        """Return cumulative cost for the given model label (case-insensitive)."""
        return float(self.model_costs.get(model.lower(), 0.0))


def ensure_chats_dir():
    if not os.path.exists("Chats"):
        os.makedirs("Chats", exist_ok=True)


def save_chat(chat_id: str, history: List[Dict], pdf_path: str | None = None):
    """Save chat transcript to Chats directory.

    Filename format: <pdf_title>_<timestamp>.md when pdf_path provided,
    otherwise <timestamp>.md.
    """
    ensure_chats_dir()
    if pdf_path:
        title = os.path.splitext(os.path.basename(pdf_path))[0]
        filename = f"{title}_{chat_id}.md"
    else:
        filename = f"{chat_id}.md"
    path = os.path.join("Chats", filename)
    with open(path, "w", encoding="utf-8") as f:
        prev_role = None
        for entry in history:
            role = entry["role"]
            text = entry["text"]
            
            # Add horizontal separator when speaker changes
            if prev_role is not None and prev_role != role:
                f.write("---\n\n")
            
            if role == "user":
                f.write(f"## 🧑 User:\n{text}\n\n")
            elif role == "assistant":
                f.write(f"## 🤖 Assistant:\n{text}\n\n")
            
            prev_role = role


def timestamp_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
