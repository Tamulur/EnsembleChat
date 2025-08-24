import json
from pathlib import Path
from typing import Dict, Any

from ensemble_chat.core.model_configs import MODEL_CONFIGS
from ensemble_chat.core.paths import project_root


# --- Settings persistence (Configurations/Settings.json) ---
SETTINGS_FILE = project_root() / "Configurations" / "Settings.json"


def _default_settings() -> Dict[str, Any]:
    return {
        "openai_model": MODEL_CONFIGS["OpenAI"][0],
        "claude_model": MODEL_CONFIGS["Claude"][0],
        "gemini_model": MODEL_CONFIGS["Gemini"][0],
        "aggregator": "Claude",
        "temperature": 0.7,
        "notifications": True,
    }


def _validate_and_merge_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_settings()
    merged = dict(defaults)

    try:
        if isinstance(raw.get("openai_model"), str) and raw["openai_model"] in MODEL_CONFIGS["OpenAI"]:
            merged["openai_model"] = raw["openai_model"]
        if isinstance(raw.get("claude_model"), str) and raw["claude_model"] in MODEL_CONFIGS["Claude"]:
            merged["claude_model"] = raw["claude_model"]
        if isinstance(raw.get("gemini_model"), str) and raw["gemini_model"] in MODEL_CONFIGS["Gemini"]:
            merged["gemini_model"] = raw["gemini_model"]
        if isinstance(raw.get("aggregator"), str) and raw["aggregator"] in ["ChatGPT", "Claude", "Gemini"]:
            merged["aggregator"] = raw["aggregator"]
        temp = raw.get("temperature")
        if isinstance(temp, (int, float)):
            merged["temperature"] = max(0.0, min(1.0, float(temp)))
        if isinstance(raw.get("notifications"), bool):
            merged["notifications"] = raw["notifications"]
    except Exception as e:
        print("[WARN] Settings validation error:", e)

    return merged


def load_settings() -> Dict[str, Any]:
    defaults = _default_settings()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            settings = _validate_and_merge_settings(raw if isinstance(raw, dict) else {})
        except Exception as e:
            print(f"[WARN] Failed to read Settings.json: {e}. Using defaults.")
            settings = defaults
    else:
        settings = defaults
    # Ensure file exists and is normalized
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write Settings.json: {e}")
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    normalized = _validate_and_merge_settings(settings if isinstance(settings, dict) else {})
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save Settings.json: {e}")


APP_SETTINGS = load_settings()


