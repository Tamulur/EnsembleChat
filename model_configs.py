import json
from pathlib import Path
from typing import List, Dict


# --- Configuration loading for model dropdowns ---
CONFIG_DIR = Path(__file__).parent / "Configurations"


DEFAULT_MODELS: Dict[str, List[str]] = {
    "OpenAI": ["GPT-5", "GPT-5-mini", "o3", "GPT-4.1"],
    "Claude": ["claude-sonnet-4-0"],
    "Gemini": ["gemini-2.5-pro"],
}


def _read_models_from_file(filename: str, fallback: List[str]) -> List[str]:
    cfg_path = CONFIG_DIR / filename
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models")
        if isinstance(models, list) and all(isinstance(x, str) for x in models) and models:
            return models
        else:
            print(f"[WARN] '{filename}' missing valid 'models' list. Using defaults.")
    except Exception as e:
        print(f"[WARN] Failed to read '{filename}': {e}. Using defaults.")
    return fallback


def load_model_configurations() -> Dict[str, List[str]]:
    return {
        "OpenAI": _read_models_from_file("OpenAI.json", DEFAULT_MODELS["OpenAI"]),
        "Claude": _read_models_from_file("Claude.json", DEFAULT_MODELS["Claude"]),
        "Gemini": _read_models_from_file("Gemini.json", DEFAULT_MODELS["Gemini"]),
    }


MODEL_CONFIGS = load_model_configurations()


