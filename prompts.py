import os
from functools import lru_cache

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

@lru_cache(maxsize=32)
def _read(path: str) -> str:
    full = os.path.join(ROOT_DIR, path)
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def proposer_system(model: str) -> str:
    return _read(os.path.join("ProposerSystemPrompts", f"{model}.txt"))


def proposer_synthesis_prompt(model: str) -> str:
    return _read(os.path.join("SynthesizeFromProposalsPrompts", f"{model}.txt"))


def aggregator_system() -> str:
    return _read("AggregatorSystemPrompt.txt")


def aggregator_user() -> str:
    return _read("AggregatorUserPrompt.txt")


def aggregator_force_user() -> str:
    return _read("AggregatorForceReplyUserPrompt.txt")
