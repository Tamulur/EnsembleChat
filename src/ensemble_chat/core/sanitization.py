import html
import re
from typing import List, Tuple


_ZWSP = "\u200B"


_INLINE_MATH_CODE_PATTERNS = [
    # ` $ ... $ ` → $ ... $
    re.compile(r"`\s*(\$(?:\\.|[^$])+\$)\s*`"),
    # ` \( ... \) ` → \( ... \)
    re.compile(r"`\s*(\\\([^`]+?\\\))\s*`"),
    # ` \[ ... \] ` → \[ ... \]
    re.compile(r"`\s*(\\\[[^`]+?\\\])\s*`", re.DOTALL),
]


def _unwrap_inline_math_in_code_spans(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for pat in _INLINE_MATH_CODE_PATTERNS:
        # Apply repeatedly in case of multiple occurrences
        prev = None
        while prev != out:
            prev = out
            out = pat.sub(r"\1", out)
    return out


def neutralize_angle_brackets(text: str) -> str:
    if not isinstance(text, str):
        return text
    # First, decode entities and unwrap inline math mistakenly wrapped in code spans
    decoded = html.unescape(text)
    decoded = _unwrap_inline_math_in_code_spans(decoded)
    # Then, neutralize potential HTML-like tags while preserving visible brackets
    decoded = re.sub(r"<(?=[A-Za-z/])", "<" + _ZWSP, decoded)
    decoded = re.sub(r"(?<=[A-Za-z0-9/])>", _ZWSP + ">", decoded)
    return decoded


def sanitize_pairs_for_display(pairs: List[Tuple[str, str]]):
    sanitized = []
    for left, right in pairs:
        left_s = neutralize_angle_brackets(left)
        right_s = neutralize_angle_brackets(right)
        sanitized.append((left_s, right_s))
    return sanitized


__all__ = [
    "neutralize_angle_brackets",
    "sanitize_pairs_for_display",
]


