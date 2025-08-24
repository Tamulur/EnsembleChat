import html
import re
from typing import List, Tuple


_ZWSP = "\u200B"


def neutralize_angle_brackets(text: str) -> str:
    if not isinstance(text, str):
        return text
    decoded = html.unescape(text)
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


