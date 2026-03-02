from __future__ import annotations
from typing import Optional

import re


def truncate_text(s: str, max_chars: Optional[int]) -> str:
    if max_chars is None:
        return s
    elif len(s) <= max_chars:
        return s
    else:
        return s[: max_chars - 3] + "..."


def any_regex_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)