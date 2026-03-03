# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from domain_qa.core.postcheck.base import PostCheck
from domain_qa.core.postcheck.safety_distress import SafetyDistressPostCheck


def default_postchecks(enabled: bool = True) -> list[PostCheck]:
    if not enabled:
        return []
    
    # Ordered list: sets post-check priority (first match wins)
    return [
        SafetyDistressPostCheck()
    ]


# TO-DO:
# - add more post-checks
# - enable/disable checks in config
# - allow multiple triggers and choose best fallback