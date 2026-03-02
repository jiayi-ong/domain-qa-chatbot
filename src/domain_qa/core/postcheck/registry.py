from __future__ import annotations

from domain_qa.core.postcheck.base import PostCheck
from domain_qa.core.postcheck.regex_oos import RegexOutOfScopePostCheck
from domain_qa.core.postcheck.safety_distress import SafetyDistressPostCheck


def default_postchecks(enabled: bool = True) -> list[PostCheck]:
    if not enabled:
        return []
    # Ordered list = priority (first match wins) in Agent.
    return [
        SafetyDistressPostCheck(),
        RegexOutOfScopePostCheck(),
    ]


# TO-DO:
# - config-driven enable/disable per check
# - allow multiple triggers and choose best fallback deterministically