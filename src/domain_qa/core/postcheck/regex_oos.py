from __future__ import annotations

import re

from domain_qa.core.postcheck.base import PostCheck
from domain_qa.core.prompts.prompt_base import PromptTemplate


class OOSFallbackPrompt(PromptTemplate):
    def system(self) -> str:
        return """You are a domain-restricted assistant.
If the user asks something out-of-scope, refuse briefly and offer what you can do instead.
Do NOT provide out-of-scope details.
"""


class RegexOutOfScopePostCheck(PostCheck):
    name = "regex_out_of_scope"
    deterministic_tag = "[POSTCHECK:OOS_FALLBACK]"
    fallback_prompt = OOSFallbackPrompt()

    # Minimal heuristic: detect if answer contains strong signals of hallucinated general knowledge.
    # In practice, tune these to your domain and your expected refusal style.
    _bad_answer_patterns = [
        r"\b(as an ai language model)\b",
        r"\bhere is a legal advice\b",
        r"\bmedical diagnosis\b",
    ]

    def trigger(self, *, answer_text: str, user_input: str) -> bool:
        return any(re.search(p, answer_text, flags=re.IGNORECASE) for p in self._bad_answer_patterns)


# TO-DO:
# - replace patterns with domain-specific OOS detection (keywords, taxonomy)
# - check the *user_input* category too, not only the answer_text