from __future__ import annotations

import re

from domain_qa.core.postcheck.base import PostCheck
from domain_qa.core.prompts.prompt_base import PromptTemplate


class SafetyFallbackPrompt(PromptTemplate):
    def system(self) -> str:
        return """You are a supportive assistant.
If the user expresses self-harm intent or severe distress:
- respond empathetically
- encourage seeking immediate professional help
- suggest contacting local emergency services or a trusted person
Keep it brief and do not provide harmful instructions.
"""


class SafetyDistressPostCheck(PostCheck):
    name = "safety_distress"
    deterministic_tag = "[POSTCHECK:SAFETY_FALLBACK]"
    fallback_prompt = SafetyFallbackPrompt()

    _distress_patterns = [
        r"\b(i want to die|kill myself|end it all)\b",
        r"\b(suicide|self harm|self-harm)\b",
    ]

    def trigger(self, *, answer_text: str, user_input: str) -> bool:
        # Deterministic check on user input (preferred for safety).
        return any(re.search(p, user_input, flags=re.IGNORECASE) for p in self._distress_patterns)


# TO-DO:
# - expand patterns carefully; add false-positive tests in your golden dataset
# - localize emergency guidance by country if you collect user locale (optional)