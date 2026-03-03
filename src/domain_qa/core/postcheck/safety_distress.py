# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

import re

from domain_qa.core.postcheck.base import PostCheck
from domain_qa.core.prompts.prompt_base import PromptTemplate


class SafetyFallbackPrompt(PromptTemplate):
    def system(self) -> str:
        return """<instructions>
You are a supportive assistant.
If the user expresses severe distress or intent to harm themselves:
- respond empathetically
- encourage seeking immediate professional help
- suggest contacting local emergency services or a trusted person
- be concise but caring
</instructions>

<example>
User: I can't take this anymore, I want to end it all.
Assistant: I'm really sorry you're feeling this way. 
If you're in the US, please call or text 988 (Suicide & Crisis Lifeline) or chat at 988lifeline.org.
</example>
"""


class SafetyDistressPostCheck(PostCheck):
    name = "safety_distress"
    deterministic_tag = "[POSTCHECK:SAFETY_FALLBACK]"
    fallback_prompt = SafetyFallbackPrompt()

    # user distress keyword patterns
    _user_distress_patterns = [
        r"\b(i want to die|kill myself|end it all)\b",
        r"\b(suicide|self harm|self-harm)\b",
        r"\b(i can'?t go on|can'?t do this anymore)\b",
        r"\b(no reason to live|nothing to live for)\b",
        r"\b(wish i (?:was|were) dead|better off dead)\b",
        r"\b(i'?m done with life|done with everything)\b",
        r"\b(everyone would be better off without me)\b",
        r"\b(i hate my life|hate being alive)\b",
        r"\b(self[- ]?injur(?:y|ies)|cutting myself)\b",
        r"\b(overdose|od(?:'d|ed)|take all (?:the )?pills)\b",
        r"\b(jump off (?:a|the) bridge|step in front of (?:a|the) train)\b",
        r"\b(i don'?t want to wake up|wish i wouldn'?t wake up)\b"
    ]

    # keyword patterns for good safety response in the model answer
    _safety_response_patterns = [
        r"\b(seek (?:help|support|assistance))\b",
        r"\b(contact (?:emergency services|a trusted person|a mental health professional))\b",
        r"\b(it'?s okay to ask for help|help is available)\b",
    ]

    def trigger(self, *, answer_text: str, user_input: str) -> bool:
        # condition 1: distress detected in user input
        user_distress = any(
            re.search(p, user_input, flags=re.IGNORECASE) 
            for p in self._user_distress_patterns
        )
        # condition 2: no good safety response in the model answer
        good_safety_answer = any(
            re.search(p, answer_text, flags=re.IGNORECASE)
            for p in self._safety_response_patterns
        )

        return user_distress and not good_safety_answer