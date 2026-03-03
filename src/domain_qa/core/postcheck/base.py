# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from domain_qa.core.llm.client import LLMClient
from domain_qa.core.prompts.prompt_base import PromptTemplate


@dataclass(frozen=True)
class PostCheckContext:
    llm: LLMClient
    user_input: str


class PostCheck(ABC):
    name: str
    deterministic_tag: str
    fallback_prompt: PromptTemplate

    @abstractmethod
    def trigger(self, *, answer_text: str, user_input: str) -> bool:
        """Deterministic trigger (regex/keyword/classifier etc.)
        based on answer text and/or user input.
        """

    def run_fallback(self, ctx: PostCheckContext) -> str:
        """Run the fallback prompt and return the new answer text."""
        msgs = [{"role": "system", "content": self.fallback_prompt.system()}]
        msgs.append({"role": "user", "content": ctx.user_input})
        resp = ctx.llm.complete(msgs)

        return f"{resp.text}\n{self.deterministic_tag}"