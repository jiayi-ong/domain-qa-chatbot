from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from domain_qa.core.llm.client import LLMClient
from domain_qa.core.prompts.prompt_base import PromptTemplate


@dataclass(frozen=True)
class PostCheckContext:
    llm: LLMClient
    user_input: str
    # You can add: history, settings, etc. later.


class PostCheck(ABC):
    name: str
    deterministic_tag: str
    fallback_prompt: PromptTemplate

    @abstractmethod
    def trigger(self, *, answer_text: str, user_input: str) -> bool:
        """Deterministic trigger only (regex/keyword/classifier)."""

    def run_fallback(self, ctx: PostCheckContext) -> str:
        msgs = [{"role": "system", "content": self.fallback_prompt.system()}]
        msgs.append({"role": "user", "content": ctx.user_input})
        resp = ctx.llm.complete(msgs)
        return f"{resp.text}\n{self.deterministic_tag}"


# TO-DO:
# - ordering/priority policy when multiple postchecks trigger