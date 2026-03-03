# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class PromptTemplate(ABC):
    @abstractmethod
    def system(self) -> str:
        """Return the system prompt text."""

    def build_messages(
        self,
        *,
        history_messages: List[Dict[str, str]],
        user_input: str,
        fewshot_messages: List[Dict[str, str]] | None = None,
    ) -> List[Dict[str, str]]:
        
        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": self.system()}
        ]

        if fewshot_messages:
            msgs.extend(fewshot_messages)
        msgs.extend(history_messages)
        msgs.append({"role": "user", "content": user_input})
        
        return msgs


# TO-DO:
# - prompt rendering with placeholders and validation (e.g., Jinja2-lite)
# - "answer schema" enforced via structured output (JSON) if desired