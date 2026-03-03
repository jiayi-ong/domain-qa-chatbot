# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict, Optional


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass
class MessageHistory:
    """Simple in-memory message history for a chat session."""
    messages: list[ChatMessage] = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[ChatMessage]:
        return list(self.messages)

    def truncate_messages(self, max_messages: Optional[int]) -> None:
        if max_messages is None:
            return
        elif len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]