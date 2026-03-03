# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from litellm import completion

from domain_qa.core.errors import LLMError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str | None = None
    usage: dict | None = None
    raw: dict | None = None


class LLMClient:
    """
    Thin wrapper around litellm.completion.
    Not a singleton; can be instantiated by DI. Stateless by design.
    """

    def __init__(self, model: str, timeout_s: float = 30.0):
        self.model = model
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 600,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        try:
            resp = completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout_s,
                **(extra or {}),
            )
            # LiteLLM format can vary by provider; keep parsing minimal.
            text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage")
            model = resp.get("model", self.model)
            return LLMResponse(text=text, model=model, usage=usage, raw=resp)
        except Exception as e:
            logger.exception("LLM completion failed")
            raise LLMError(str(e)) from e