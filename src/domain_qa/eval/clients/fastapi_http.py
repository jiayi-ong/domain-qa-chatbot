# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class ChatbotClientConfig:
    base_url: str
    chat_path: str
    timeout_s: float
    response_answer_key: str = "answer_text"


class FastAPIChatbotClient:
    """
    Minimal HTTP client for your FastAPI chatbot.

    Assumes:
      Request:  POST {base_url}{chat_path}  JSON {"query": "..."} (adjust below if needed)
      Response: JSON matching ChatResponse:
        {
          "session_id": "...",
          "answer_text": "...",
          "used_fallback": false,
          "postcheck_tags": [...],
          "model": "..."
        }
    """

    def __init__(self, cfg: ChatbotClientConfig):
        self.cfg = cfg

    def ask(
        self,
        query: str,
        *,
        return_meta: bool = False,
    ) -> str | Tuple[str, Dict[str, Any]]:
        
        url = self.cfg.base_url.rstrip("/") + "/" + self.cfg.chat_path.lstrip("/")

        payload: Dict[str, Any] = {
            "session_id": "1",
            "user_input": query
        }

        r = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        key = self.cfg.response_answer_key
        if key not in data:
            raise ValueError(f"Response missing key '{key}'. Got keys: {list(data.keys())}")

        answer = data[key]
        if not isinstance(answer, str):
            raise ValueError(f"'{key}' must be a string; got {type(answer)}")

        if not return_meta:
            return answer

        meta = {
            "session_id": data.get("session_id"),
            "used_fallback": data.get("used_fallback", False),
            "postcheck_tags": data.get("postcheck_tags", []),
            "model": data.get("model"),
        }
        return answer, meta