from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    session_id: str
    answer_text: str
    used_fallback: bool = False
    postcheck_tags: list[str] = []
    model: str | None = None


# TO-DO:
# - return usage stats (tokens, latency)
# - return structured fields (e.g., "refusal": bool, "uncertainty": bool)