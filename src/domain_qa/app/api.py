# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from domain_qa.app.schemas import ChatRequest, ChatResponse
from domain_qa.core.agent import Agent
from domain_qa.core.errors import AgentError
from domain_qa.app.dependencies import get_agent

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: Agent = Depends(get_agent)):
    try:
        result = agent.run(session_id=req.session_id, user_input=req.user_input)
        return ChatResponse(
            session_id=req.session_id,
            answer_text=result.answer_text,
            used_fallback=result.used_fallback,
            postcheck_tags=result.postcheck_tags,
            model=result.model,
        )
    except AgentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error") from e


# TO-DO:
# - rate limiting, request size limits, and basic abuse prevention