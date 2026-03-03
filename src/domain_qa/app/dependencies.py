# IN-CODE CITATION:
# Contributed by ChatGPT 5.2 (OpenAI) in February, 2026.
# Level of contribution: (1) Scaffold Assistance
# ChatGPT generated only structural elements
# (e.g., file template, class definitions, function signatures, docstrings).

from __future__ import annotations

from functools import lru_cache

from domain_qa.app.settings import Settings
from domain_qa.core.agent import Agent
from domain_qa.core.llm.client import LLMClient
from domain_qa.core.postcheck.registry import default_postchecks
from domain_qa.core.session_store import InMemorySessionStore, SessionStore


@lru_cache
def get_settings() -> Settings:
    return Settings()


# WARNING: not durable across instances
# i.e. Cloud Run may run multiple instances; sessions won't be shared across them.
_SESSION_STORE: InMemorySessionStore = InMemorySessionStore()


def get_session_store() -> SessionStore:
    return _SESSION_STORE


def get_agent() -> Agent:
    settings = get_settings()

    llm = LLMClient(
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
    )
    
    postchecks = default_postchecks(enabled=settings.enable_postchecks)

    return Agent(
        llm=llm,
        session_store=get_session_store(),
        postchecks=postchecks,
        max_context_messages=settings.max_messages,
        max_chars_per_message=settings.max_chars_per_message,
        fewshot_path=settings.fewshot_path,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


# TO-DO:
# - swap InMemorySessionStore for Redis/Firestore with same interface
# - create Agent per-request if you add per-request tracing context