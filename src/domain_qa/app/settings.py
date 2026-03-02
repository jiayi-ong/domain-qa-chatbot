from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    port: int = 8080

    # LLM
    llm_model: str = "vertex_ai/gemini-2.0-flash-lite"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 600
    llm_timeout_s: float = 30.0

    # Agent/session behavior
    max_messages: int = 64
    max_chars_per_message: int = 10000

    # Prompts / few-shot loading
    fewshot_path: str = "src/domain_qa/core/prompts/data/fewshot_examples.jsonl"

    # Post-check toggles
    enable_postchecks: bool = True

    model_config = SettingsConfigDict(env_prefix="DOMAIN_QA_", extra="ignore")


# TO-DO:
# - typed secrets handling (SecretStr) for API keys if read here
# - environment-specific settings (dev/prod) and validation