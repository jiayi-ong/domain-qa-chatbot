from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from domain_qa.core.llm.client import LLMClient
from domain_qa.core.llm.message_history import MessageHistory
from domain_qa.core.postcheck.base import PostCheck, PostCheckContext
from domain_qa.core.prompts.domain_qa_prompt import FinanceQAPrompt
from domain_qa.core.prompts.fewshot_loader import load_fewshot_messages
from domain_qa.core.session_store import SessionStore
from domain_qa.core.utils import truncate_text


@dataclass
class AgentResult:
    answer_text: str
    used_fallback: bool = False
    postcheck_tags: list[str] = field(default_factory=list)
    model: str | None = None
    usage: dict | None = None
    trace: dict | None = None


@dataclass
class Agent:
    """
    """
    llm: LLMClient
    session_store: SessionStore
    postchecks: list[PostCheck]
    max_context_messages: Optional[int] = None
    max_chars_per_message: Optional[int] = None
    fewshot_path: str = ""
    temperature: float = 0.2
    max_tokens: int = 600

    def run(self, *, session_id: str, user_input: str) -> AgentResult:
        t0 = time.time()

        # pre-processing: limit input message character length
        user_input = truncate_text(user_input, self.max_chars_per_message)

        # retrieve session history
        history: MessageHistory = self.session_store.get(session_id)
        history.truncate_messages(self.max_context_messages)

        # prompt construction
        prompt = FinanceQAPrompt()
        fewshot = load_fewshot_messages(self.fewshot_path) if self.fewshot_path else []

        messages: List[Dict[str, str]] = prompt.build_messages(
            history_messages=history.get_messages(),
            user_input=user_input,
            fewshot_messages=fewshot,
        )

        # LLM completion
        llm_response = self.llm.complete(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer_text = (llm_response.text or "").strip()

        # update chat history
        history.add_user_message(user_input)
        history.add_assistant_message(answer_text)
        history.truncate_messages(self.max_context_messages)
        self.session_store.save(session_id, history)

        # post-checks + fallback prompts
        used_fallback = False
        tags: list[str] = []

        ctx = PostCheckContext(llm=self.llm, user_input=user_input)

        for check in self.postchecks:
            if check.trigger(answer_text=answer_text, user_input=user_input):
                fallback_text = check.run_fallback(ctx)
                answer_text = fallback_text.strip()
                used_fallback = True
                tags.append(check.deterministic_tag)
                break  # first-match-wins policy (deterministic)

        trace = {"latency_s": round(time.time() - t0, 4)}
        
        return AgentResult(
            answer_text=answer_text,
            used_fallback=used_fallback,
            postcheck_tags=tags,
            model=llm_response.model,
            usage=llm_response.usage,
            trace=trace,
        )