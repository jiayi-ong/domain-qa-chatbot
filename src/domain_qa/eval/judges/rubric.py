from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from domain_qa.core.llm.client import LLMClient
from domain_qa.eval.config import RubricItemScore, RubricJudgeOutput


@dataclass(frozen=True)
class RubricItem:
    title: str
    description: str
    weight: float  # must sum to 1.0


# Start with TWO items; extend to 10 by adding more entries.
DEFAULT_RUBRIC: List[RubricItem] = [
    RubricItem(
        title="Correctness",
        description="Is the answer factually correct and aligned with the domain? Penalize hallucinations.",
        weight=0.60,
    ),
    RubricItem(
        title="Conciseness",
        description="Does the answer avoid redundancy and unnecessary verbosity while remaining clear?",
        weight=0.40,
    ),
]


class _RubricLLMOutput(BaseModel):
    overall_score: int = Field(..., ge=0, le=10)
    item_scores: List[RubricItemScore]
    summary: str = Field(..., min_length=1)


RUBRIC_SYSTEM_PROMPT = """You are a strict rubric-based evaluator for a domain QA chatbot.

You will be given:
- a user query
- the model's answer
- a rubric: a list of items (title, description, weight)

For each rubric item, assign an integer score from 0 to 10 and provide a brief rationale.
Then compute an overall_score (0..10) as a WEIGHTED average of item scores (rounded to nearest integer).

Output MUST be valid JSON and MUST match this schema exactly:
{
  "overall_score": integer (0..10),
  "item_scores": [
     {"title": string, "score": integer (0..10), "rationale": string}
  ],
  "summary": string
}

Rules:
- The "title" fields MUST exactly match the rubric item titles provided.
- No extra keys. No markdown. No trailing commentary.
"""


def _safe_json_extract(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(t[start : end + 1])

    raise ValueError("No JSON object found in judge output.")


class RubricJudge:
    def __init__(self, llm: LLMClient, rubric: List[RubricItem] | None = None):
        self.llm = llm
        self.rubric = rubric or DEFAULT_RUBRIC

    def evaluate(self, *, query: str, model_answer: str) -> RubricJudgeOutput:
        rubric_text = "\n".join(
            [f"- {r.title} (weight={r.weight}): {r.description}" for r in self.rubric]
        )
        user_prompt = (
            "Evaluate the model answer using the rubric.\n\n"
            f"RUBRIC:\n{rubric_text}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"MODEL ANSWER:\n{model_answer}\n"
        )

        resp = self.llm.complete(
            messages=[
                {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=600,
        )

        try:
            data = _safe_json_extract(resp.text)
            parsed = _RubricLLMOutput.model_validate(data)

            # Enforce exact titles match rubric (schema + policy)
            expected_titles = [r.title for r in self.rubric]
            got_titles = [x.title for x in parsed.item_scores]
            if got_titles != expected_titles:
                return RubricJudgeOutput(
                    overall_score=0,
                    item_scores=[],
                    summary=f"Invalid rubric titles/order. Expected {expected_titles}, got {got_titles}.",
                )

            return RubricJudgeOutput(
                overall_score=parsed.overall_score,
                item_scores=parsed.item_scores,
                summary=parsed.summary,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            return RubricJudgeOutput(
                overall_score=0,
                item_scores=[],
                summary=f"Invalid judge JSON output: {e}",
            )