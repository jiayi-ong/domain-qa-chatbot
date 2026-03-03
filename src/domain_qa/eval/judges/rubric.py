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
    weight: float


DEFAULT_RUBRIC: List[RubricItem] = [
    RubricItem(
        title="Factual and Logical Accuracy",
        description=(
            "Evaluate whether the answer is factually correct (when verifiable) and internally logically consistent."
            "Penalize incorrect claims, fabricated information, contradictions, or flawed reasoning chains."
        ),
        weight=0.15,
    ),
    RubricItem(
        title="Completeness",
        description=(
            "Determine whether the answer sufficiently covers all major aspects of the user's request. "
            "Penalize significant omissions, underdeveloped explanations, or partial responses when a comprehensive answer was required."
        ),
        weight=0.15,
    ),
    RubricItem(
        title="Relevance",
        description=(
            "Judge whether the response directly addresses the user's query without digressions or unrelated content."
            "Penalize off-topic material and failure to prioritize the core question."
        ),
        weight=0.10,
    ),
    RubricItem(
        title="Clarity and Structure",
        description=(
            "Assess how clearly the response is written and organized. "
            "Consider logical flow, readability, appropriate formatting, and unambiguous phrasing. "
            "Penalize disorganized, confusing, or poorly structured answers."
        ),
        weight=0.10,
    ),
    RubricItem(
        title="Justification and Explanation",
        description=(
            "Evaluate whether conclusions, recommendations, or claims are adequately supported by reasoning, evidence, or explanation appropriate to the task. "
            "Penalize unsupported assertions or opaque reasoning when justification is expected."
        ),
        weight=0.10,
    ),
    RubricItem(
        title="Uncertainty Handling",
        description=(
            "Assess whether the response appropriately acknowledges uncertainty, assumptions, limitations, or insufficient information when relevant. "
            "Penalize unwarranted certainty, overgeneralization, or failure to note ambiguity."
        ),
        weight=0.10,
    ),
    RubricItem(
        title="Safety and Ethical Considerations",
        description=(
            "Determine whether the answer avoids harmful, unsafe, unethical, or policy-violating content. "
            "In distress or high-risk scenarios, assess whether the model responds responsibly and proportionately. "
            "Penalize unsafe guidance or inappropriate tone."
        ),
        weight=0.10,
    ),
    RubricItem(
        title="Conciseness",
        description=(
            "Evaluate whether the response is appropriately concise while still complete. "
            "Penalize excessive verbosity, repetition, filler content, or, "
            "conversely, overly terse responses that sacrifice necessary substance."
        ),
        weight=0.10,
    ),
    RubricItem(
        title="Tone and Professionalism",
        description=(
            "Assess whether the response maintains an appropriate, respectful, and professional tone. "
            "Penalize dismissiveness, condescension, emotional overreach."
        ),
        weight=0.05,
    ),
    RubricItem(
        title="Validity Checking",
        description=(
            "Evaluate whether the response appropriately identifies, questions, or corrects problematic, unsupported, or false assumptions in the user's query. "
            "The model should not blindly accept flawed premises; instead, it should clarify, qualify, or correct them when necessary. "
            "Penalize uncritical acceptance of incorrect assumptions or failure to surface key validity concerns."
        ),
        weight=0.05,
    ),
]


class _RubricLLMOutput(BaseModel):
    overall_score: int = Field(..., ge=0, le=10)
    item_scores: List[RubricItemScore]
    summary: str = Field(..., min_length=1)


RUBRIC_SYSTEM_PROMPT = """
<instructions>
You are a strict but fair evaluator for a Q&A chatbot in the domain of financial analysis using quantitative metrics.

You will be given:
- a user query
- the chatbot's answer
- a rubric: a list of items (title, description, weight)

For each rubric item, assign an integer score from 0 to 10 and provide a brief rationale.
Then compute an overall_score (0..10) as a WEIGHTED average of item scores (rounded to nearest integer).
</instructions>

<output_format>
Output MUST be valid JSON and MUST match this schema exactly:
{
  "overall_score": integer (0..10),
  "item_scores": [
     {"title": string, "score": integer (0..10), "rationale": string}
  ],
  "summary": string
}
- The "title" fields MUST exactly match the rubric item titles provided.
</output_format>
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
            f"<RUBRIC>:\n{rubric_text}\n</RUBRIC>\n"
            f"<USER QUERY>:\n{query}\n</USER QUERY>\n"
            f"<MODEL ANSWER>:\n{model_answer}\n</MODEL ANSWER>\n"
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