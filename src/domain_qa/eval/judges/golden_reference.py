from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import ValidationError

from domain_qa.core.llm.client import LLMClient
from domain_qa.eval.config import GoldenJudgeOutput


GOLDEN_REFERENCE_SYSTEM_PROMPT = """
<instructions>
You are a strict but fair evaluator for a Q&A chatbot in the domain of financial analysis using quantitative metrics.

You will be given:
- a user query
- an expected (golden) answer
- the chatbot's answer

You will score the chatbot answer from 0 to 10 (inclusive) based on factual correctness and completeness relative to the golden reference.
</instructions>

<scoring_guidelines>
- 10: fully correct, covers all key points, no meaningful errors
- 7-9: mostly correct; minor omissions or minor issues
- 4-6: partially correct; missing important points or has notable issues
- 1-3: mostly incorrect or irrelevant but not empty
- 0: completely wrong, unsafe, or refusal when an answer is expected
</scoring_guidelines>

<output_format>
Output MUST be valid JSON and MUST match this schema exactly:
{
  "score": integer (0..10),
  "rationale": string
}
</output_format>
"""


def _safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Minimal robustness: if model returns extra text, attempt to extract the first JSON object.
    """
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(t[start : end + 1])

    raise ValueError("No JSON object found in judge output.")


class GoldenReferenceJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, *, query: str, expected_answer: str, model_answer: str) -> GoldenJudgeOutput:
        user_prompt = (
            "Evaluate the model answer.\n\n"
            f"<USER QUERY>\n{query}\n</USER QUERY>\n"
            f"<EXPECTED ANSWER>\n{expected_answer}\n</EXPECTED ANSWER>\n"
            f"<MODEL ANSWER>\n{model_answer}\n</MODEL ANSWER>\n"
        )

        resp = self.llm.complete(
            messages=[
                {"role": "system", "content": GOLDEN_REFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        try:
            data = _safe_json_extract(resp.text)
            return GoldenJudgeOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            return GoldenJudgeOutput(score=0, rationale=f"Invalid judge JSON output: {e}")