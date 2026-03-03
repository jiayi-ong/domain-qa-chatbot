from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ----------------------------
# Dataset + Evaluation Output Schemas
# ----------------------------

class DatasetItem(BaseModel):
    """Schema for a single evaluation case (Golden Reference)."""
    id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    expected_answer: str = ""


class DeterministicScores(BaseModel):
    jaccard: float = Field(..., ge=0.0, le=1.0)
    rouge1_f1: float = Field(..., ge=0.0, le=1.0)
    rouge2_f1: float = Field(..., ge=0.0, le=1.0)


class GoldenJudgeOutput(BaseModel):
    score: int = Field(..., ge=0, le=10)
    rationale: str = Field(..., min_length=1)


class RubricItemScore(BaseModel):
    title: str
    score: int = Field(..., ge=0, le=10)
    rationale: str


class RubricJudgeOutput(BaseModel):
    overall_score: int = Field(..., ge=0, le=10)
    item_scores: List[RubricItemScore]
    summary: str = Field(..., min_length=1)


class CaseResult(BaseModel):
    id: str
    category: str
    query: str
    expected_answer: str
    model_answer: str

    deterministic: DeterministicScores
    golden_judge: Optional[GoldenJudgeOutput] = None
    rubric_judge: Optional[RubricJudgeOutput] = None

    passed: bool
    failure_reasons: List[str] = []


class EvalSummary(BaseModel):
    total: int
    passed: int
    pass_rate: float
    by_category: Dict[str, Dict[str, float]]


class EvalReport(BaseModel):
    config: Dict[str, str]
    summary: EvalSummary
    results: List[CaseResult]


# ----------------------------
# Runtime Config
# ----------------------------

@dataclass(frozen=True)
class EvalConfig:
    # FastAPI chatbot endpoint
    base_url: str = "http://127.0.0.1:8000"
    chat_path: str = "/chat"
    timeout_s: float = 30.0

    # Response parsing
    response_answer_key: str = "answer_text"

    # Datasets
    datasets_dir: Path = Path(__file__).parent / "datasets"
    in_domain_path: Path = datasets_dir / "golden.v1.jsonl"
    expected_refusals_path: Path = datasets_dir / "golden.v1.expected_refusals.jsonl"
    adversarial_path: Path = datasets_dir / "golden.v1.adversarial.jsonl"

    # Deterministic thresholds (kept intentionally simple)
    min_jaccard_pass: float = 0.20
    min_rouge1_f1_pass: float = 0.20
    min_rouge2_f1_pass: float = 0.05

    # Judges
    judge_model: str = "vertex_ai/gemini-2.0-flash-lite"
    judge_timeout_s: float = 30.0
    judge_temperature: float = 0.0
    judge_max_tokens: int = 500

    # Pass/Fail policy
    # - require deterministic OR (golden judge >= min_golden_score)
    min_golden_score_pass: int = 7
    rubric_enabled: bool = True
    golden_judge_enabled: bool = True