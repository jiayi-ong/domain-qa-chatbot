from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from domain_qa.core.llm.client import LLMClient
from domain_qa.eval.clients.fastapi_http import ChatbotClientConfig, FastAPIChatbotClient
from domain_qa.eval.config import (
    CaseResult,
    DatasetItem,
    EvalConfig,
    EvalReport,
    EvalSummary,
)
from domain_qa.eval.judges.golden_reference import GoldenReferenceJudge
from domain_qa.eval.judges.rubric import RubricJudge
from domain_qa.eval.metrics.deterministic import compute_deterministic
from domain_qa.eval.report.writer import write_reports


def _read_jsonl(path: Path) -> List[DatasetItem]:
    """Read a JSONL file from path and return a list of DatasetItem."""
    if not path.exists():
        return []
    
    items: List[DatasetItem] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            validated_line = DatasetItem.model_validate(json.loads(line))
            items.append(validated_line)
        except Exception as e:
            raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
    return items


def load_all_datasets(cfg: EvalConfig) -> List[DatasetItem]:
    """Load datasets for in-domain, expected refusals (out-of-scoep), 
    and adversarial evaluation cases, returning the combined results
    as a list of DatasetItem."""
    all_items: List[DatasetItem] = []
    all_items.extend(_read_jsonl(cfg.in_domain_path))
    all_items.extend(_read_jsonl(cfg.expected_refusals_path))
    all_items.extend(_read_jsonl(cfg.adversarial_path))

    return all_items


def categorize_summary(results: List[CaseResult]) -> EvalSummary:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    by_cat: Dict[str, Dict[str, float]] = {}

    for r in results:
        d = by_cat.setdefault(r.category, {"total": 0.0, "passed": 0.0, "pass_rate": 0.0})
        d["total"] += 1.0
        if r.passed:
            d["passed"] += 1.0

    for cat, d in by_cat.items():
        d["pass_rate"] = (d["passed"] / d["total"]) if d["total"] else 0.0

    return EvalSummary(
        total=total,
        passed=passed,
        pass_rate=(passed / total) if total else 0.0,
        by_category=by_cat,
    )


def decide_pass_fail(
    cfg: EvalConfig,
    item: DatasetItem,
    deterministic,
    golden_score: int | None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    # Answer-expected cases:
    # Deterministic pass if any threshold met (kept simple)
    det_ok = (
        deterministic.jaccard >= cfg.min_jaccard_pass
        and deterministic.rouge1_f1 >= cfg.min_rouge1_f1_pass
        and deterministic.rouge2_f1 >= cfg.min_rouge2_f1_pass
    )
    if not det_ok:
        reasons.append(
            "Deterministic thresholds not met "
            f"(jaccard>={cfg.min_jaccard_pass}, "
            "rouge1>={cfg.min_rouge1_f1_pass}, rouge2>={cfg.min_rouge2_f1_pass})"
        )

    # Golden judge pass
    judge_ok = False
    if cfg.golden_judge_enabled and golden_score is not None:
        judge_ok = golden_score >= cfg.min_golden_score_pass
        if not judge_ok:
            reasons.append(f"Golden judge score too low (score={golden_score}, min={cfg.min_golden_score_pass}).")

    # Pass if deterministic OR judge passes
    passed = det_ok or judge_ok
    if passed:
        reasons = []  # keep failure reasons only
    return passed, reasons


def main() -> int:
    """Main function to run the evaluation."""
    p = argparse.ArgumentParser(description="Run golden-dataset evals against a FastAPI chatbot.")
    p.add_argument("--base-url", type=str, default=None, help="FastAPI base URL (e.g., http://127.0.0.1:8000)")
    p.add_argument("--chat-path", type=str, default=None, help="Chat endpoint path (default: /chat)")
    p.add_argument("--timeout-s", type=float, default=None, help="HTTP timeout seconds")
    p.add_argument("--report-path", type=str, required=True, help="Output report text path")
    p.add_argument("--json-report-path", type=str, default=None, help="Optional JSON report path")
    p.add_argument("--judge-model", type=str, default=None, help="LiteLLM judge model string")
    args = p.parse_args()

    cfg = EvalConfig(
        base_url=args.base_url or EvalConfig.base_url,
        chat_path=args.chat_path or EvalConfig.chat_path,
        timeout_s=args.timeout_s or EvalConfig.timeout_s,
        judge_model=args.judge_model or EvalConfig.judge_model,
    )

    items = load_all_datasets(cfg)
    if not items:
        raise SystemExit(
            f"No dataset items found. Checked:\n"
            f"- {cfg.in_domain_path}\n- {cfg.expected_refusals_path}\n- {cfg.adversarial_path}"
        )

    client = FastAPIChatbotClient(
        ChatbotClientConfig(
            base_url=cfg.base_url,
            chat_path=cfg.chat_path,
            timeout_s=cfg.timeout_s,
            response_answer_key=cfg.response_answer_key,
        )
    )

    judge_llm = LLMClient(model=cfg.judge_model, timeout_s=cfg.judge_timeout_s)
    golden_judge = GoldenReferenceJudge(judge_llm)
    rubric_judge = RubricJudge(judge_llm)

    results: List[CaseResult] = []

    for item in items:
        model_answer = client.ask(item.query)

        det = compute_deterministic(
            expected_answer=item.expected_answer,
            model_answer=model_answer
        )

        golden_out = None
        if cfg.golden_judge_enabled and item.expected_answer.strip():
            golden_out = golden_judge.evaluate(
                query=item.query,
                expected_answer=item.expected_answer,
                model_answer=model_answer,
            )

        rubric_out = None
        if cfg.rubric_enabled:
            rubric_out = rubric_judge.evaluate(
                query=item.query,
                model_answer=model_answer,
            )

        passed, reasons = decide_pass_fail(
            cfg=cfg,
            item=item,
            deterministic=det,
            golden_score=(golden_out.score if golden_out else None),
        )

        results.append(
            CaseResult(
                id=item.id,
                category=item.category,
                query=item.query,
                expected_answer=item.expected_answer,
                model_answer=model_answer,
                deterministic=det,
                golden_judge=golden_out,
                rubric_judge=rubric_out,
                passed=passed,
                failure_reasons=reasons,
            )
        )

    summary = categorize_summary(results)
    report = EvalReport(
        config={
            "base_url": cfg.base_url,
            "chat_path": cfg.chat_path,
            "timeout_s": str(cfg.timeout_s),
            "judge_model": cfg.judge_model,
            "datasets": f"{cfg.in_domain_path.name}, {cfg.expected_refusals_path.name}, {cfg.adversarial_path.name}",
            "min_jaccard_pass": str(cfg.min_jaccard_pass),
            "min_rouge1_f1_pass": str(cfg.min_rouge1_f1_pass),
            "min_rouge2_f1_pass": str(cfg.min_rouge2_f1_pass),
            "min_golden_score_pass": str(cfg.min_golden_score_pass),
        },
        summary=summary,
        results=results,
    )

    text_path = Path(args.report_path)
    json_path = Path(args.json_report_path) if args.json_report_path else None
    write_reports(report, text_path=text_path, json_path=json_path)

    # Console output (simple)
    print(f"Total: {summary.total} | Passed: {summary.passed} | Pass rate: {summary.pass_rate:.1%}")
    for cat, d in summary.by_category.items():
        print(f"- {cat}: {int(d['passed'])}/{int(d['total'])} ({d['pass_rate']:.1%})")
    print(f"Wrote report: {text_path}")
    if json_path:
        print(f"Wrote JSON:  {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())