from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence, Tuple

from domain_qa.eval.config import DeterministicScores


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def jaccard(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(ref_tokens: Sequence[str], hyp_tokens: Sequence[str], n: int) -> float:
    ref_ngrams = ngrams(ref_tokens, n)
    hyp_ngrams = ngrams(hyp_tokens, n)
    if not ref_ngrams and not hyp_ngrams:
        return 1.0
    if not ref_ngrams or not hyp_ngrams:
        return 0.0

    ref_counts = Counter(ref_ngrams)
    hyp_counts = Counter(hyp_ngrams)
    overlap = sum((ref_counts & hyp_counts).values())

    precision = overlap / max(1, sum(hyp_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def refusal_detected(answer: str, refusal_phrases: Iterable[str]) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True  # empty answer counts as refusal/failure-safe
    return any(p in a for p in refusal_phrases)


def compute_deterministic(
    expected_answer: str,
    model_answer: str,
    refusal_phrases: Iterable[str],
) -> DeterministicScores:
    ref_toks = tokenize(expected_answer)
    hyp_toks = tokenize(model_answer)

    return DeterministicScores(
        jaccard=jaccard(ref_toks, hyp_toks),
        rouge1_f1=rouge_n_f1(ref_toks, hyp_toks, 1),
        rouge2_f1=rouge_n_f1(ref_toks, hyp_toks, 2),
        refusal_detected=refusal_detected(model_answer, refusal_phrases),
    )