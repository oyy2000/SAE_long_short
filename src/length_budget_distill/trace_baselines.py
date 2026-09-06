"""Student-specific surface baselines for candidate-trace selection."""

from __future__ import annotations

import math
import hashlib
from typing import Dict, Mapping, Sequence


def rank_with_ties(
    values: Sequence[float], *, descending: bool, clip: int | None = None
) -> list[float]:
    ordered = sorted(
        range(len(values)), key=lambda index: values[index], reverse=descending
    )
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average_rank = 1.0 + (cursor + end - 1) / 2.0
        if clip is not None:
            average_rank = min(float(clip), average_rank)
        for position in range(cursor, end):
            ranks[ordered[position]] = average_rank
        cursor = end
    return ranks


def rank_surprisal_ratio(
    token_logprobs: Sequence[float], token_ranks: Sequence[float], rank_clip: int = 100
) -> float:
    """RSR from arXiv:2601.14249; lower values are preferred."""

    if not token_logprobs or len(token_logprobs) != len(token_ranks):
        raise ValueError(
            "RSR requires aligned non-empty log-probability and rank sequences."
        )
    nll = -sum(float(value) for value in token_logprobs) / len(token_logprobs)
    mean_rank = sum(min(float(rank_clip), float(value)) for value in token_ranks) / len(
        token_ranks
    )
    return mean_rank / max(nll, 1e-12)


def scas_score(
    candidate_answer_alignment: float,
    candidate_question_alignment: float,
    weight: float = 0.5,
) -> float:
    """SCAS convex combination from arXiv:2605.26872."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("SCAS weight must lie in [0, 1].")
    return (1.0 - weight) * float(candidate_answer_alignment) + weight * float(
        candidate_question_alignment
    )


def lark_g_hat(losses: Sequence[float], brier_scores: Sequence[float]) -> list[float]:
    """Official forward-pass LARK g-hat formula, evaluated within question.

    This follows Tianrun-Yu/LARK commit 38cfd4f, top-K Brier approximation.
    Larger values are preferred by the budget-one selector.
    """

    if not losses or len(losses) != len(brier_scores):
        raise ValueError("LARK requires aligned non-empty loss and Brier sequences.")
    values = [float(value) for value in [*losses, *brier_scores]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("LARK inputs must be finite.")
    rhos = [
        brier / loss if loss > 1e-30 else 0.0
        for loss, brier in zip(losses, brier_scores)
    ]
    total_loss = sum(float(value) for value in losses)
    weighted_rho = sum(rho * float(loss) for rho, loss in zip(rhos, losses))
    if total_loss <= 1e-30:
        return [float("nan")] * len(losses)
    return [
        (float(loss) / total_loss) * (2.0 * rho - weighted_rho / total_loss)
        for loss, rho in zip(losses, rhos)
    ]


def scas_blocks(
    *,
    answer_mean_nll: float,
    question_mean_nll: float,
    answer_answer_similarity: float,
    answer_question_similarity: float,
    weight: float = 0.5,
) -> Dict[str, float]:
    """Official SCAS forward-only cost from commit 4cec3a6; lower is preferred."""

    answer_answer = float(answer_mean_nll) ** 2 * float(answer_answer_similarity)
    answer_question = (
        float(answer_mean_nll)
        * float(question_mean_nll)
        * float(answer_question_similarity)
    )
    return {
        "scas_answer_answer_block": answer_answer,
        "scas_answer_question_block": answer_question,
        "scas_score": scas_score(answer_answer, answer_question, weight),
    }


def surface_baseline_record(
    *,
    completion_tokens: int,
    student_loss_sum: float,
    student_loss_mean: float,
    rsr: float | None,
    scas: float | None,
    lark_style: float | None,
) -> Dict[str, float | int | None]:
    return {
        "completion_tokens": int(completion_tokens),
        "log_completion_tokens": math.log(max(1, int(completion_tokens))),
        "student_nll_sum": float(student_loss_sum),
        "student_nll_mean": float(student_loss_mean),
        "rsr": rsr,
        "scas": scas,
        "lark_style": lark_style,
    }


def select_scas_lowest_group(
    rows: Sequence[Mapping[str, object]],
    *,
    score_field: str = "scas_score",
    groups: int = 5,
    seed: int,
) -> Mapping[str, object]:
    """Deterministically sample one candidate from SCAS's lowest-score group."""

    if groups <= 0 or not rows:
        raise ValueError("SCAS grouping requires non-empty rows and positive groups.")
    ranked = sorted(
        rows, key=lambda row: (float(row[score_field]), str(row["trace_id"]))
    )
    lowest_size = max(1, math.ceil(len(ranked) / groups))
    lowest = ranked[:lowest_size]
    return min(
        lowest,
        key=lambda row: hashlib.sha256(
            f"{seed}|{row['trace_id']}".encode("utf-8")
        ).hexdigest(),
    )
