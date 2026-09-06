"""Question-paired statistics and sparse-event helpers for SAE interpretation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values with monotonic correction."""

    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.ones_like(values)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    finite_indices = np.flatnonzero(finite)
    order = finite_indices[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.ones_like(values)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    finite_indices = np.flatnonzero(finite)
    order = finite_indices[np.argsort(values[finite])]
    ranked = values[order] * (len(order) - np.arange(len(order)))
    ranked = np.maximum.accumulate(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def paired_feature_statistics(
    values: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    frequency_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute vectorized short-minus-long effects paired by problem.

    Each problem first contributes one short-group mean and one long-group mean.
    Consequently, questions rather than traces or tokens are the inferential units.
    """

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(rows):
        raise ValueError("values must have shape [trace, feature]")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"short": [], "long": []}
    )
    for index, row in enumerate(rows):
        label = str(row["analysis_length_label"])
        if label not in {"short", "long"}:
            raise ValueError(f"Unexpected length label: {label}")
        grouped[str(row["problem_id"])][label].append(index)
    differences = []
    problem_ids = []
    for problem_id in sorted(grouped):
        cells = grouped[problem_id]
        if not cells["short"] or not cells["long"]:
            raise ValueError(f"Incomplete paired problem: {problem_id}")
        differences.append(
            matrix[cells["short"]].mean(axis=0)
            - matrix[cells["long"]].mean(axis=0)
        )
        problem_ids.append(problem_id)
    diff = np.stack(differences, axis=0)
    n_questions = diff.shape[0]
    if n_questions < 2:
        raise ValueError("At least two paired questions are required.")
    mean = diff.mean(axis=0)
    standard_deviation = diff.std(axis=0, ddof=1)
    standard_error = standard_deviation / math.sqrt(n_questions)
    paired_d = np.divide(
        mean,
        standard_deviation,
        out=np.zeros_like(mean),
        where=standard_deviation > 0,
    )
    t_statistic = np.divide(
        mean,
        standard_error,
        out=np.zeros_like(mean),
        where=standard_error > 0,
    )
    try:
        from scipy.stats import t as student_t

        p_value = 2.0 * student_t.sf(np.abs(t_statistic), df=n_questions - 1)
        critical = float(student_t.ppf(0.975, df=n_questions - 1))
    except ImportError as exc:
        raise RuntimeError("scipy is required for paired feature statistics") from exc
    zero_variance_nonzero = (standard_error == 0) & (mean != 0)
    p_value[zero_variance_nonzero] = 0.0
    q_value = benjamini_hochberg(p_value)
    if frequency_values is None:
        frequency_values = matrix
    frequencies = np.asarray(frequency_values)
    if frequencies.shape != matrix.shape:
        raise ValueError("frequency_values must match values")
    prevalence = (frequencies > 0).mean(axis=0)
    return {
        "problem_ids": problem_ids,
        "question_count": n_questions,
        "effect": mean,
        "ci_low": mean - critical * standard_error,
        "ci_high": mean + critical * standard_error,
        "paired_d": paired_d,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "bh_q_value": q_value,
        "prevalence": prevalence,
        "question_differences": diff,
    }


def select_discovery_features(
    primary: Mapping[str, np.ndarray],
    early: Mapping[str, np.ndarray],
    *,
    per_direction: int,
    minimum_prevalence: float,
    maximum_bh_q: float,
    minimum_abs_paired_d: float,
) -> list[dict[str, Any]]:
    """Select dev-discovered features without inspecting validation effects."""

    primary_d = np.asarray(primary["paired_d"])
    early_d = np.asarray(early["paired_d"])
    prevalence = np.asarray(primary["prevalence"])
    q_value = np.asarray(primary["bh_q_value"])
    if not (
        primary_d.shape == early_d.shape == prevalence.shape == q_value.shape
    ):
        raise ValueError("Feature-statistic arrays must align.")
    sign_consistent = primary_d * early_d > 0
    available = (prevalence >= minimum_prevalence) & sign_consistent
    passes = (
        available
        & (q_value <= maximum_bh_q)
        & (np.abs(primary_d) >= minimum_abs_paired_d)
    )
    score = np.abs(primary_d) + 0.5 * np.abs(early_d)
    selected = []
    for direction, sign in (("short", 1), ("long", -1)):
        directional = available & (np.sign(primary_d) == sign)
        preferred = np.flatnonzero(directional & passes)
        fallback = np.flatnonzero(directional & ~passes)
        preferred = preferred[np.argsort(-score[preferred])]
        fallback = fallback[np.argsort(-score[fallback])]
        chosen = np.concatenate((preferred, fallback))[:per_direction]
        for rank, feature_id in enumerate(chosen, start=1):
            selected.append(
                {
                    "feature_id": int(feature_id),
                    "direction": direction,
                    "discovery_rank": rank,
                    "discovery_score": float(score[feature_id]),
                    "passes_discovery_gate": bool(passes[feature_id]),
                }
            )
    return selected


def relative_position_bin(position: int, length: int, bin_count: int) -> int:
    """Map a zero-based token position to an equal-width relative-position bin."""

    if length <= 0 or bin_count <= 0 or not 0 <= position < length:
        raise ValueError("Invalid relative-position inputs.")
    return min(bin_count - 1, (position * bin_count) // length)

