"""Question-paired statistics for Counterfactual Teaching Value (CTV)."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from .factorial_analysis import holm_adjust


def paired_sign_randomization(left, right, *, samples, seed):
    """Two-sided paired-label randomization; requires exchangeability under H0.

    Enumerate up to 16 nonzero pairs; otherwise report a seeded Monte Carlo
    estimate with the plus-one correction. This is not a bootstrap tail area.
    """
    import numpy as np
    if set(left) != set(right) or len(left) < 2 or samples < 1:
        raise ValueError('Paired randomization requires identical nonempty support and positive samples')
    differences = np.array([float(left[k])-float(right[k]) for k in sorted(left)], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError('Nonfinite paired outcomes')
    active = differences[differences != 0]; n = len(active)
    if not n:
        return {'p_value':1., 'method':'exact_paired_sign_randomization', 'draws':1, 'nonzero_pairs':0}
    threshold = abs(active.sum()); exceed = 0
    exact = n <= 16; total = 2**n if exact else samples
    rng = np.random.default_rng(seed)
    for start in range(0,total,1024):
        count = min(1024,total-start)
        if exact:
            bits = (np.arange(start,start+count,dtype=np.uint64)[:,None] >> np.arange(n,dtype=np.uint64)) & 1
        else:
            bits = rng.integers(0,2,size=(count,n),dtype=np.int8)
        draws = (2*bits.astype(float)-1) @ active
        exceed += int(np.count_nonzero(np.abs(draws) >= threshold-1e-12))
    return {'p_value':exceed/total if exact else (exceed+1)/(total+1),
        'method':'exact_paired_sign_randomization' if exact else 'monte_carlo_paired_sign_randomization',
        'draws':total, 'nonzero_pairs':n}


def per_question_spearman(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    score: str,
    question_field: str = "problem_id",
) -> Dict[str, float]:
    """Compute rank correlation within each question, omitting undefined ties."""

    from scipy.stats import spearmanr

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[question_field])].append(row)
    values: Dict[str, float] = {}
    for question, group in grouped.items():
        if len(group) < 3:
            continue
        result = float(
            spearmanr(
                [float(row[outcome]) for row in group],
                [float(row[score]) for row in group],
            ).statistic
        )
        if math.isfinite(result):
            values[question] = result
    return values


def paired_question_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap the mean paired-question difference left minus right."""

    if samples <= 0:
        raise ValueError("samples must be positive.")
    support = sorted(set(left) & set(right))
    if len(support) < 2:
        raise ValueError("At least two common questions are required.")
    effects = [float(left[key]) - float(right[key]) for key in support]
    rng = random.Random(seed)
    draws = [
        statistics.fmean(effects[rng.randrange(len(effects))] for _ in effects)
        for _ in range(samples)
    ]
    draws.sort()
    non_positive = sum(value <= 0.0 for value in draws) / samples
    non_negative = sum(value >= 0.0 for value in draws) / samples
    return {
        "estimate": statistics.fmean(effects),
        "ci_low": draws[max(0, int(0.025 * samples) - 1)],
        "ci_high": draws[min(samples - 1, int(0.975 * samples))],
        "bootstrap_p_value": min(1.0, 2.0 * min(non_positive, non_negative)),
        "question_count": len(support),
        "bootstrap_samples": samples,
        "resampling_unit": "paired_question",
    }


def compare_rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    metrics: Mapping[str, str],
    primary_metric: str,
    baseline_metrics: Sequence[str],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Summarize within-question correlations and paired primary contrasts."""

    correlations = {
        name: per_question_spearman(rows, outcome=outcome, score=field)
        for name, field in metrics.items()
    }
    summaries = []
    for name in metrics:
        values = correlations[name]
        summaries.append(
            {
                "metric": name,
                "score_field": metrics[name],
                "mean_within_question_spearman": statistics.fmean(values.values()),
                "question_count": len(values),
            }
        )
    contrasts = []
    for index, baseline in enumerate(baseline_metrics):
        result = paired_question_bootstrap(
            correlations[primary_metric],
            correlations[baseline],
            samples=samples,
            seed=seed + 1009 * index,
        )
        contrasts.append(
            {
                "left_metric": primary_metric,
                "right_metric": baseline,
                **result,
            }
        )
    adjusted = holm_adjust([float(row["bootstrap_p_value"]) for row in contrasts])
    for row, value in zip(contrasts, adjusted):
        row["holm_p_value"] = value
        row["passed"] = float(row["ci_low"]) > 0.0 and value < 0.05
    return {"metric_summaries": summaries, "paired_contrasts": contrasts}
