"""Crossed seed-by-question bootstrap for paired accuracy effects."""

from __future__ import annotations

import random
from statistics import mean, stdev
from typing import Any, Dict, Mapping


def crossed_seed_problem_bootstrap(
    effects_by_seed: Mapping[int, Mapping[str, float]],
    *,
    samples: int = 10_000,
    seed: int = 20260826,
) -> Dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive.")
    seeds = sorted(int(value) for value in effects_by_seed)
    if len(seeds) < 2:
        raise ValueError("At least two training seeds are required.")
    support = set(effects_by_seed[seeds[0]])
    if not support or any(
        set(effects_by_seed[value]) != support for value in seeds[1:]
    ):
        raise ValueError(
            "Training seeds must have identical non-empty problem support."
        )
    problem_ids = sorted(support)
    matrix = {
        value: [float(effects_by_seed[value][problem_id]) for problem_id in problem_ids]
        for value in seeds
    }
    per_seed = {value: mean(matrix[value]) for value in seeds}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        problem_indices = [rng.randrange(len(problem_ids)) for _ in problem_ids]
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        draws.append(
            mean(
                mean(matrix[value][index] for index in problem_indices)
                for value in sampled_seeds
            )
        )
    draws.sort()
    non_positive = sum(value <= 0.0 for value in draws) / samples
    non_negative = sum(value >= 0.0 for value in draws) / samples
    return {
        "estimate": mean(per_seed.values()),
        "ci_low": draws[max(0, int(0.025 * samples) - 1)],
        "ci_high": draws[min(samples - 1, int(0.975 * samples))],
        "bootstrap_p_value": min(1.0, 2.0 * min(non_positive, non_negative)),
        "seed_count": len(seeds),
        "problem_count": len(problem_ids),
        "bootstrap_samples": samples,
        "per_seed_effects": {str(key): value for key, value in per_seed.items()},
        "per_seed_sample_sd": stdev(per_seed.values()),
        "resampling_units": ["training_seed", "paired_problem"],
    }
