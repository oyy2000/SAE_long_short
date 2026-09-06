"""Statistical analysis for the controlled Phase-0 observation matrix."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from length_budget_distill.factorial_analysis import holm_adjust
from length_budget_distill.ranked_multiseed_analysis import crossed_seed_problem_bootstrap
from .trace_observation import LENGTH_RANKS


CellKey = Tuple[str, str, str, str]


def index_predictions(
    evaluation_runs: Sequence[Mapping[str, Any]],
    prediction_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[Dict[CellKey, Dict[int, Dict[str, Dict[str, Any]]]], Dict[str, Any]]:
    """Index prediction rows by controlled cell and training seed."""

    indexed: Dict[CellKey, Dict[int, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    base: Dict[str, Any] = {}
    common_support: set[str] | None = None
    for run in evaluation_runs:
        model_id = str(run["model_id"])
        rows = list(prediction_rows[model_id])
        by_problem = {str(row["problem_id"]): dict(row) for row in rows}
        if len(by_problem) != len(rows):
            raise ValueError(f"Duplicate evaluation problem IDs for {model_id}")
        support = set(by_problem)
        if common_support is None:
            common_support = support
        elif support != common_support:
            raise ValueError("Evaluation runs do not share identical problem support.")
        if model_id == "base":
            base = summarize_prediction_rows(rows)
            continue
        key = (
            str(run["budget_regime"]),
            str(run["loss_normalization"]),
            str(run["loss_mask"]),
            str(run["length_rank"]),
        )
        seed = int(run["seed"])
        if seed in indexed[key]:
            raise ValueError(f"Duplicate seed in analysis cell: {key} seed={seed}")
        indexed[key][seed] = by_problem
    if not base:
        raise ValueError("Base-model evaluation is missing.")
    return dict(indexed), base


def analyze_cells(
    indexed: Mapping[CellKey, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    *,
    rank_contrasts: Sequence[Sequence[str]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return arm summaries and paired rank contrasts with crossed intervals."""

    arm_rows: List[Dict[str, Any]] = []
    for key in sorted(indexed):
        regime, normalization, loss_mask, rank = key
        by_seed = indexed[key]
        correctness = {
            seed: {
                problem_id: float(bool(row["is_correct"]))
                for problem_id, row in by_problem.items()
            }
            for seed, by_problem in by_seed.items()
        }
        accuracy_bootstrap = crossed_seed_problem_bootstrap(
            correctness,
            samples=bootstrap_samples,
            seed=_derived_seed(bootstrap_seed, key, "arm"),
        )
        per_seed_accuracy = {
            seed: statistics.fmean(values.values()) for seed, values in correctness.items()
        }
        per_seed_length = {
            seed: statistics.fmean(
                float(row["output_token_count"]) for row in by_problem.values()
            )
            for seed, by_problem in by_seed.items()
        }
        arm_rows.append(
            {
                "budget_regime": regime,
                "loss_normalization": normalization,
                "loss_mask": loss_mask,
                "length_rank": rank,
                "seed_count": len(by_seed),
                "problem_count": int(accuracy_bootstrap["problem_count"]),
                "mean_accuracy": accuracy_bootstrap["estimate"],
                "accuracy_ci_low": accuracy_bootstrap["ci_low"],
                "accuracy_ci_high": accuracy_bootstrap["ci_high"],
                "accuracy_seed_sd": statistics.stdev(per_seed_accuracy.values()),
                "mean_output_tokens": statistics.fmean(per_seed_length.values()),
                "output_tokens_seed_sd": statistics.stdev(per_seed_length.values()),
                "per_seed_accuracy": {str(seed): value for seed, value in sorted(per_seed_accuracy.items())},
                "per_seed_output_tokens": {str(seed): value for seed, value in sorted(per_seed_length.items())},
            }
        )

    contrast_rows: List[Dict[str, Any]] = []
    family_groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for regime, normalization, loss_mask in sorted({key[:3] for key in indexed}):
        for left_rank, right_rank in rank_contrasts:
            left = indexed[(regime, normalization, loss_mask, str(left_rank))]
            right = indexed[(regime, normalization, loss_mask, str(right_rank))]
            effects = _paired_effects(left, right)
            result = crossed_seed_problem_bootstrap(
                effects,
                samples=bootstrap_samples,
                seed=_derived_seed(
                    bootstrap_seed,
                    (regime, normalization, loss_mask, left_rank, right_rank),
                    "contrast",
                ),
            )
            row = {
                "budget_regime": regime,
                "loss_normalization": normalization,
                "loss_mask": loss_mask,
                "left_rank": str(left_rank),
                "right_rank": str(right_rank),
                **result,
            }
            family_groups[(regime, normalization, loss_mask)].append(len(contrast_rows))
            contrast_rows.append(row)
    for indices in family_groups.values():
        adjusted = holm_adjust([float(contrast_rows[index]["bootstrap_p_value"]) for index in indices])
        for index, value in zip(indices, adjusted):
            contrast_rows[index]["bootstrap_holm_p_value"] = value
            contrast_rows[index]["holm_significant"] = value < 0.05
    return arm_rows, contrast_rows


def attach_training_accounting(
    arm_rows: List[Dict[str, Any]],
    training_runs: Sequence[Mapping[str, Any]],
) -> None:
    by_cell: Dict[CellKey, List[Mapping[str, Any]]] = defaultdict(list)
    for run in training_runs:
        key = (
            str(run["budget_regime"]),
            str(run["loss_normalization"]),
            str(run["loss_mask"]),
            str(run["length_rank"]),
        )
        by_cell[key].append(run)
    for row in arm_rows:
        key = (
            str(row["budget_regime"]),
            str(row["loss_normalization"]),
            str(row["loss_mask"]),
            str(row["length_rank"]),
        )
        runs = by_cell.get(key, [])
        if len(runs) != int(row["seed_count"]):
            raise ValueError(f"Training accounting is incomplete for {key}")
        for field in (
            "record_count",
            "unique_problem_count",
            "completion_token_updates",
            "model_input_token_updates",
            "effective_loss_token_updates",
            "optimizer_steps",
            "total_parameter_count",
            "trainable_parameter_count",
            "approximate_nonpadding_training_flops",
            "approximate_padded_training_flops",
        ):
            values = [int(run[field]) for run in runs]
            if len(set(values)) != 1:
                raise ValueError(f"Training accounting differs across seeds: {key} {field}")
            row[field] = values[0]
        padded = [int(run["padded_model_token_updates"]) for run in runs]
        row["mean_padded_model_token_updates"] = statistics.fmean(padded)
        row["padded_model_token_updates_seed_sd"] = statistics.stdev(padded)
        row["mean_train_loss"] = statistics.fmean(float(run["mean_train_loss"]) for run in runs)
        row["mean_training_elapsed_seconds"] = statistics.fmean(
            float(run["elapsed_seconds"]) for run in runs
        )


def decide_gate0(
    contrasts: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    gate = dict(config["gate0"])
    primary_left, primary_right = [str(value) for value in gate["primary_contrast"]]
    required: List[Dict[str, Any]] = []
    for cell in gate["required_short_vs_long_cells"]:
        matches = [
            row
            for row in contrasts
            if row["budget_regime"] == cell["budget_regime"]
            and row["loss_normalization"] == cell["loss_normalization"]
            and row["loss_mask"] == gate["primary_loss_mask"]
            and row["left_rank"] == primary_left
            and row["right_rank"] == primary_right
        ]
        if len(matches) != 1:
            raise ValueError(f"Gate-0 contrast is missing or duplicated: {cell}")
        row = matches[0]
        estimate_pass = float(row["estimate"]) > float(gate["minimum_accuracy_effect"])
        interval_pass = float(row["ci_low"]) > float(gate["minimum_ci_lower_bound"])
        required.append(
            {
                "budget_regime": row["budget_regime"],
                "loss_normalization": row["loss_normalization"],
                "loss_mask": row["loss_mask"],
                "estimate": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "bootstrap_holm_p_value": row["bootstrap_holm_p_value"],
                "estimate_pass": estimate_pass,
                "interval_pass": interval_pass,
                "passed": estimate_pass and interval_pass,
            }
        )
    passed = all(row["passed"] for row in required)
    return {
        "status": "passed" if passed else "failed",
        "decision": "continue_to_phase1" if passed else "stop_sae_story",
        "primary_contrast": [primary_left, primary_right],
        "primary_loss_mask": gate["primary_loss_mask"],
        "required_cells": required,
        "require_all_cells": bool(gate["require_all_cells"]),
        "pass_action": gate["pass_action"],
        "fail_action": gate["fail_action"],
        "interpretation": (
            "The short advantage survived equal-token/update budgets and both registered loss normalizations."
            if passed
            else "At least one required controlled cell did not establish a positive short advantage."
        ),
    }


def summarize_prediction_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("Cannot summarize empty predictions.")
    correct = sum(bool(row["is_correct"]) for row in values)
    return {
        "n": len(values),
        "correct": correct,
        "accuracy": correct / len(values),
        "mean_output_tokens": statistics.fmean(float(row["output_token_count"]) for row in values),
    }


def _paired_effects(
    left: Mapping[int, Mapping[str, Mapping[str, Any]]],
    right: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> Dict[int, Dict[str, float]]:
    if set(left) != set(right):
        raise ValueError("Paired rank contrast has different seed support.")
    result: Dict[int, Dict[str, float]] = {}
    for seed in sorted(left):
        if set(left[seed]) != set(right[seed]):
            raise ValueError("Paired rank contrast has different question support.")
        result[seed] = {
            problem_id: float(bool(left[seed][problem_id]["is_correct"]))
            - float(bool(right[seed][problem_id]["is_correct"]))
            for problem_id in sorted(left[seed])
        }
    return result


def _derived_seed(base: int, values: Sequence[Any], label: str) -> int:
    import hashlib

    payload = ":".join([str(base), label, *(str(value) for value in values)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
