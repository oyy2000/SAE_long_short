#!/usr/bin/env python3
"""Analyze teacher interventions and paired downstream student accuracy."""

from __future__ import annotations

import argparse
import csv
import errno
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    publish_files_hash_verified,
    read_json,
    validated_training_artifacts,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


LABELS = {
    "no_steering": "No steering",
    "short_feature_enhance": "Enhance short features",
    "long_feature_suppress": "Suppress long features",
    "random_feature_enhance": "Random feature control",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    checkpoint_root = _resolve(args.checkpoint_root)
    evaluation_root = _resolve(args.evaluation_root)
    publish_output_dir, publish_figure_dir = _resolve(args.output_dir), _resolve(args.figure_dir)
    if publish_output_dir.exists() or publish_figure_dir.exists():
        raise FileExistsError("Analysis output or figure directory already exists.")
    stage_root_value = os.environ.get("SAE_ANALYSIS_OUTPUT_STAGE_ROOT")
    stage_root = Path(stage_root_value) if stage_root_value else None
    output_dir = stage_root / "analysis" if stage_root else publish_output_dir
    figure_dir = stage_root / "figures" if stage_root else publish_figure_dir
    if output_dir.exists() or figure_dir.exists():
        raise FileExistsError("Runtime analysis output or figure directory already exists.")
    output_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)

    generation_manifest_path = _resolve(config["outputs"]["result_root"]) / "main_generation/generation_merge_manifest.json"
    generation = read_json(generation_manifest_path)
    run_rows, prediction_rows, evidence = _load_evaluations(config, checkpoint_root, evaluation_root)
    grouped = _summarize_cells(run_rows)
    paired = _paired_differences(config, prediction_rows)
    base = next(row for row in run_rows if row["model_id"] == "base_student")
    training = _training_summary(config, checkpoint_root)
    decision = _decision(config, grouped, paired)

    _write_csv(output_dir / "student_run_metrics.csv", run_rows)
    _write_csv(output_dir / "student_condition_metrics.csv", grouped)
    _write_csv(output_dir / "paired_accuracy_differences.csv", paired)
    _write_csv(output_dir / "training_budget_metrics.csv", training)
    report = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "claim_boundary": config["claim_boundary"],
        "teacher_generation_metrics": generation["condition_metrics"],
        "teacher_common_support_question_count": generation["common_support_question_count"],
        "teacher_common_support_fraction": generation["common_support_fraction"],
        "base_student": base,
        "student_condition_metrics": grouped,
        "paired_accuracy_differences": paired,
        "training_budget_metrics": training,
        "decision": decision,
        "evaluation_evidence": evidence,
        "formal_claim_allowed": False,
        "limitations": [
            "The intervened SAE features were selected for short/long association, not teaching utility.",
            "This is an exploratory single-SAE and single-strength causal pilot.",
            "All branches share a token-identical natural prefix, reconstructed rather than a literal cloned KV cache.",
            "Downstream comparisons use only the all-condition correct common-support training questions.",
        ],
    }
    report_path = output_dir / "analysis_report.json"
    write_json_exclusive(report_path, report)
    _plot_generation(generation["condition_metrics"], figure_dir / "teacher_intervention_outcomes.png")
    _plot_accuracy(grouped, float(base["accuracy"]), figure_dir / "student_accuracy_by_budget.png")
    _plot_paired(paired, figure_dir / "paired_student_accuracy_delta.png")
    figure_hashes = {path.name: file_sha256(path) for path in sorted(figure_dir.glob("*.png"))}
    artifact_hashes = {path.name: file_sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "analysis_artifacts": artifact_hashes,
        "figure_artifacts": figure_hashes,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "analysis_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        "status=complete\n"
        f"config_hash={manifest['config_hash']}\n"
        f"report_sha256={file_sha256(report_path)}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    if stage_root is not None:
        publish_files_hash_verified(
            output_dir,
            publish_output_dir,
            tuple(sorted(path.name for path in output_dir.iterdir() if path.is_file())),
            attempts=120,
            wait_seconds=5.0,
        )
        publish_files_hash_verified(
            figure_dir,
            publish_figure_dir,
            tuple(sorted(path.name for path in figure_dir.iterdir() if path.is_file())),
            attempts=120,
            wait_seconds=5.0,
        )
    print(json.dumps(report["decision"], indent=2), flush=True)


def _load_evaluations(config, checkpoint_root, evaluation_root):
    run_rows, predictions, evidence = [], {}, []
    ids = []
    for budget in config["student_sft"]["budget_regimes"]:
        for condition in config["main_generation"]["conditions"]:
            for seed in config["student_sft"]["seeds"]:
                ids.append((f"{budget}__{condition}__seed_{seed}", budget, condition, int(seed)))
    ids.append(("base_student", "base", "base_student", None))
    expected_support = None
    for model_id, budget, condition, seed in ids:
        recovery_root_value = os.environ.get("SAE_EVALUATION_RECOVERY_ROOT")
        recovery = Path(recovery_root_value) / model_id if recovery_root_value else None
        root = recovery if recovery is not None and recovery.is_dir() else evaluation_root / model_id
        manifest_path, summary_path, prediction_path = root / "evaluation_manifest.json", root / "summary.json", root / "predictions.jsonl"
        marker, manifest, summary, rows = _remote_io_retry(
            lambda: _read_evaluation_bundle(root), f"evaluation:{model_id}"
        )
        support = tuple(row["problem_id"] for row in rows)
        if expected_support is None:
            expected_support = support
        elif support != expected_support:
            raise ValueError(f"Evaluation support/order mismatch: {model_id}")
        predictions[model_id] = {row["problem_id"]: bool(row["is_correct"]) for row in rows}
        train_metrics = None
        if seed is not None:
            train_metrics = _remote_io_retry(
                lambda: _validated_training(checkpoint_root, model_id),
                f"training:{model_id}",
            )["metrics"]
        run_rows.append(
            {
                "model_id": model_id,
                "budget_regime": budget,
                "condition": condition,
                "seed": seed,
                "n": int(summary["n"]),
                "correct": int(summary["correct"]),
                "accuracy": float(summary["accuracy"]),
                "mean_output_tokens": float(summary["mean_output_tokens"]),
                "completion_token_updates": int(train_metrics["completion_token_updates"]) if train_metrics else None,
                "optimizer_steps": int(train_metrics["optimizer_steps"]) if train_metrics else None,
                "elapsed_training_seconds": float(train_metrics["elapsed_seconds"]) if train_metrics else None,
            }
        )
        evidence.append({"model_id": model_id, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path)})
    return run_rows, predictions, evidence


def _summarize_cells(run_rows):
    cells = defaultdict(list)
    for row in run_rows:
        if row["seed"] is not None:
            cells[(row["budget_regime"], row["condition"])].append(row)
    output = []
    for (budget, condition), rows in sorted(cells.items()):
        accuracy = np.array([row["accuracy"] for row in rows], dtype=float)
        length = np.array([row["mean_output_tokens"] for row in rows], dtype=float)
        half = 4.3026527299 * float(accuracy.std(ddof=1)) / math.sqrt(len(accuracy)) if len(accuracy) > 1 else 0.0
        output.append(
            {
                "budget_regime": budget,
                "condition": condition,
                "seed_count": len(rows),
                "mean_accuracy": float(accuracy.mean()),
                "seed_t95_low": float(accuracy.mean() - half),
                "seed_t95_high": float(accuracy.mean() + half),
                "mean_student_output_tokens": float(length.mean()),
                "min_accuracy": float(accuracy.min()),
                "max_accuracy": float(accuracy.max()),
            }
        )
    return output


def _paired_differences(config, predictions):
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    samples = int(config["evaluation"]["bootstrap_samples"])
    output = []
    seeds = [int(seed) for seed in config["student_sft"]["seeds"]]
    for budget in config["student_sft"]["budget_regimes"]:
        base_ids = [f"{budget}__no_steering__seed_{seed}" for seed in seeds]
        problem_ids = sorted(predictions[base_ids[0]])
        for condition in config["main_generation"]["conditions"]:
            if condition == "no_steering":
                continue
            differences = np.array(
                [
                    [
                        int(predictions[f"{budget}__{condition}__seed_{seed}"][problem_id])
                        - int(predictions[f"{budget}__no_steering__seed_{seed}"][problem_id])
                        for problem_id in problem_ids
                    ]
                    for seed in seeds
                ], dtype=float,
            )
            boot = np.empty(samples, dtype=float)
            for index in range(samples):
                seed_index = rng.integers(0, len(seeds), size=len(seeds))
                question_index = rng.integers(0, len(problem_ids), size=len(problem_ids))
                boot[index] = differences[np.ix_(seed_index, question_index)].mean()
            output.append(
                {
                    "budget_regime": budget,
                    "condition": condition,
                    "comparison": f"{condition} - no_steering",
                    "paired_accuracy_delta": float(differences.mean()),
                    "bootstrap_95_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_95_high": float(np.quantile(boot, 0.975)),
                    "bootstrap_probability_positive": float((boot > 0).mean()),
                    "question_count": len(problem_ids),
                    "seed_count": len(seeds),
                }
            )
    return output


def _training_summary(config, checkpoint_root):
    output = []
    for budget in config["student_sft"]["budget_regimes"]:
        for condition in config["main_generation"]["conditions"]:
            metrics = [
                _remote_io_retry(
                    lambda model_id=f"{budget}__{condition}__seed_{seed}": _validated_training(checkpoint_root, model_id),
                    f"training:{budget}:{condition}:seed_{seed}",
                )["metrics"]
                for seed in config["student_sft"]["seeds"]
            ]
            output.append(
                {
                    "budget_regime": budget,
                    "condition": condition,
                    "completion_token_updates": int(metrics[0]["completion_token_updates"]),
                    "model_input_token_updates": int(metrics[0]["model_input_token_updates"]),
                    "optimizer_steps": int(metrics[0]["optimizer_steps"]),
                    "mean_training_seconds": float(np.mean([row["elapsed_seconds"] for row in metrics])),
                }
            )
    return output


def _decision(config, grouped, paired):
    by_pair = {(row["budget_regime"], row["condition"]): row for row in paired}
    effects = {}
    for condition in ("short_feature_enhance", "long_feature_suppress"):
        effects[condition] = {}
        for budget in config["student_sft"]["budget_regimes"]:
            row = by_pair[(budget, condition)]
            effects[condition][budget] = {
                "paired_accuracy_delta": row["paired_accuracy_delta"],
                "bootstrap_95_ci": [row["bootstrap_95_low"], row["bootstrap_95_high"]],
                "evidence_of_improvement": row["bootstrap_95_low"] > 0,
            }
    successful = [condition for condition, budgets in effects.items() if all(value["evidence_of_improvement"] for value in budgets.values())]
    return {
        "primary_endpoint": "student exact-match accuracy on GSM8K test[50:1319]",
        "improvement_requires": "paired 95% bootstrap CI above zero in both equal-example and equal-target-token regimes",
        "effects": effects,
        "successful_interventions": successful,
        "overall_conclusion": "improvement_detected" if successful else "no_robust_improvement_detected",
    }


def _plot_generation(rows, path):
    import matplotlib.pyplot as plt
    labels = [LABELS[row["condition"]] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(labels, [100 * row["unconditional_accuracy"] for row in rows], color="#4472C4")
    axes[0].set_ylabel("Unconditional correctness (%)")
    axes[1].bar(labels, [row["mean_output_tokens"] for row in rows], color="#70AD47")
    axes[1].set_ylabel("Teacher output tokens")
    axes[2].bar(labels, [100 * row["question_coverage"] for row in rows], color="#ED7D31")
    axes[2].set_ylabel("Question coverage (%)")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=28)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_accuracy(rows, base_accuracy, path):
    import matplotlib.pyplot as plt
    budgets = sorted({row["budget_regime"] for row in rows})
    conditions = ["no_steering", "short_feature_enhance", "long_feature_suppress", "random_feature_enhance"]
    figure, axes = plt.subplots(1, len(budgets), figsize=(13.5, 4.6), sharey=True)
    for axis, budget in zip(np.atleast_1d(axes), budgets):
        selected = {(row["condition"]): row for row in rows if row["budget_regime"] == budget}
        means = np.array([100 * selected[c]["mean_accuracy"] for c in conditions])
        low = np.array([100 * selected[c]["seed_t95_low"] for c in conditions])
        high = np.array([100 * selected[c]["seed_t95_high"] for c in conditions])
        axis.bar(range(len(conditions)), means, color=["#7F7F7F", "#4472C4", "#ED7D31", "#A5A5A5"])
        axis.errorbar(range(len(conditions)), means, yerr=np.vstack([means-low, high-means]), fmt="none", color="black", capsize=3)
        axis.axhline(100 * base_accuracy, color="#C00000", linestyle="--", linewidth=1, label="Base student")
        axis.set_xticks(range(len(conditions)), [LABELS[c] for c in conditions], rotation=28, ha="right")
        axis.set_title(budget.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("Student exact-match accuracy (%)")
    np.atleast_1d(axes)[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_paired(rows, path):
    import matplotlib.pyplot as plt
    labels = [f"{row['budget_regime'].replace('_', ' ')}\n{LABELS[row['condition']]}" for row in rows]
    center = np.array([100 * row["paired_accuracy_delta"] for row in rows])
    low = np.array([100 * row["bootstrap_95_low"] for row in rows])
    high = np.array([100 * row["bootstrap_95_high"] for row in rows])
    y = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(9, 5.8))
    axis.errorbar(center, y, xerr=np.vstack([center-low, high-center]), fmt="o", color="#4472C4", capsize=4)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Paired student accuracy difference vs no steering (percentage points)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_evaluation_bundle(root: Path):
    marker = read_key_value_marker(root / "EVALUATION_COMPLETE")
    manifest_path = root / "evaluation_manifest.json"
    summary_path = root / "summary.json"
    prediction_path = root / "predictions.jsonl"
    if marker.get("status") != "complete" or marker.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError(f"Invalid evaluation marker: {root.name}")
    manifest, summary = read_json(manifest_path), read_json(summary_path)
    if marker.get("predictions_sha256") != file_sha256(prediction_path) or marker.get("summary_sha256") != file_sha256(summary_path):
        raise ValueError(f"Evaluation artifact hash mismatch: {root.name}")
    rows = [json.loads(line) for line in prediction_path.open("r", encoding="utf-8") if line.strip()]
    return marker, manifest, summary, rows


def _remote_io_retry(function, label: str):
    for attempt in range(1, 121):
        try:
            return function()
        except OSError as exc:
            if exc.errno != errno.EREMOTEIO or attempt == 120:
                raise
            print(f"Transient remote I/O while reading {label}; retry={attempt}/120", flush=True)
            time.sleep(5)
    raise AssertionError("Unreachable remote-I/O retry state")


def _validated_training(checkpoint_root: Path, model_id: str):
    recovery_root = os.environ.get("SAE_ADAPTER_RECOVERY_ROOT")
    recovery = Path(recovery_root) / model_id if recovery_root else None
    root = recovery if recovery is not None and recovery.is_dir() else checkpoint_root / model_id
    artifact = validated_training_artifacts(root)
    artifact["canonical_root"] = str(checkpoint_root / model_id)
    artifact["used_recovery_mirror"] = root != checkpoint_root / model_id
    return artifact


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
