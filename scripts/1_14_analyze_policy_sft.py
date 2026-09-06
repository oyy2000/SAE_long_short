#!/usr/bin/env python3
"""Analyze confirmed policy SFT runs and issue the complete Gate-1 decision."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    validated_training_artifacts,
    write_json_exclusive,
)
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
)
from length_budget_distill.factorial_analysis import holm_adjust
from length_budget_distill.ranked_multiseed_analysis import (
    crossed_seed_problem_bootstrap,
)
from length_budget_distill.records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--ctv-analysis",
        default="results/phase1_teaching_utility_v1/formal/ctv_analysis/ctv_analysis.json",
    )
    parser.add_argument(
        "--mid-ctv-analysis",
        default="results/phase1_teaching_utility_v1/formal/mid_anchor_sensitivity/ctv_analysis/ctv_analysis.json",
    )
    parser.add_argument(
        "--confirmation-plan",
        default="results/phase1_teaching_utility_v1/formal/confirmation_plan/confirmation_plan.json",
    )
    parser.add_argument("--test-root", required=True)
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/analysis"
    )
    parser.add_argument(
        "--figure-dir", default="figures/phase1_teaching_utility_v1/policy_sft"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    root_marker = output_dir.parents[1] / "PHASE1_COMPLETE"
    if output_dir.exists() or figure_dir.exists() or root_marker.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {figure_dir}")
    config = read_json(config_path)
    entry = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    ctv_path = _resolve(args.ctv_analysis)
    ctv_analysis = read_json(ctv_path)
    ctv_marker_path = ctv_path.parent / "CTV_ANALYSIS_COMPLETE"
    ctv_marker = read_key_value_marker(ctv_marker_path)
    if ctv_analysis.get("status") != "complete" or ctv_marker.get(
        "analysis_sha256"
    ) != file_sha256(ctv_path):
        raise ValueError("Base-anchor CTV analysis is not marker-bound.")
    gradient_gate = dict(ctv_analysis["gradient_gate"])
    mid_ctv_path = _resolve(args.mid_ctv_analysis)
    mid_ctv_analysis = read_json(mid_ctv_path)
    mid_marker_path = mid_ctv_path.parent / "CTV_ANALYSIS_COMPLETE"
    mid_marker = read_key_value_marker(mid_marker_path)
    if (
        mid_ctv_analysis.get("status") != "complete"
        or mid_ctv_analysis.get("analysis_label") != "mid_sft_anchor"
        or mid_marker.get("analysis_sha256") != file_sha256(mid_ctv_path)
    ):
        raise ValueError(
            "Mid-SFT anchor sensitivity analysis is missing or mislabeled."
        )
    plan_path = _resolve(args.confirmation_plan)
    plan = read_json(plan_path)
    plan_marker_path = plan_path.parent / "CONFIRMATION_PLAN_COMPLETE"
    plan_marker = read_key_value_marker(plan_marker_path)
    if plan.get("status") != "complete" or plan_marker.get(
        "plan_sha256"
    ) != file_sha256(plan_path):
        raise ValueError("Confirmation plan is not marker-bound.")
    label_to_policy = dict(plan["confirmation_label_to_policy"])
    seeds = [int(value) for value in config["policy_sft"]["confirm_seeds"]]
    expected_policies = set(label_to_policy.values())
    runs = _load_runs(_resolve(args.test_root), expected_policies, set(seeds))
    indexed = {
        policy: {
            seed: {str(row["problem_id"]): dict(row) for row in run["predictions"]}
            for seed, run in by_seed.items()
        }
        for policy, by_seed in runs.items()
    }
    summaries = []
    for label, policy in label_to_policy.items():
        by_seed = runs[policy]
        summaries.append(
            {
                "method_label": label,
                "selection_policy": policy,
                "seed_count": len(by_seed),
                "question_count": len(by_seed[seeds[0]]["predictions"]),
                "mean_accuracy": statistics.fmean(
                    float(by_seed[seed]["summary"]["accuracy"]) for seed in seeds
                ),
                "accuracy_seed_sd": statistics.stdev(
                    float(by_seed[seed]["summary"]["accuracy"]) for seed in seeds
                ),
                "mean_output_tokens": statistics.fmean(
                    float(by_seed[seed]["summary"]["mean_output_tokens"])
                    for seed in seeds
                ),
                "mean_training_elapsed_seconds": statistics.fmean(
                    float(by_seed[seed]["training_metrics"]["elapsed_seconds"])
                    for seed in seeds
                ),
                "mean_completion_token_updates": statistics.fmean(
                    float(by_seed[seed]["training_metrics"]["completion_token_updates"])
                    for seed in seeds
                ),
                "mean_final_train_loss": statistics.fmean(
                    float(by_seed[seed]["training_metrics"]["final_train_loss"])
                    for seed in seeds
                ),
                "mean_initial_step_loss": statistics.fmean(
                    _convergence_values(by_seed[seed]["training_metrics"])[0]
                    for seed in seeds
                ),
                "mean_ten_percent_step_loss": statistics.fmean(
                    _convergence_values(by_seed[seed]["training_metrics"])[1]
                    for seed in seeds
                ),
                "mean_early_loss_change": statistics.fmean(
                    _convergence_values(by_seed[seed]["training_metrics"])[1]
                    - _convergence_values(by_seed[seed]["training_metrics"])[0]
                    for seed in seeds
                ),
                "per_seed_accuracy": {
                    str(seed): float(by_seed[seed]["summary"]["accuracy"])
                    for seed in seeds
                },
            }
        )
    baseline_labels_by_policy = defaultdict(list)
    for baseline_label in ("random", "strongest_non_ctv"):
        baseline_labels_by_policy[label_to_policy[baseline_label]].append(
            baseline_label
        )
    contrasts = []
    for index, (baseline_policy, baseline_labels) in enumerate(
        sorted(baseline_labels_by_policy.items())
    ):
        effects = {}
        for seed in seeds:
            left = indexed[label_to_policy["ctv"]][seed]
            right = indexed[baseline_policy][seed]
            if set(left) != set(right):
                raise ValueError("Confirmation runs have different test support.")
            effects[seed] = {
                problem_id: float(bool(left[problem_id]["is_correct"]))
                - float(bool(right[problem_id]["is_correct"]))
                for problem_id in left
            }
        result = crossed_seed_problem_bootstrap(
            effects,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"]) + 700_001 + index,
        )
        contrasts.append(
            {
                "left_label": "ctv",
                "left_policy": label_to_policy["ctv"],
                "right_labels": baseline_labels,
                "right_policy": baseline_policy,
                **result,
            }
        )
    adjusted = holm_adjust([float(row["bootstrap_p_value"]) for row in contrasts])
    for row, value in zip(contrasts, adjusted):
        row["holm_p_value"] = value
        row["passed"] = float(row["ci_low"]) > 0.0 and value < 0.05
    sft_gate_passed = all(row["passed"] for row in contrasts)
    gate1_passed = gradient_gate["status"] == "passed" and sft_gate_passed
    gate1 = {
        "status": "passed" if gate1_passed else "failed",
        "decision": "continue_to_phase1_5"
        if gate1_passed
        else "stop_before_phase1_5_and_sae",
        "gradient_gate_status": gradient_gate["status"],
        "policy_sft_gate_status": "passed" if sft_gate_passed else "failed",
        "required_baseline_labels": ["random", "strongest_non_ctv"],
        "policy_sft_contrasts": contrasts,
        "pass_action": config["gate1"]["pass_action"],
        "fail_action": config["gate1"]["fail_action"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "policy_sft_summary.csv"
    _write_csv(summary_path, summaries)
    contrast_path = output_dir / "policy_sft_paired_contrasts.csv"
    _write_csv(contrast_path, contrasts)
    gate_path = output_dir / "gate1_decision.json"
    write_json_exclusive(gate_path, gate1)
    figures = _plot(summaries, figure_dir)
    figures.extend(_plot_convergence(runs, label_to_policy, seeds, figure_dir))
    payload = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "entry_evidence": entry,
        "ctv_analysis_path": str(ctv_path),
        "ctv_analysis_sha256": file_sha256(ctv_path),
        "mid_ctv_analysis_path": str(mid_ctv_path),
        "mid_ctv_analysis_sha256": file_sha256(mid_ctv_path),
        "mid_anchor_sensitivity": mid_ctv_analysis["analyses"],
        "upstream_completion_markers": [
            {"path": str(ctv_marker_path), "sha256": file_sha256(ctv_marker_path)},
            {"path": str(mid_marker_path), "sha256": file_sha256(mid_marker_path)},
            {
                "path": str(plan_marker_path),
                "sha256": file_sha256(plan_marker_path),
            },
        ],
        "confirmation_plan_path": str(plan_path),
        "confirmation_plan_sha256": file_sha256(plan_path),
        "run_count": sum(len(value) for value in runs.values()),
        "unique_policy_count": len(runs),
        "policy_summaries": summaries,
        "paired_contrasts": contrasts,
        "gate1": gate1,
        "run_evidence": [
            {
                "policy": policy,
                "seed": seed,
                "summary_path": str(run["summary_path"]),
                "summary_sha256": file_sha256(run["summary_path"]),
                "predictions_sha256": run["summary"]["predictions_sha256"],
            }
            for policy, by_seed in sorted(runs.items())
            for seed, run in sorted(by_seed.items())
        ],
        "artifacts": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_path, contrast_path, gate_path, *figures)
        ],
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    analysis_path = output_dir / "policy_sft_analysis.json"
    write_json_exclusive(analysis_path, payload)
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={payload['config_hash']}\nanalysis_sha256={file_sha256(analysis_path)}\n"
        f"gate1_decision_sha256={file_sha256(gate_path)}\ngate1_status={gate1['status']}\n",
        encoding="utf-8",
    )
    root_marker.write_text(
        f"status=complete\nconfig_hash={payload['config_hash']}\n"
        f"analysis_sha256={file_sha256(analysis_path)}\n"
        f"gate1_decision_sha256={file_sha256(gate_path)}\n"
        f"gate1_status={gate1['status']}\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )


def _load_runs(root: Path, expected_policies, expected_seeds):
    runs = defaultdict(dict)
    for path in sorted(root.glob("*/*/evaluation_summary.json")):
        row = read_json(path)
        if row.get("status") != "complete" or row.get("split") != "test":
            raise ValueError(f"Invalid confirmation result: {path}")
        policy, seed = str(row["policy"]), int(row["seed"])
        prediction_path = Path(row["predictions_path"])
        if file_sha256(prediction_path) != row["predictions_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        if seed in runs[policy]:
            raise ValueError(f"Duplicate confirmation run: {policy} seed {seed}")
        runs[policy][seed] = {
            "summary": row,
            "summary_path": path,
            "predictions": list(read_jsonl(prediction_path)),
            "training_metrics": _load_training_metrics(row),
        }
    if set(runs) != expected_policies:
        raise ValueError(
            f"Confirmation policies mismatch: {set(runs)} != {expected_policies}"
        )
    for policy in runs:
        if set(runs[policy]) != expected_seeds:
            raise ValueError(f"Confirmation seed support mismatch for {policy}")
    support_hashes = {
        run["summary"]["problem_ids_sha256"]
        for by_seed in runs.values()
        for run in by_seed.values()
    }
    if len(support_hashes) != 1:
        raise ValueError("Confirmation runs do not have identical question support.")
    return dict(runs)


def _plot(rows, figure_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [row["method_label"].replace("_", "\n") for row in rows]
    axes[0].bar(
        range(len(rows)),
        [100 * float(row["mean_accuracy"]) for row in rows],
        color="#4C78A8",
    )
    axes[1].bar(
        range(len(rows)),
        [float(row["mean_output_tokens"]) for row in rows],
        color="#F2A541",
    )
    for axis in axes:
        axis.set_xticks(range(len(rows)), labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("GSM8K exact-match accuracy (%)")
    axes[1].set_ylabel("Mean output tokens")
    figure.suptitle("Phase 1 confirmed policy SFT")
    figure.tight_layout()
    png = figure_dir / "policy_sft_accuracy_length.png"
    pdf = figure_dir / "policy_sft_accuracy_length.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _plot_convergence(runs, label_to_policy, seeds, figure_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    colors = {
        "ctv": "#0072B2",
        "strongest_non_ctv": "#D55E00",
        "quantile_short": "#009E73",
        "random": "#777777",
    }
    for label, policy in label_to_policy.items():
        curves = [
            runs[policy][seed]["training_metrics"]["optimizer_step_metrics"]
            for seed in seeds
        ]
        width = min(len(curve) for curve in curves)
        x = [int(curves[0][index]["optimizer_step"]) for index in range(width)]
        y = [
            statistics.fmean(
                float(curve[index]["mean_microbatch_loss"]) for curve in curves
            )
            for index in range(width)
        ]
        axis.plot(x, y, color=colors[label], alpha=0.35, linewidth=0.8)
        window = max(1, min(25, width // 10))
        smoothed = [
            statistics.fmean(y[max(0, index - window + 1) : index + 1])
            for index in range(width)
        ]
        axis.plot(x, smoothed, color=colors[label], linewidth=2.0, label=label)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Mean training loss")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    png = figure_dir / "policy_sft_loss_convergence.png"
    pdf = figure_dir / "policy_sft_loss_convergence.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _load_training_metrics(evaluation_summary):
    evidence = validated_training_artifacts(evaluation_summary["adapter_path"])
    metrics = evidence["metrics"]
    if (
        evaluation_summary.get("training_metrics_sha256")
        != evidence["hashes"]["training_metrics"]
    ):
        raise ValueError("Evaluation is bound to another training-metrics artifact.")
    if "optimizer_step_metrics" not in metrics:
        raise ValueError(
            "Training convergence curve is missing: "
            f"{evidence['root']}/training_metrics.json"
        )
    return metrics


def _convergence_values(metrics):
    curve = metrics["optimizer_step_metrics"]
    if not curve:
        raise ValueError("Training convergence curve is empty.")
    tenth_index = min(len(curve) - 1, max(0, (len(curve) + 9) // 10 - 1))
    return (
        float(curve[0]["mean_microbatch_loss"]),
        float(curve[tenth_index]["mean_microbatch_loss"]),
    )


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
