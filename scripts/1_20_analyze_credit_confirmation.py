#!/usr/bin/env python3
"""Analyze the three-seed credit-allocation confirmation and issue Gate 1.5."""

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
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.factorial_analysis import holm_adjust
from length_budget_distill.ranked_multiseed_analysis import (
    crossed_seed_problem_bootstrap,
)
from length_budget_distill.records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase1_5_credit_allocation_v1.json"
    )
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument(
        "--output-dir", default="results/phase1_5_credit_allocation_v1/formal/analysis"
    )
    parser.add_argument(
        "--figure-dir", default="figures/phase1_5_credit_allocation_v1/confirmation"
    )
    args = parser.parse_args()
    config_path, output_dir, figure_dir = (
        _resolve(args.config),
        _resolve(args.output_dir),
        _resolve(args.figure_dir),
    )
    root_marker = output_dir.parents[1] / "PHASE1_5_COMPLETE"
    if output_dir.exists() or figure_dir.exists() or root_marker.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {figure_dir}")
    config = read_json(config_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    pilot = require_passed_gate(PROJECT_ROOT, config["pilot_gate_dependency"])
    conditions = set(config["confirmation"]["conditions"])
    seeds = {int(value) for value in config["confirmation"]["seeds"]}
    runs = _load(_resolve(args.evaluation_root), conditions, seeds)
    summaries = []
    for condition in sorted(conditions):
        summaries.append(
            {
                "condition": condition,
                "seed_count": len(seeds),
                "question_count": int(
                    next(iter(runs[condition].values()))["summary"]["n"]
                ),
                "mean_accuracy": statistics.fmean(
                    float(run["summary"]["accuracy"])
                    for run in runs[condition].values()
                ),
                "accuracy_seed_sd": statistics.stdev(
                    float(run["summary"]["accuracy"])
                    for run in runs[condition].values()
                ),
                "mean_output_tokens": statistics.fmean(
                    float(run["summary"]["mean_output_tokens"])
                    for run in runs[condition].values()
                ),
                "mean_training_elapsed_seconds": statistics.fmean(
                    float(run["training_metrics"]["elapsed_seconds"])
                    for run in runs[condition].values()
                ),
                "mean_completion_token_updates": statistics.fmean(
                    float(run["training_metrics"]["completion_token_updates"])
                    for run in runs[condition].values()
                ),
                "mean_effective_supervision_weight_updates": statistics.fmean(
                    float(
                        run["training_metrics"]["effective_supervision_weight_updates"]
                    )
                    for run in runs[condition].values()
                ),
                "per_seed_accuracy": {
                    str(seed): float(runs[condition][seed]["summary"]["accuracy"])
                    for seed in sorted(seeds)
                },
            }
        )
    target = str(config["gate1_5"]["confirmation_contrast"])
    contrasts = []
    for index, baseline in enumerate(
        config["gate1_5"]["confirmation_required_ci_lower_above"]
    ):
        effects = {}
        for seed in sorted(seeds):
            left = {row["problem_id"]: row for row in runs[target][seed]["predictions"]}
            right = {
                row["problem_id"]: row for row in runs[baseline][seed]["predictions"]
            }
            if set(left) != set(right):
                raise ValueError("Confirmation prediction support differs.")
            effects[seed] = {
                problem_id: float(bool(left[problem_id]["is_correct"]))
                - float(bool(right[problem_id]["is_correct"]))
                for problem_id in left
            }
        result = crossed_seed_problem_bootstrap(
            effects,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"]) + 500_009 + index,
        )
        contrasts.append(
            {"left_condition": target, "right_condition": baseline, **result}
        )
    adjusted = holm_adjust([row["bootstrap_p_value"] for row in contrasts])
    for row, value in zip(contrasts, adjusted):
        row["holm_p_value"] = value
        row["passed"] = float(row["ci_low"]) > 0.0 and value < 0.05
    passed = all(row["passed"] for row in contrasts)
    gate = {
        "status": "passed" if passed else "failed",
        "decision": "allow_phase2_sae" if passed else "stop_before_sae",
        "target_condition": target,
        "paired_contrasts": contrasts,
        "pass_action": config["gate1_5"]["pass_action"],
        "fail_action": config["gate1_5"]["fail_action"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    summary_path, contrast_path = (
        output_dir / "confirmation_summary.csv",
        output_dir / "paired_contrasts.csv",
    )
    _write_csv(summary_path, summaries)
    _write_csv(contrast_path, contrasts)
    gate_path = output_dir / "gate1_5_decision.json"
    write_json_exclusive(gate_path, gate)
    figures = _plot(summaries, figure_dir)
    analysis = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate1_evidence": gate1,
        "pilot_evidence": pilot,
        "run_count": sum(len(value) for value in runs.values()),
        "summaries": summaries,
        "contrasts": contrasts,
        "gate1_5": gate,
        "run_evidence": [
            {
                "condition": condition,
                "seed": seed,
                "summary_path": str(run["path"]),
                "summary_sha256": file_sha256(run["path"]),
            }
            for condition, by_seed in sorted(runs.items())
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
    analysis_path = output_dir / "credit_allocation_analysis.json"
    write_json_exclusive(analysis_path, analysis)
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={analysis['config_hash']}\nanalysis_sha256={file_sha256(analysis_path)}\n"
        f"gate1_5_decision_sha256={file_sha256(gate_path)}\ngate1_5_status={gate['status']}\n",
        encoding="utf-8",
    )
    root_marker.write_text(
        f"status=complete\nconfig_hash={analysis['config_hash']}\n"
        f"analysis_sha256={file_sha256(analysis_path)}\n"
        f"gate1_5_decision_sha256={file_sha256(gate_path)}\n"
        f"gate1_5_status={gate['status']}\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )


def _load(root, conditions, seeds):
    runs = defaultdict(dict)
    for path in sorted(root.glob("**/evaluation_summary.json")):
        row = read_json(path)
        if row.get("split") != "test":
            continue
        condition, seed = str(row["condition"]), int(row["seed"])
        prediction_path = Path(row["predictions_path"])
        if file_sha256(prediction_path) != row["predictions_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        if seed in runs[condition]:
            raise ValueError(f"Duplicate confirmation run: {condition} seed {seed}")
        training = validated_training_artifacts(row["adapter_path"])
        if row.get("training_metrics_sha256") != training["hashes"]["training_metrics"]:
            raise ValueError(
                "Confirmation evaluation is bound to other training metrics."
            )
        runs[condition][seed] = {
            "summary": row,
            "path": path,
            "predictions": list(read_jsonl(prediction_path)),
            "training_metrics": training["metrics"],
        }
    if set(runs) != conditions:
        raise ValueError(
            f"Confirmation conditions mismatch: {set(runs)} != {conditions}"
        )
    if any(set(by_seed) != seeds for by_seed in runs.values()):
        raise ValueError("Confirmation seed support is incomplete.")
    if (
        len(
            {
                run["summary"]["problem_ids_sha256"]
                for by_seed in runs.values()
                for run in by_seed.values()
            }
        )
        != 1
    ):
        raise ValueError("Confirmation question support differs.")
    return dict(runs)


def _plot(rows, figure_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["condition"].replace("_", "\n") for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
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
    axes[0].set_ylabel("Test exact-match accuracy (%)")
    axes[1].set_ylabel("Mean output tokens")
    figure.suptitle("Phase 1.5 fixed-context credit-allocation confirmation")
    figure.tight_layout()
    png, pdf = (
        figure_dir / "credit_confirmation.png",
        figure_dir / "credit_confirmation.pdf",
    )
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


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
