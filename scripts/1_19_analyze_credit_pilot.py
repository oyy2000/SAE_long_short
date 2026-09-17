#!/usr/bin/env python3
"""Analyze the nine-condition seed-17 credit-allocation pilot."""

from __future__ import annotations

import argparse
import csv
import sys
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
from length_budget_distill.records import read_jsonl
from length_budget_distill.utility_analysis import paired_question_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase1_5_credit_allocation_v1.json"
    )
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/phase1_5_credit_allocation_v1/formal/pilot_analysis",
    )
    parser.add_argument(
        "--figure-dir", default="figures/phase1_5_credit_allocation_v1/pilot"
    )
    args = parser.parse_args()
    config_path, output_dir, figure_dir = (
        _resolve(args.config),
        _resolve(args.output_dir),
        _resolve(args.figure_dir),
    )
    if output_dir.exists() or figure_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {figure_dir}")
    config = read_json(config_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    expected = set(config["pilot"]["conditions"])
    seed = int(config["pilot"]["seed"])
    runs = _load(_resolve(args.evaluation_root), expected, seed, "dev")
    summaries = [
        {
            "condition": condition,
            "seed": seed,
            "accuracy": run["summary"]["accuracy"],
            "mean_output_tokens": run["summary"]["mean_output_tokens"],
            "question_count": run["summary"]["n"],
            "completion_token_updates": run["training_metrics"][
                "completion_token_updates"
            ],
            "model_input_token_updates": run["training_metrics"][
                "model_input_token_updates"
            ],
            "effective_supervision_weight_updates": run["training_metrics"][
                "effective_supervision_weight_updates"
            ],
            "training_elapsed_seconds": run["training_metrics"]["elapsed_seconds"],
        }
        for condition, run in sorted(runs.items())
    ]
    target = str(config["gate1_5"]["pilot_contrast"])
    contrasts = []
    for index, baseline in enumerate(
        config["gate1_5"]["pilot_required_ci_lower_above"]
    ):
        left = {
            row["problem_id"]: float(bool(row["is_correct"]))
            for row in runs[target]["predictions"]
        }
        right = {
            row["problem_id"]: float(bool(row["is_correct"]))
            for row in runs[baseline]["predictions"]
        }
        result = paired_question_bootstrap(
            left,
            right,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"]) + index,
        )
        contrasts.append(
            {"left_condition": target, "right_condition": baseline, **result}
        )
    adjusted = holm_adjust([row["bootstrap_p_value"] for row in contrasts])
    for row, value in zip(contrasts, adjusted):
        row["holm_p_value"] = value
        row["passed"] = float(row["ci_low"]) > 0.0 and value < 0.05
    passed = all(row["passed"] for row in contrasts)
    decision = {
        "status": "passed" if passed else "failed",
        "decision": "run_confirmation"
        if passed
        else "stop_before_confirmation_and_sae",
        "target_condition": target,
        "paired_contrasts": contrasts,
        "single_seed_selection_stage": True,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "pilot_summary.csv"
    _write_csv(summary_path, summaries)
    contrast_path = output_dir / "pilot_paired_contrasts.csv"
    _write_csv(contrast_path, contrasts)
    decision_path = output_dir / "pilot_decision.json"
    write_json_exclusive(decision_path, decision)
    figures = _plot(summaries, figure_dir)
    analysis = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate1_evidence": gate1,
        "summaries": summaries,
        "contrasts": contrasts,
        "pilot_decision": decision,
        "run_evidence": [
            {
                "condition": condition,
                "summary_path": str(run["path"]),
                "summary_sha256": file_sha256(run["path"]),
            }
            for condition, run in sorted(runs.items())
        ],
        "artifacts": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary_path, contrast_path, decision_path, *figures)
        ],
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    analysis_path = output_dir / "pilot_analysis.json"
    write_json_exclusive(analysis_path, analysis)
    (output_dir / "PILOT_ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={analysis['config_hash']}\nanalysis_sha256={file_sha256(analysis_path)}\n"
        f"pilot_decision_sha256={file_sha256(decision_path)}\npilot_status={decision['status']}\n",
        encoding="utf-8",
    )


def _load(root, expected, seed, split):
    runs = {}
    for path in sorted(root.glob("**/evaluation_summary.json")):
        row = read_json(path)
        if row.get("split") != split or int(row.get("seed", -1)) != seed:
            continue
        condition = str(row["condition"])
        if condition in runs:
            raise ValueError(f"Duplicate pilot result: {condition}")
        prediction_path = Path(row["predictions_path"])
        if file_sha256(prediction_path) != row["predictions_sha256"]:
            raise ValueError(f"Pilot prediction hash mismatch: {prediction_path}")
        training = validated_training_artifacts(row["adapter_path"])
        if row.get("training_metrics_sha256") != training["hashes"]["training_metrics"]:
            raise ValueError("Pilot evaluation is bound to other training metrics.")
        runs[condition] = {
            "summary": row,
            "path": path,
            "predictions": list(read_jsonl(prediction_path)),
            "training_metrics": training["metrics"],
        }
    if set(runs) != expected:
        raise ValueError(f"Pilot condition support mismatch: {set(runs)} != {expected}")
    if len({run["summary"]["problem_ids_sha256"] for run in runs.values()}) != 1:
        raise ValueError("Pilot question support differs across conditions.")
    return runs


def _plot(rows, figure_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["condition"].replace("_", "\n") for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(
        range(len(rows)),
        [100 * float(row["accuracy"]) for row in rows],
        color="#4C78A8",
    )
    axes[1].bar(
        range(len(rows)),
        [float(row["mean_output_tokens"]) for row in rows],
        color="#F2A541",
    )
    axes[0].set_ylabel("Dev exact-match accuracy (%)")
    axes[1].set_ylabel("Mean output tokens")
    axes[1].set_xticks(range(len(rows)), labels)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Phase 1.5 fixed-context credit-allocation pilot")
    figure.tight_layout()
    png, pdf = figure_dir / "credit_pilot.png", figure_dir / "credit_pilot.pdf"
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
