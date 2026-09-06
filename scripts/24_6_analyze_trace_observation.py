#!/usr/bin/env python3
"""Analyze Phase-0 accuracy, length, training accounting, and Gate 0."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import file_sha256
from trace_length_observation.trace_observation import (
    BUDGET_REGIMES,
    LENGTH_RANKS,
    LOSS_NORMALIZATIONS,
    protocol_hash,
    validate_trace_observation_config,
)
from trace_length_observation.trace_observation_analysis import (
    analyze_cells,
    attach_training_accounting,
    decide_gate0,
    index_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    evaluation_path = _resolve(args.evaluation_manifest)
    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    if output_dir.exists() or figure_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite Phase-0 analysis or figures: {output_dir}, {figure_dir}"
        )
    config = _read_json(config_path)
    validate_trace_observation_config(config)
    config_hash = protocol_hash(config)
    evaluation = _read_json(evaluation_path)
    if evaluation.get("status") != "complete" or evaluation.get("config_hash") != config_hash:
        raise ValueError("Evaluation manifest is incomplete or bound to another protocol.")
    evaluation_marker = evaluation_path.parent / "EVALUATION_COMPLETE"
    if not evaluation_marker.is_file():
        raise FileNotFoundError(evaluation_marker)
    training_audit_path = _resolve(str(evaluation["training_audit_path"]))
    training_audit = _read_json(training_audit_path)
    if training_audit.get("status") != "passed":
        raise ValueError("Training audit did not pass.")
    predictions = {
        str(run["model_id"]): _read_jsonl(_resolve(str(run["prediction_path"])))
        for run in evaluation["runs"]
    }
    indexed, base_metrics = index_predictions(evaluation["runs"], predictions)
    analysis_config = dict(config["analysis"])
    arm_rows, contrast_rows = analyze_cells(
        indexed,
        rank_contrasts=analysis_config["rank_contrasts"],
        bootstrap_samples=int(analysis_config["bootstrap_samples"]),
        bootstrap_seed=int(analysis_config["bootstrap_seed"]),
    )
    attach_training_accounting(arm_rows, training_audit["runs"])
    gate0 = decide_gate0(contrast_rows, config)
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    arms_path = output_dir / "arm_metrics.csv"
    contrasts_path = output_dir / "paired_rank_contrasts.csv"
    analysis_path = output_dir / "trace_observation_analysis.json"
    gate_path = output_dir / "gate0_decision.json"
    report_path = output_dir / "experiment_report.md"
    _write_csv(arms_path, arm_rows)
    _write_csv(contrasts_path, contrast_rows)
    _write_json(gate_path, gate0)
    analysis_payload = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "evaluation_manifest_path": str(evaluation_path),
        "evaluation_manifest_sha256": file_sha256(evaluation_path),
        "training_audit_path": str(training_audit_path),
        "training_audit_sha256": file_sha256(training_audit_path),
        "evidence_level": config["evidence_level"],
        "formal_claim_allowed": False,
        "source_hashes": {
            "analysis": file_sha256(
                SRC_ROOT / "trace_length_observation/trace_observation_analysis.py"
            ),
            "entrypoint": file_sha256(Path(__file__).resolve()),
        },
        "base_metrics": base_metrics,
        "arm_count": len(arm_rows),
        "contrast_count": len(contrast_rows),
        "arm_metrics": arm_rows,
        "paired_rank_contrasts": contrast_rows,
        "gate0": gate0,
    }
    _write_json(analysis_path, analysis_payload)
    figure_paths = _plot_controls(arm_rows, figure_dir)
    figure_paths.extend(_plot_training_accounting(arm_rows, figure_dir))
    _write_report(report_path, config, base_metrics, arm_rows, gate0)
    artifacts = [arms_path, contrasts_path, analysis_path, gate_path, report_path, *figure_paths]
    artifact_manifest = {
        "status": "complete",
        "config_hash": config_hash,
        "artifacts": [
            {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    _write_json(artifact_manifest_path, artifact_manifest)
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        "status=complete\n"
        f"config_hash={config_hash}\n"
        f"analysis_sha256={file_sha256(analysis_path)}\n"
        f"gate0_decision_sha256={file_sha256(gate_path)}\n"
        f"artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n"
        f"gate0_status={gate0['status']}\n",
        encoding="utf-8",
    )


def _plot_controls(rows: List[Mapping[str, Any]], figure_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in rows if row["loss_mask"] == "full_completion"]
    colors = {"token_mean": "#0072B2", "sequence_mean": "#D55E00"}
    rank_x = list(range(len(LENGTH_RANKS)))
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex="col")
    for column, regime in enumerate(BUDGET_REGIMES):
        for normalization in LOSS_NORMALIZATIONS:
            selected = {
                str(row["length_rank"]): row
                for row in primary
                if row["budget_regime"] == regime
                and row["loss_normalization"] == normalization
            }
            accuracy = [100.0 * float(selected[rank]["mean_accuracy"]) for rank in LENGTH_RANKS]
            lower = [
                100.0
                * (float(selected[rank]["mean_accuracy"]) - float(selected[rank]["accuracy_ci_low"]))
                for rank in LENGTH_RANKS
            ]
            upper = [
                100.0
                * (float(selected[rank]["accuracy_ci_high"]) - float(selected[rank]["mean_accuracy"]))
                for rank in LENGTH_RANKS
            ]
            axes[0, column].errorbar(
                rank_x,
                accuracy,
                yerr=[lower, upper],
                marker="o",
                capsize=3,
                color=colors[normalization],
                label=normalization.replace("_", " "),
            )
            axes[1, column].plot(
                rank_x,
                [float(selected[rank]["mean_output_tokens"]) for rank in LENGTH_RANKS],
                marker="o",
                color=colors[normalization],
                label=normalization.replace("_", " "),
            )
        axes[0, column].set_title(regime.replace("_", " "))
        axes[1, column].set_xticks(rank_x, [rank.title() for rank in LENGTH_RANKS])
        for row_index in range(2):
            axes[row_index, column].grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("GSM8K exact-match accuracy (%)")
    axes[1, 0].set_ylabel("Mean student output tokens")
    axes[0, 2].legend(frameon=False)
    figure.suptitle("Phase 0: trace-length observation under budget and loss controls")
    figure.tight_layout()
    png = figure_dir / "phase0_accuracy_length_controls.png"
    pdf = figure_dir / "phase0_accuracy_length_controls.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _plot_training_accounting(rows: List[Mapping[str, Any]], figure_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Most accounting is invariant to seed; padding-inclusive tokens are averaged.
    selected = [
        row
        for row in rows
        if row["loss_normalization"] == "token_mean" and row["loss_mask"] == "full_completion"
    ]
    by_key = {(row["budget_regime"], row["length_rank"]): row for row in selected}
    colors = {"short": "#4C78A8", "medium": "#F2A541", "long": "#C44E52"}
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes = axes.ravel()
    width = 0.22
    x = list(range(len(BUDGET_REGIMES)))
    for offset, rank in enumerate(LENGTH_RANKS):
        positions = [value + (offset - 1) * width for value in x]
        axes[0].bar(
            positions,
            [int(by_key[(regime, rank)]["completion_token_updates"]) for regime in BUDGET_REGIMES],
            width=width,
            color=colors[rank],
            label=rank.title(),
        )
        axes[1].bar(
            positions,
            [int(by_key[(regime, rank)]["model_input_token_updates"]) for regime in BUDGET_REGIMES],
            width=width,
            color=colors[rank],
            label=rank.title(),
        )
        axes[2].bar(
            positions,
            [
                float(by_key[(regime, rank)]["mean_padded_model_token_updates"])
                for regime in BUDGET_REGIMES
            ],
            width=width,
            color=colors[rank],
            label=rank.title(),
        )
        axes[3].bar(
            positions,
            [int(by_key[(regime, rank)]["optimizer_steps"]) for regime in BUDGET_REGIMES],
            width=width,
            color=colors[rank],
            label=rank.title(),
        )
    labels = [value.replace("_", " ") for value in BUDGET_REGIMES]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Completion-token updates")
    axes[1].set_ylabel("Non-padding model-input tokens")
    axes[2].set_ylabel("Mean padding-inclusive tokens")
    axes[3].set_ylabel("Optimizer steps")
    axes[0].set_title("Completion supervision")
    axes[1].set_title("Matched token-update exposure")
    axes[2].set_title("Padding-inclusive compute diagnostic")
    axes[3].set_title("Optimization exposure")
    axes[3].legend(frameon=False)
    figure.suptitle("Phase-0 training-budget audit")
    figure.tight_layout()
    png = figure_dir / "phase0_training_budget_audit.png"
    pdf = figure_dir / "phase0_training_budget_audit.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _write_report(
    path: Path,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    arm_rows: List[Mapping[str, Any]],
    gate0: Mapping[str, Any],
) -> None:
    lines = [
        "# Phase-0 Controlled Trace-Length Observation",
        "",
        "## Evidence scope",
        "",
        "- Teacher: Qwen2.5-7B-Instruct; student: Qwen2.5-1.5B-Instruct.",
        "- Dataset scope: GSM8K only.",
        "- The evaluation cohort is locked but was previously observed, so this is comparative Gate-0 evidence rather than a fresh confirmatory test.",
        "- The matrix explicitly crosses fairness budget, loss normalization, answer masking, trace rank, and three training seeds.",
        "",
        "## Gate 0",
        "",
        f"Decision: `{gate0['decision']}` (`{gate0['status']}`).",
        "",
        "| Budget regime | Loss normalization | Short-long effect | 95% interval | Holm p | Passed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in gate0["required_cells"]:
        lines.append(
            f"| {row['budget_regime']} | {row['loss_normalization']} | "
            f"{100 * float(row['estimate']):+.2f} pp | "
            f"[{100 * float(row['ci_low']):+.2f}, {100 * float(row['ci_high']):+.2f}] pp | "
            f"{float(row['bootstrap_holm_p_value']):.4g} | {'yes' if row['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Primary controlled arms",
            "",
            "The compact table below reports full-completion loss. Answer-only diagnostics are preserved in `arm_metrics.csv`.",
            "",
            "| Budget | Normalization | Rank | Accuracy | 95% interval | Output tokens | Completion tokens | Model-input tokens | Optimizer steps |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in arm_rows:
        if row["loss_mask"] != "full_completion":
            continue
        lines.append(
            f"| {row['budget_regime']} | {row['loss_normalization']} | {row['length_rank']} | "
            f"{100 * float(row['mean_accuracy']):.2f}% | "
            f"[{100 * float(row['accuracy_ci_low']):.2f}, {100 * float(row['accuracy_ci_high']):.2f}]% | "
            f"{float(row['mean_output_tokens']):.1f} | {int(row['completion_token_updates'])} | "
            f"{int(row['model_input_token_updates'])} | "
            f"{int(row['optimizer_steps'])} |"
        )
    lines.extend(
        [
            "",
            f"Base student accuracy: {100 * float(base['accuracy']):.2f}%; mean output length: {float(base['mean_output_tokens']):.1f} tokens.",
            "",
            "## Interpretation boundary",
            "",
            "Passing Gate 0 permits utility validation in Phase 1. It does not establish a latent SAE mechanism. Failing Gate 0 stops the SAE story and attributes the original observation to budget or loss-normalization sensitivity until stronger evidence exists.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
