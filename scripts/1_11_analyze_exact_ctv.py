#!/usr/bin/env python3
"""Merge exact CTV shards and test CTV against registered surface baselines."""

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
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.teaching_utility import (
    within_question_fixed_effect_regression,
)
from length_budget_distill.utility_analysis import compare_rank_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--exact-root", required=True)
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/ctv_analysis"
    )
    parser.add_argument(
        "--figure-dir", default="figures/phase1_teaching_utility_v1/ctv_analysis"
    )
    parser.add_argument("--expected-question-count", type=int, default=None)
    parser.add_argument("--analysis-label", default="base_anchor")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    if output_dir.exists() or figure_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {figure_dir}")
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    score_rows, score_manifests = _load_shards(
        _resolve(args.score_root), "score_manifest.json", "score_path", "score_sha256"
    )
    exact_rows, exact_manifests = _load_shards(
        _resolve(args.exact_root), "exact_ctv_manifest.json", "rows_path", "rows_sha256"
    )
    scores = {str(row["trace_id"]): dict(row) for row in score_rows}
    if len(scores) != len(score_rows):
        raise ValueError("Duplicate trace IDs in gradient-score shards.")
    merged = []
    exact_keys = set()
    for source in exact_rows:
        key = (str(source["trace_id"]), str(source["loss_reduction"]))
        if key in exact_keys:
            raise ValueError(f"Duplicate exact CTV row: {key}")
        exact_keys.add(key)
        if key[0] not in scores:
            raise ValueError(f"Exact CTV trace is absent from score shards: {key[0]}")
        score = scores[key[0]]
        row = dict(source)
        for field in (
            "student_nll_mean",
            "rsr_rank_clip_100",
            "scas_score",
            "lark_g_hat",
        ):
            row[field] = score[field]
        row["negative_log_length"] = -float(row["log_solution_token_count"])
        row["negative_student_nll"] = -float(row["student_nll_mean"])
        row["negative_rsr"] = -float(row["rsr_rank_clip_100"])
        row["negative_scas"] = -float(row["scas_score"])
        merged.append(row)
    expected_questions = args.expected_question_count or int(
        config["exact_utility"]["question_count"]
    )
    expected = (
        expected_questions * int(config["exact_utility"]["candidates_per_question"]) * 2
    )
    if len(merged) != expected:
        raise ValueError(f"Exact CTV coverage mismatch: {len(merged)} != {expected}")
    if len({row["problem_id"] for row in merged}) != expected_questions:
        raise ValueError("Exact CTV question support is incomplete.")
    metrics = {
        "ctv_gradient": "ctv_gradient_global",
        "length": "negative_log_length",
        "student_nll": "negative_student_nll",
        "rsr": "negative_rsr",
        "scas": "negative_scas",
        "lark_style": "lark_g_hat",
    }
    baselines = list(config["gate1"]["surface_baselines"])
    all_candidate_gradient_regressions = {}
    for reduction in config["utility"]["candidate_loss_reductions"]:
        all_candidate_gradient_regressions[reduction] = {
            "global": _utility_regressions(
                score_rows,
                outcome=f"ctv_gradient_global_{reduction}",
            ),
            "local": _utility_regressions(
                score_rows,
                outcome=f"ctv_gradient_local_{reduction}",
            ),
        }
    analyses = {}
    regressions = {}
    local_analyses = {}
    local_regressions = {}
    answer_analyses = {}
    answer_regressions = {}
    local_answer_analyses = {}
    local_answer_regressions = {}
    for reduction_index, reduction in enumerate(
        config["utility"]["candidate_loss_reductions"]
    ):
        rows = [row for row in merged if row["loss_reduction"] == reduction]
        analyses[reduction] = compare_rank_metrics(
            rows,
            outcome="ctv_exact_global",
            metrics=metrics,
            primary_metric="ctv_gradient",
            baseline_metrics=baselines,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"]) + reduction_index * 100_003,
        )
        regressions[reduction] = {
            "length_only": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_global",
                predictors=["log_solution_token_count"],
            ),
            "nll_only": within_question_fixed_effect_regression(
                rows, outcome="ctv_exact_global", predictors=["student_nll_mean"]
            ),
            "length_and_nll": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_global",
                predictors=["log_solution_token_count", "student_nll_mean"],
            ),
        }
        local_metrics = {
            **metrics,
            "ctv_gradient": "ctv_gradient_local",
        }
        local_analyses[reduction] = compare_rank_metrics(
            rows,
            outcome="ctv_exact_local",
            metrics=local_metrics,
            primary_metric="ctv_gradient",
            baseline_metrics=baselines,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"])
            + 400_009
            + reduction_index * 100_003,
        )
        local_regressions[reduction] = {
            "length_only": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_local",
                predictors=["log_solution_token_count"],
            ),
            "nll_only": within_question_fixed_effect_regression(
                rows, outcome="ctv_exact_local", predictors=["student_nll_mean"]
            ),
            "length_and_nll": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_local",
                predictors=["log_solution_token_count", "student_nll_mean"],
            ),
        }
        answer_metrics = {
            **metrics,
            "ctv_gradient": "ctv_gradient_global_answer",
        }
        answer_analyses[reduction] = compare_rank_metrics(
            rows,
            outcome="ctv_exact_global_answer",
            metrics=answer_metrics,
            primary_metric="ctv_gradient",
            baseline_metrics=baselines,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"])
            + 900_001
            + reduction_index * 100_003,
        )
        answer_regressions[reduction] = {
            "length_only": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_global_answer",
                predictors=["log_solution_token_count"],
            ),
            "nll_only": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_global_answer",
                predictors=["student_nll_mean"],
            ),
            "length_and_nll": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_global_answer",
                predictors=["log_solution_token_count", "student_nll_mean"],
            ),
        }
        local_answer_metrics = {
            **metrics,
            "ctv_gradient": "ctv_gradient_local_answer",
        }
        local_answer_analyses[reduction] = compare_rank_metrics(
            rows,
            outcome="ctv_exact_local_answer",
            metrics=local_answer_metrics,
            primary_metric="ctv_gradient",
            baseline_metrics=baselines,
            samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["bootstrap_seed"])
            + 1_300_007
            + reduction_index * 100_003,
        )
        local_answer_regressions[reduction] = {
            "length_only": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_local_answer",
                predictors=["log_solution_token_count"],
            ),
            "nll_only": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_local_answer",
                predictors=["student_nll_mean"],
            ),
            "length_and_nll": within_question_fixed_effect_regression(
                rows,
                outcome="ctv_exact_local_answer",
                predictors=["log_solution_token_count", "student_nll_mean"],
            ),
        }
    primary = str(config["gate1"]["primary_loss_reduction"])
    gradient_gate_passed = all(
        row["passed"] for row in analyses[primary]["paired_contrasts"]
    )
    gradient_gate = {
        "status": "passed" if gradient_gate_passed else "failed",
        "primary_loss_reduction": primary,
        "criterion": "CTV-gradient within-question Spearman exceeds every registered surface baseline with positive paired bootstrap interval and Holm p < 0.05",
        "surface_baselines": baselines,
        "paired_contrasts": analyses[primary]["paired_contrasts"],
        "full_gate1_status": "pending_policy_sft",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    merged_path = output_dir / "exact_ctv_merged.jsonl"
    write_jsonl(merged_path, merged)
    correlations_path = output_dir / "within_question_correlations.csv"
    local_correlations_path = output_dir / "local_within_question_correlations.csv"
    regression_path = output_dir / "fixed_effect_regressions.json"
    analysis_path = output_dir / "ctv_analysis.json"
    gate_path = output_dir / "ctv_gradient_gate.json"
    _write_correlation_csv(correlations_path, analyses)
    _write_correlation_csv(local_correlations_path, local_analyses)
    write_json_exclusive(
        regression_path,
        {
            "global_rationale": regressions,
            "local_rationale": local_regressions,
            "global_answer_only": answer_regressions,
            "local_answer_only": local_answer_regressions,
        },
    )
    write_json_exclusive(gate_path, gradient_gate)
    figures = _plot(merged, analyses, primary, figure_dir, scope="global")
    figures.extend(_plot(merged, local_analyses, primary, figure_dir, scope="local"))
    payload = {
        "status": "complete",
        "analysis_label": args.analysis_label,
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "score_manifests": score_manifests,
        "exact_manifests": exact_manifests,
        "row_count": len(merged),
        "question_count": len({row["problem_id"] for row in merged}),
        "analyses": analyses,
        "regressions": regressions,
        "all_candidate_gradient_regressions": all_candidate_gradient_regressions,
        "local_probe_analyses": local_analyses,
        "local_probe_regressions": local_regressions,
        "answer_only_probe_analyses": answer_analyses,
        "answer_only_probe_regressions": answer_regressions,
        "local_answer_only_probe_analyses": local_answer_analyses,
        "local_answer_only_probe_regressions": local_answer_regressions,
        "gradient_gate": gradient_gate,
        "artifacts": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (
                merged_path,
                correlations_path,
                local_correlations_path,
                regression_path,
                gate_path,
                *figures,
            )
        ],
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "analysis_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/utility_analysis.py"
        ),
        "formal_claim_allowed": False,
    }
    write_json_exclusive(analysis_path, payload)
    (output_dir / "CTV_ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={payload['config_hash']}\nanalysis_sha256={file_sha256(analysis_path)}\n"
        f"gradient_gate_status={gradient_gate['status']}\nfull_gate1_status=pending_policy_sft\n",
        encoding="utf-8",
    )


def _load_shards(root: Path, manifest_name: str, path_field: str, hash_field: str):
    rows = []
    evidence = []
    for manifest_path in sorted(root.glob(f"shard_*/{manifest_name}")):
        manifest = read_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete shard manifest: {manifest_path}")
        data_path = Path(str(manifest[path_field]))
        if file_sha256(data_path) != manifest[hash_field]:
            raise ValueError(f"Shard hash mismatch: {data_path}")
        rows.extend(dict(row) for row in read_jsonl(data_path))
        evidence.append(
            {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}
        )
    if not evidence:
        raise FileNotFoundError(f"No {manifest_name} files under {root}")
    return rows, evidence


def _write_correlation_csv(path: Path, analyses) -> None:
    fields = [
        "loss_reduction",
        "metric",
        "score_field",
        "mean_within_question_spearman",
        "question_count",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for reduction, result in analyses.items():
            for row in result["metric_summaries"]:
                writer.writerow({"loss_reduction": reduction, **row})


def _utility_regressions(rows, *, outcome):
    return {
        "length_only": within_question_fixed_effect_regression(
            rows,
            outcome=outcome,
            predictors=["log_solution_token_count"],
        ),
        "nll_only": within_question_fixed_effect_regression(
            rows,
            outcome=outcome,
            predictors=["student_nll_mean"],
        ),
        "length_and_nll": within_question_fixed_effect_regression(
            rows,
            outcome=outcome,
            predictors=["log_solution_token_count", "student_nll_mean"],
        ),
    }


def _plot(rows, analyses, primary, figure_dir: Path, *, scope: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["loss_reduction"] == primary]
    if scope not in {"global", "local"}:
        raise ValueError(f"Unknown probe scope: {scope}")
    gradient_field = f"ctv_gradient_{scope}"
    exact_field = f"ctv_exact_{scope}"
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].scatter(
        [float(row[gradient_field]) for row in selected],
        [float(row[exact_field]) for row in selected],
        s=10,
        alpha=0.35,
        color="#0072B2",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].axvline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel(f"First-order {scope} CTV")
    axes[0].set_ylabel(f"Exact one-step {scope} CTV")
    summary = analyses[primary]["metric_summaries"]
    axes[1].bar(
        range(len(summary)),
        [float(row["mean_within_question_spearman"]) for row in summary],
        color=[
            "#0072B2" if row["metric"] == "ctv_gradient" else "#999999"
            for row in summary
        ],
    )
    axes[1].set_xticks(
        range(len(summary)), [row["metric"].replace("_", "\n") for row in summary]
    )
    axes[1].set_ylabel("Mean within-question Spearman")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Phase 1 {scope} CTV validation ({primary.replace('_', ' ')})")
    figure.tight_layout()
    suffix = "" if scope == "global" else f"_{scope}"
    png = figure_dir / f"ctv_exact_validation{suffix}.png"
    pdf = figure_dir / f"ctv_exact_validation{suffix}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
