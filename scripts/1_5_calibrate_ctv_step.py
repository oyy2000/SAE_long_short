#!/usr/bin/env python3
"""Calibrate one fixed CTV micro-update step for token-sum and token-mean gradients."""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.teaching_utility import (
    first_order_ctv,
    gradient_for_completion,
    gradient_norm,
    load_lora_student,
    mean_probe_gradient,
    mean_probe_loss,
    normalized_linearization_error,
    reversible_micro_update,
    tokenize_prompt_completion,
    trainable_state_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--input-dir", default="results/phase1_teaching_utility_v1/formal/ctv_inputs"
    )
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/calibration"
    )
    parser.add_argument("--anchor-adapter", default=None)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    model, tokenizer, trainable, model_evidence = load_lora_student(
        config["student"], adapter_path=args.anchor_adapter
    )
    max_length = int(config["utility"]["max_length"])
    probes = [
        tokenize_prompt_completion(
            tokenizer,
            record_id=row["problem_id"],
            prompt=row["student_prompt"],
            completion=row["official_completion"],
            max_length=max_length,
        )
        for row in read_jsonl(input_dir / "global_probes.jsonl")
    ]
    candidates = list(read_jsonl(input_dir / "calibration_candidates.jsonl"))
    probe_gradient, base_probe_loss = mean_probe_gradient(model, trainable, probes)
    no_update_losses = [
        mean_probe_loss(model, probes)
        for _ in range(int(config["calibration"]["no_update_repeat_forwards"]))
    ]
    no_update_range = max(no_update_losses) - min(no_update_losses)
    no_update_tolerance = float(config["calibration"]["no_update_max_absolute_range"])
    if no_update_range > no_update_tolerance:
        raise RuntimeError(
            f"No-update probe forwards exceeded tolerance: {no_update_range} > {no_update_tolerance}."
        )
    parameter_norm = math.sqrt(
        sum(
            float((parameter.detach().double() ** 2).sum().cpu())
            for _, parameter in trainable
        )
    )
    mean_gradient_norms = []
    for row in candidates:
        encoded = _encode_candidate(tokenizer, row, max_length)
        result = gradient_for_completion(
            model, trainable, encoded, reduction="token_mean"
        )
        mean_gradient_norms.append(float(result["gradient_norm"]))
        del result
    median_mean_gradient_norm = statistics.median(mean_gradient_norms)
    relative_grid = [
        float(value) for value in config["calibration"]["relative_update_grid"]
    ]
    eta_grid = {
        relative: relative * parameter_norm / max(median_mean_gradient_norm, 1e-30)
        for relative in relative_grid
    }
    raw_rows = []
    for candidate_index, row in enumerate(candidates):
        encoded = _encode_candidate(tokenizer, row, max_length)
        mean_result = gradient_for_completion(
            model, trainable, encoded, reduction="token_mean"
        )
        token_count = int(mean_result["token_count"])
        mean_gradients = mean_result.pop("gradients")
        sum_gradients = [gradient * token_count for gradient in mean_gradients]
        reductions = {
            "token_mean": (
                mean_gradients,
                float(mean_result["loss_mean"]),
                gradient_norm(mean_gradients),
            ),
            "token_sum": (
                sum_gradients,
                float(mean_result["loss_sum"]),
                gradient_norm(sum_gradients),
            ),
        }
        identity_error = abs(
            reductions["token_sum"][2] - token_count * reductions["token_mean"][2]
        )
        for reduction, (gradients, trace_loss, grad_norm) in reductions.items():
            dot_score = first_order_ctv(probe_gradient, gradients, eta=1.0)
            for relative in relative_grid:
                eta = eta_grid[relative]
                exact = reversible_micro_update(
                    model,
                    trainable,
                    gradients,
                    eta=eta,
                    probe=probes,
                    base_probe_loss=base_probe_loss,
                )
                predicted = eta * dot_score
                raw_rows.append(
                    {
                        "record_id": row["trace_id"],
                        "problem_id": row["problem_id"],
                        "candidate_role": row["candidate_role"],
                        "loss_reduction": reduction,
                        "relative_update": relative,
                        "eta": eta,
                        "trace_loss": trace_loss,
                        "trace_token_count": token_count,
                        "gradient_norm": grad_norm,
                        "gradient_sum_mean_identity_error": identity_error,
                        "ctv_predicted": predicted,
                        "ctv_exact": exact["ctv_exact"],
                        "linearization_error": normalized_linearization_error(
                            exact["ctv_exact"], predicted
                        ),
                        "parameters_restored_exactly": exact[
                            "parameters_restored_exactly"
                        ],
                    }
                )
        del mean_gradients, sum_gradients, mean_result
        if candidate_index % 8 == 0:
            gc.collect()
    summaries = _summarize(raw_rows, relative_grid)
    passing_relative = []
    thresholds = config["calibration"]
    for relative in relative_grid:
        cells = [row for row in summaries if row["relative_update"] == relative]
        if len(cells) == 2 and all(
            row["sign_agreement"] >= float(thresholds["minimum_sign_agreement"])
            and row["adjacent_spearman"] is not None
            and row["adjacent_spearman"]
            >= float(thresholds["minimum_adjacent_spearman"])
            and row["median_linearization_error"]
            <= float(thresholds["maximum_median_linearization_error"])
            for row in cells
        ):
            passing_relative.append(relative)
    if not passing_relative:
        raise RuntimeError("No single eta passed calibration for both loss reductions.")
    selected_relative = max(passing_relative)
    selected_eta = eta_grid[selected_relative]
    selected_mean_exact = {
        str(row["record_id"]): float(row["ctv_exact"])
        for row in raw_rows
        if row["loss_reduction"] == "token_mean"
        and row["relative_update"] == selected_relative
    }
    norm_matched_rows = []
    for candidate_index, row in enumerate(candidates):
        encoded = _encode_candidate(tokenizer, row, max_length)
        result = gradient_for_completion(
            model, trainable, encoded, reduction="token_mean"
        )
        token_count = int(result["token_count"])
        mean_gradients = result.pop("gradients")
        sum_gradients = [gradient * token_count for gradient in mean_gradients]
        matched_eta = selected_eta / token_count
        exact = reversible_micro_update(
            model,
            trainable,
            sum_gradients,
            eta=matched_eta,
            probe=probes,
            base_probe_loss=base_probe_loss,
        )
        reference = selected_mean_exact[str(row["trace_id"])]
        norm_matched_rows.append(
            {
                "record_id": row["trace_id"],
                "problem_id": row["problem_id"],
                "candidate_role": row["candidate_role"],
                "trace_token_count": token_count,
                "token_mean_eta": selected_eta,
                "norm_matched_token_sum_eta": matched_eta,
                "token_mean_exact_ctv": reference,
                "norm_matched_token_sum_exact_ctv": exact["ctv_exact"],
                "absolute_ctv_difference": abs(exact["ctv_exact"] - reference),
                "parameters_restored_exactly": exact["parameters_restored_exactly"],
            }
        )
        del mean_gradients, sum_gradients, result
        if candidate_index % 8 == 0:
            gc.collect()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "calibration_micro_updates.jsonl"
    write_jsonl(rows_path, raw_rows)
    norm_matched_path = output_dir / "norm_matched_token_sum_sanity.jsonl"
    write_jsonl(norm_matched_path, norm_matched_rows)
    report = {
        "status": "passed",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "anchor_adapter": args.anchor_adapter,
        "model_evidence": model_evidence,
        "base_global_probe_loss": base_probe_loss,
        "no_update_probe_losses": no_update_losses,
        "no_update_probe_loss_range": no_update_range,
        "no_update_max_absolute_range": no_update_tolerance,
        "parameter_norm": parameter_norm,
        "median_token_mean_gradient_norm": median_mean_gradient_norm,
        "eta_grid": {str(key): value for key, value in eta_grid.items()},
        "summaries": summaries,
        "selected_relative_update": selected_relative,
        "selected_eta": selected_eta,
        "same_eta_across_reductions": True,
        "norm_matched_update_sanity_only": {
            "status": "complete",
            "row_count": len(norm_matched_rows),
            "maximum_absolute_ctv_difference": max(
                row["absolute_ctv_difference"] for row in norm_matched_rows
            ),
            "median_absolute_ctv_difference": statistics.median(
                row["absolute_ctv_difference"] for row in norm_matched_rows
            ),
            "path": str(norm_matched_path),
            "sha256": file_sha256(norm_matched_path),
            "used_for_gate": False,
        },
        "calibration_rows_path": str(rows_path),
        "calibration_rows_sha256": file_sha256(rows_path),
        "final_trainable_state_sha256": trainable_state_sha256(trainable),
        "runtime": runtime_metadata(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "teaching_utility_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/teaching_utility.py"
        ),
        "formal_claim_allowed": False,
    }
    report_path = output_dir / "calibration_report.json"
    write_json_exclusive(report_path, report)
    (output_dir / "CALIBRATION_COMPLETE").write_text(
        f"status=passed\nconfig_hash={report['config_hash']}\nreport_sha256={file_sha256(report_path)}\nselected_eta={selected_eta}\n",
        encoding="utf-8",
    )


def _encode_candidate(tokenizer, row, max_length):
    return tokenize_prompt_completion(
        tokenizer,
        record_id=row["trace_id"],
        prompt=row["student_prompt"],
        completion=row["solution"],
        max_length=max_length,
    )


def _summarize(rows, relative_grid):
    from scipy.stats import spearmanr

    summaries = []
    for reduction in ("token_mean", "token_sum"):
        previous = None
        for relative in relative_grid:
            subset = [
                row
                for row in rows
                if row["loss_reduction"] == reduction
                and row["relative_update"] == relative
            ]
            actual = [float(row["ctv_exact"]) for row in subset]
            predicted = [float(row["ctv_predicted"]) for row in subset]
            adjacent = None
            if previous is not None:
                statistic = float(spearmanr(previous, actual).statistic)
                adjacent = statistic if math.isfinite(statistic) else None
            summaries.append(
                {
                    "loss_reduction": reduction,
                    "relative_update": relative,
                    "row_count": len(subset),
                    "sign_agreement": sum(
                        (a >= 0) == (p >= 0) for a, p in zip(actual, predicted)
                    )
                    / len(subset),
                    "median_linearization_error": statistics.median(
                        float(row["linearization_error"]) for row in subset
                    ),
                    "adjacent_spearman": adjacent,
                }
            )
            previous = actual
    return summaries


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
