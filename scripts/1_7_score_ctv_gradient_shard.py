#!/usr/bin/env python3
"""Score all correct traces in one shard by global/local first-order CTV and surface metrics."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    validated_artifact_marker,
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
    load_lora_student,
    mean_probe_gradient,
    scas_forward_metrics,
    tokenize_prompt_completion,
    trainable_state_sha256,
)
from length_budget_distill.trace_baselines import lark_g_hat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument("--candidate-shard", required=True)
    parser.add_argument(
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument(
        "--local-probe-map",
        default="results/phase1_teaching_utility_v1/formal/local_probes/local_probe_map.jsonl",
    )
    parser.add_argument(
        "--calibration-report",
        default="results/phase1_teaching_utility_v1/formal/calibration/calibration_report.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-adapter", default=None)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    candidate_path = _resolve(args.candidate_shard)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    calibration_path = _resolve(args.calibration_report)
    calibration = read_json(calibration_path)
    calibration_evidence = validated_artifact_marker(
        calibration_path.parent / "CALIBRATION_COMPLETE",
        expected_status="passed",
        hash_bindings={"report_sha256": calibration_path},
    )
    if calibration.get("status") != "passed":
        raise ValueError("CTV calibration did not pass.")
    eta = float(calibration["selected_eta"])
    model, tokenizer, trainable, model_evidence = load_lora_student(
        config["student"], adapter_path=args.anchor_adapter
    )
    max_length = int(config["utility"]["max_length"])
    questions_dir = _resolve(args.questions_dir)
    probe_rows = {
        row["problem_id"]: row
        for row in read_jsonl(questions_dir / "global_probe_questions.jsonl")
    }
    encoded_probes = {
        problem_id: tokenize_prompt_completion(
            tokenizer,
            record_id=problem_id,
            prompt=row["student_prompt"],
            completion=row["official_completion"],
            max_length=max_length,
        )
        for problem_id, row in probe_rows.items()
    }
    global_probe = [encoded_probes[key] for key in sorted(encoded_probes)]
    global_gradient, global_loss = mean_probe_gradient(model, trainable, global_probe)
    global_answer_gradient, global_answer_loss = mean_probe_gradient(
        model, trainable, global_probe, answer_only=True
    )
    local_map_path = _resolve(args.local_probe_map)
    local_report_path = local_map_path.parent / "local_probe_report.json"
    local_report = read_json(local_report_path)
    local_probe_evidence = validated_artifact_marker(
        local_map_path.parent / "LOCAL_PROBES_COMPLETE",
        expected_status="complete",
        hash_bindings={"report_sha256": local_report_path},
    )
    if local_report.get("local_probe_map_sha256") != file_sha256(local_map_path):
        raise ValueError("Local-probe map hash differs from its report.")
    local_map = {
        row["problem_id"]: list(row["local_probe_problem_ids"])
        for row in read_jsonl(local_map_path)
    }
    grouped = defaultdict(list)
    for row in read_jsonl(candidate_path):
        grouped[row["problem_id"]].append(dict(row))
    output_rows = []
    scas_config = config["baselines"]["scas"]
    scas_layer_suffix = str(scas_config["target_layer"])
    scas_weight = float(scas_config["lambda"])
    for question_index, problem_id in enumerate(sorted(grouped)):
        question_output_start = len(output_rows)
        local_probe = [encoded_probes[value] for value in local_map[problem_id]]
        local_gradient, local_loss = mean_probe_gradient(model, trainable, local_probe)
        local_answer_gradient, local_answer_loss = mean_probe_gradient(
            model, trainable, local_probe, answer_only=True
        )
        for row in grouped[problem_id]:
            encoded = tokenize_prompt_completion(
                tokenizer,
                record_id=row["trace_id"],
                prompt=row["student_prompt"],
                completion=row["solution"],
                max_length=max_length,
            )
            mean_result = gradient_for_completion(
                model, trainable, encoded, reduction="token_mean"
            )
            scas = scas_forward_metrics(
                model,
                tokenizer,
                encoded,
                layer_suffix=scas_layer_suffix,
                weight=scas_weight,
            )
            mean_gradients = mean_result.pop("gradients")
            gradient_target_token_count = int(mean_result["token_count"])
            raw_solution_token_count = int(row["solution_token_count"])
            sum_gradients = [
                gradient * gradient_target_token_count for gradient in mean_gradients
            ]
            output_rows.append(
                {
                    "trace_id": row["trace_id"],
                    "problem_id": problem_id,
                    "question_split": row["question_split"],
                    "solution_token_count": raw_solution_token_count,
                    "gradient_target_token_count": gradient_target_token_count,
                    "log_solution_token_count": math.log(raw_solution_token_count),
                    "student_nll_mean": mean_result["loss_mean"],
                    "student_nll_sum": mean_result["loss_sum"],
                    "student_answer_nll_mean": mean_result["answer_loss_mean"],
                    "mean_token_rank": mean_result["mean_token_rank"],
                    "rsr_rank_clip_100": mean_result["rsr_rank_clip_100"],
                    "lark_brier_top50": mean_result["brier_top50"],
                    **scas,
                    "gradient_norm_token_mean": mean_result["gradient_norm"],
                    "gradient_norm_token_sum": mean_result["gradient_norm"]
                    * gradient_target_token_count,
                    "ctv_gradient_global_token_mean": first_order_ctv(
                        global_gradient, mean_gradients, eta
                    ),
                    "ctv_gradient_local_token_mean": first_order_ctv(
                        local_gradient, mean_gradients, eta
                    ),
                    "ctv_gradient_global_token_sum": first_order_ctv(
                        global_gradient, sum_gradients, eta
                    ),
                    "ctv_gradient_local_token_sum": first_order_ctv(
                        local_gradient, sum_gradients, eta
                    ),
                    "ctv_gradient_global_answer_token_mean": first_order_ctv(
                        global_answer_gradient, mean_gradients, eta
                    ),
                    "ctv_gradient_local_answer_token_mean": first_order_ctv(
                        local_answer_gradient, mean_gradients, eta
                    ),
                    "ctv_gradient_global_answer_token_sum": first_order_ctv(
                        global_answer_gradient, sum_gradients, eta
                    ),
                    "ctv_gradient_local_answer_token_sum": first_order_ctv(
                        local_answer_gradient, sum_gradients, eta
                    ),
                    "global_probe_loss": global_loss,
                    "local_probe_loss": local_loss,
                    "global_answer_probe_loss": global_answer_loss,
                    "local_answer_probe_loss": local_answer_loss,
                    "eta": eta,
                }
            )
            del mean_gradients, sum_gradients, mean_result
        del local_gradient, local_answer_gradient
        problem_outputs = output_rows[question_output_start:]
        g_hat = lark_g_hat(
            [float(row["student_nll_mean"]) for row in problem_outputs],
            [float(row["lark_brier_top50"]) for row in problem_outputs],
        )
        for row, value in zip(problem_outputs, g_hat):
            row["lark_g_hat"] = value
        if question_index % 8 == 0:
            gc.collect()
    output_dir.mkdir(parents=True, exist_ok=False)
    score_path = output_dir / "ctv_gradient_scores.jsonl"
    count = write_jsonl(score_path, output_rows)
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "anchor_adapter": args.anchor_adapter,
        "model_evidence": model_evidence,
        "candidate_shard_path": str(candidate_path),
        "candidate_shard_sha256": file_sha256(candidate_path),
        "calibration_report_path": str(calibration_path),
        "calibration_report_sha256": file_sha256(calibration_path),
        "calibration_evidence": calibration_evidence,
        "local_probe_evidence": local_probe_evidence,
        "local_probe_map_sha256": file_sha256(local_map_path),
        "eta": eta,
        "question_count": len(grouped),
        "score_count": count,
        "score_path": str(score_path),
        "score_sha256": file_sha256(score_path),
        "final_trainable_state_sha256": trainable_state_sha256(trainable),
        "runtime": runtime_metadata(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "teaching_utility_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/teaching_utility.py"
        ),
        "baseline_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/trace_baselines.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "score_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "SCORING_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\nscore_sha256={manifest['score_sha256']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
