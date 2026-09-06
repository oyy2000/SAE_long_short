#!/usr/bin/env python3
"""Measure exact local/global CTV for one question shard of the 300-by-4 dev design."""

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
    reversible_micro_update_multi,
    tokenize_prompt_completion,
    trainable_state_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--exact-candidates",
        default="results/phase1_teaching_utility_v1/formal/ctv_inputs/exact_ctv_candidates.jsonl",
    )
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
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-adapter", default=None)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard topology.")
    config_path = _resolve(args.config)
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
    exact_path = _resolve(args.exact_candidates)
    all_grouped = defaultdict(list)
    for row in read_jsonl(exact_path):
        all_grouped[row["problem_id"]].append(dict(row))
    selected_ids = [
        problem_id
        for index, problem_id in enumerate(sorted(all_grouped))
        if index % args.shard_count == args.shard_index
    ]
    output_rows = []
    for question_index, problem_id in enumerate(selected_ids):
        local_probe = [encoded_probes[value] for value in local_map[problem_id]]
        local_gradient, local_loss = mean_probe_gradient(model, trainable, local_probe)
        local_answer_gradient, local_answer_loss = mean_probe_gradient(
            model, trainable, local_probe, answer_only=True
        )
        for row in all_grouped[problem_id]:
            encoded = tokenize_prompt_completion(
                tokenizer,
                record_id=row["trace_id"],
                prompt=row["student_prompt"],
                completion=row["solution"],
                max_length=max_length,
            )
            result = gradient_for_completion(
                model, trainable, encoded, reduction="token_mean"
            )
            mean_gradients = result.pop("gradients")
            gradient_target_token_count = int(result["token_count"])
            raw_solution_token_count = int(row["solution_token_count"])
            sum_gradients = [
                gradient * gradient_target_token_count for gradient in mean_gradients
            ]
            for reduction, gradients in (
                ("token_mean", mean_gradients),
                ("token_sum", sum_gradients),
            ):
                exact = reversible_micro_update_multi(
                    model,
                    trainable,
                    gradients,
                    eta=eta,
                    probes={
                        "global": global_probe,
                        "local": local_probe,
                        "global_answer": global_probe,
                        "local_answer": local_probe,
                    },
                    base_probe_losses={
                        "global": global_loss,
                        "local": local_loss,
                        "global_answer": global_answer_loss,
                        "local_answer": local_answer_loss,
                    },
                    answer_only_by_probe={
                        "global_answer": True,
                        "local_answer": True,
                    },
                )
                output_rows.append(
                    {
                        "trace_id": row["trace_id"],
                        "problem_id": problem_id,
                        "candidate_role": row["candidate_role"],
                        "loss_reduction": reduction,
                        "solution_token_count": raw_solution_token_count,
                        "gradient_target_token_count": gradient_target_token_count,
                        "log_solution_token_count": math.log(raw_solution_token_count),
                        "student_nll_mean": result["loss_mean"],
                        "student_nll_sum": result["loss_sum"],
                        "gradient_norm": result["gradient_norm"]
                        * (
                            gradient_target_token_count
                            if reduction == "token_sum"
                            else 1.0
                        ),
                        "ctv_gradient_global": first_order_ctv(
                            global_gradient, gradients, eta
                        ),
                        "ctv_gradient_local": first_order_ctv(
                            local_gradient, gradients, eta
                        ),
                        "ctv_gradient_global_answer": first_order_ctv(
                            global_answer_gradient, gradients, eta
                        ),
                        "ctv_gradient_local_answer": first_order_ctv(
                            local_answer_gradient, gradients, eta
                        ),
                        "ctv_exact_global": exact["ctv_exact"]["global"],
                        "ctv_exact_local": exact["ctv_exact"]["local"],
                        "ctv_exact_global_answer": exact["ctv_exact"]["global_answer"],
                        "ctv_exact_local_answer": exact["ctv_exact"]["local_answer"],
                        "eta": eta,
                        "parameters_restored_exactly": exact[
                            "parameters_restored_exactly"
                        ],
                    }
                )
            del mean_gradients, sum_gradients, result
        del local_gradient, local_answer_gradient
        if question_index % 4 == 0:
            gc.collect()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "exact_ctv.jsonl"
    count = write_jsonl(rows_path, output_rows)
    expected = (
        len(selected_ids) * int(config["exact_utility"]["candidates_per_question"]) * 2
    )
    if count != expected:
        raise ValueError(f"Exact CTV row count mismatch: {count} != {expected}")
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "anchor_adapter": args.anchor_adapter,
        "model_evidence": model_evidence,
        "exact_candidates_path": str(exact_path),
        "exact_candidates_sha256": file_sha256(exact_path),
        "calibration_report_sha256": file_sha256(calibration_path),
        "calibration_evidence": calibration_evidence,
        "local_probe_evidence": local_probe_evidence,
        "local_probe_map_sha256": file_sha256(local_map_path),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "question_count": len(selected_ids),
        "micro_update_count": count,
        "rows_path": str(rows_path),
        "rows_sha256": file_sha256(rows_path),
        "final_trainable_state_sha256": trainable_state_sha256(trainable),
        "runtime": runtime_metadata(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "teaching_utility_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/teaching_utility.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "exact_ctv_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "EXACT_CTV_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\nrows_sha256={manifest['rows_sha256']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
