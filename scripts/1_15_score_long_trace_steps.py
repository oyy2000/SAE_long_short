#!/usr/bin/env python3
"""Score reasoning steps inside full long-trace contexts by NLL and first-order CTV."""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.credit_allocation import segment_reasoning_steps
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
    gradient_for_completion_positions,
    load_lora_student,
    mean_probe_gradient,
    select_candidate_roles,
    tokenize_prompt_completion,
    trainable_state_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase1_5_credit_allocation_v1.json"
    )
    parser.add_argument(
        "--utility-config", default="configs/phase1_teaching_utility_v1.json"
    )
    parser.add_argument(
        "--candidate-manifest",
        default="results/phase1_teaching_utility_v1/formal/candidate_pool/candidate_pool_manifest.json",
    )
    parser.add_argument(
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument(
        "--calibration-report",
        default="results/phase1_teaching_utility_v1/formal/calibration/calibration_report.json",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard topology.")
    config_path = _resolve(args.config)
    utility_config_path = _resolve(args.utility_config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    utility_config = read_json(utility_config_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    calibration_path = _resolve(args.calibration_report)
    calibration = read_json(calibration_path)
    calibration_evidence = validated_artifact_marker(
        calibration_path.parent / "CALIBRATION_COMPLETE",
        expected_status="passed",
        hash_bindings={"report_sha256": calibration_path},
    )
    if calibration.get("status") != "passed":
        raise ValueError("CTV calibration is incomplete.")
    eta = float(calibration["selected_eta"])
    model, tokenizer, trainable, model_evidence = load_lora_student(
        utility_config["student"]
    )
    questions_dir = _resolve(args.questions_dir)
    split_path = questions_dir / "question_splits.json"
    train_ids = set(read_json(split_path)["train"])
    prompts = {
        row["problem_id"]: row["student_prompt"]
        for row in read_jsonl(questions_dir / "pool_questions.jsonl")
    }
    candidate_manifest_path = _resolve(args.candidate_manifest)
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_path = Path(candidate_manifest["pool_path"])
    if file_sha256(candidate_path) != candidate_manifest["pool_sha256"]:
        raise ValueError("Candidate pool hash mismatch.")
    grouped = defaultdict(list)
    for source in read_jsonl(candidate_path):
        if source["problem_id"] in train_ids:
            row = dict(source)
            row["solution_token_count"] = len(
                tokenizer.encode(row["solution"], add_special_tokens=False)
            )
            grouped[row["problem_id"]].append(row)
    selected = {
        problem_id: dict(
            select_candidate_roles(
                rows, int(utility_config["dataset"]["question_split"]["seed"])
            )["quantile_long"]
        )
        for problem_id, rows in grouped.items()
    }
    shard_ids = [
        problem_id
        for index, problem_id in enumerate(sorted(selected))
        if index % args.shard_count == args.shard_index
    ]
    probe_rows = list(read_jsonl(questions_dir / "global_probe_questions.jsonl"))
    max_length = int(utility_config["utility"]["max_length"])
    probe = [
        tokenize_prompt_completion(
            tokenizer,
            record_id=row["problem_id"],
            prompt=row["student_prompt"],
            completion=row["official_completion"],
            max_length=max_length,
        )
        for row in probe_rows
    ]
    probe_gradient, probe_loss = mean_probe_gradient(model, trainable, probe)
    output_rows = []
    minimum_step_tokens = int(config["segmentation"]["minimum_step_tokens"])
    for question_index, problem_id in enumerate(shard_ids):
        trace = selected[problem_id]
        completion = str(trace["solution"])
        steps = segment_reasoning_steps(
            completion, tokenizer, minimum_step_tokens=minimum_step_tokens
        )
        encoded = tokenize_prompt_completion(
            tokenizer,
            record_id=trace["trace_id"],
            prompt=prompts[problem_id],
            completion=completion,
            max_length=max_length,
        )
        for step in steps:
            positions = list(range(step.token_start, step.token_end))
            result = gradient_for_completion_positions(
                model, trainable, encoded, target_positions=positions
            )
            gradients = result.pop("gradients")
            step_ctv_mean = first_order_ctv(probe_gradient, gradients, eta)
            output_rows.append(
                {
                    "problem_id": problem_id,
                    "trace_id": trace["trace_id"],
                    "step_index": step.index,
                    "step_text": step.text,
                    "char_start": step.char_start,
                    "char_end": step.char_end,
                    "token_start": step.token_start,
                    "token_end": step.token_end,
                    "token_count": step.token_count,
                    "is_answer": step.is_answer,
                    "step_nll_mean": result["loss_mean"],
                    "step_nll_sum": result["loss_mean"] * step.token_count,
                    "step_gradient_norm": result["gradient_norm"],
                    "step_ctv_gradient_global_mean": step_ctv_mean,
                    "step_ctv_gradient_global_total": step_ctv_mean * step.token_count,
                    "eta": eta,
                }
            )
            del gradients, result
        if question_index % 8 == 0:
            gc.collect()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "long_trace_step_scores.jsonl"
    count = write_jsonl(rows_path, output_rows)
    if {row["problem_id"] for row in output_rows} != set(shard_ids):
        raise ValueError("Step scoring lost one or more questions.")
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "utility_config_hash": canonical_sha256(utility_config),
        "utility_config_sha256": file_sha256(utility_config_path),
        "gate1_evidence": gate1,
        "model_evidence": model_evidence,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "question_splits_sha256": file_sha256(split_path),
        "calibration_report_sha256": file_sha256(calibration_path),
        "calibration_evidence": calibration_evidence,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "question_count": len(shard_ids),
        "step_count": count,
        "global_probe_loss": probe_loss,
        "rows_path": str(rows_path),
        "rows_sha256": file_sha256(rows_path),
        "final_trainable_state_sha256": trainable_state_sha256(trainable),
        "runtime": runtime_metadata(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "teaching_utility_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/teaching_utility.py"
        ),
        "credit_allocation_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/credit_allocation.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "step_score_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "STEP_SCORING_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        f"rows_sha256={manifest['rows_sha256']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
