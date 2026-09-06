#!/usr/bin/env python3
"""Build the registered fixed-long-context supervision-allocation conditions."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.credit_allocation import (
    ReasoningStep,
    deterministic_random_step_scores,
    equal_step_total_weights,
    relative_budget_gap,
    segment_reasoning_steps,
    select_step_knapsack,
    supervision_weights,
)
from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.teaching_utility import select_candidate_roles


CONDITIONS = (
    "long_full",
    "long_step_balanced",
    "long_random_mask",
    "long_high_nll_mask",
    "long_ctv_mask",
    "long_answer_only",
    "same_source_pruned_ctv",
    "short_band_full",
    "answer_only_pruned",
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
    parser.add_argument("--step-score-root", required=True)
    parser.add_argument(
        "--output-dir", default="results/phase1_5_credit_allocation_v1/formal/data"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    utility_config_path = _resolve(args.utility_config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    utility_config = read_json(utility_config_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    from transformers import AutoTokenizer

    student = utility_config["student"]
    tokenizer = AutoTokenizer.from_pretrained(
        student["model_name"],
        revision=student["revision"],
        cache_dir=student["cache_dir"],
        local_files_only=True,
    )
    questions_dir = _resolve(args.questions_dir)
    train_ids = set(read_json(questions_dir / "question_splits.json")["train"])
    prompts = {
        row["problem_id"]: row["student_prompt"]
        for row in read_jsonl(questions_dir / "pool_questions.jsonl")
    }
    candidate_manifest_path = _resolve(args.candidate_manifest)
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_path = Path(candidate_manifest["pool_path"])
    if file_sha256(candidate_path) != candidate_manifest["pool_sha256"]:
        raise ValueError("Candidate pool hash mismatch.")
    candidates = defaultdict(list)
    trace_by_id = {}
    for source in read_jsonl(candidate_path):
        if source["problem_id"] not in train_ids:
            continue
        row = dict(source)
        row["solution_token_count"] = len(
            tokenizer.encode(row["solution"], add_special_tokens=False)
        )
        candidates[row["problem_id"]].append(row)
        trace_by_id[row["trace_id"]] = row
    step_rows, step_manifests = _load_step_scores(_resolve(args.step_score_root))
    steps_by_problem = defaultdict(list)
    for row in step_rows:
        steps_by_problem[row["problem_id"]].append(row)
    if set(steps_by_problem) != train_ids:
        raise ValueError("Step-score shards do not cover the full training split.")
    role_seed = int(utility_config["dataset"]["question_split"]["seed"])
    maximum_gap = float(config["mask_budget"]["maximum_relative_gap"])
    random_seed = int(config["pilot"]["seed"])
    selected_by_condition = {condition: [] for condition in CONDITIONS}
    audit_rows = []
    dropped = []
    for problem_id in sorted(train_ids):
        roles = select_candidate_roles(candidates[problem_id], role_seed)
        short_trace = roles["quantile_short"]
        long_trace_id = str(steps_by_problem[problem_id][0]["trace_id"])
        if any(
            str(row["trace_id"]) != long_trace_id
            for row in steps_by_problem[problem_id]
        ):
            raise ValueError(f"Multiple long traces in step scores for {problem_id}")
        if long_trace_id != str(roles["quantile_long"]["trace_id"]):
            raise ValueError(f"Long-trace selection drift for {problem_id}")
        long_trace = trace_by_id[long_trace_id]
        try:
            long_steps = _restore_steps(steps_by_problem[problem_id])
            short_steps = segment_reasoning_steps(
                str(short_trace["solution"]),
                tokenizer,
                minimum_step_tokens=int(config["segmentation"]["minimum_step_tokens"]),
            )
            target = sum(step.token_count for step in short_steps if not step.is_answer)
            if target <= 0:
                raise ValueError("paired short trace has no reasoning-token budget")
            score_vectors = {
                "long_random_mask": deterministic_random_step_scores(
                    problem_id, long_steps, random_seed
                ),
                "long_high_nll_mask": [
                    float(row["step_nll_sum"])
                    for row in sorted(
                        steps_by_problem[problem_id],
                        key=lambda value: int(value["step_index"]),
                    )
                ],
                "long_ctv_mask": [
                    float(row["step_ctv_gradient_global_total"])
                    for row in sorted(
                        steps_by_problem[problem_id],
                        key=lambda value: int(value["step_index"]),
                    )
                ],
            }
            selections = {
                condition: select_step_knapsack(
                    long_steps,
                    scores,
                    target_reasoning_tokens=target,
                    maximum_relative_gap=maximum_gap,
                )
                for condition, scores in score_vectors.items()
            }
        except ValueError as exc:
            dropped.append({"problem_id": problem_id, "reason": str(exc)})
            continue
        long_completion = str(long_trace["solution"])
        short_completion = str(short_trace["solution"])
        long_token_count = len(
            tokenizer.encode(long_completion, add_special_tokens=False)
        )
        short_token_count = len(
            tokenizer.encode(short_completion, add_special_tokens=False)
        )
        all_long = [1.0] * long_token_count
        answer_only = supervision_weights(long_token_count, long_steps, [])
        balanced = equal_step_total_weights(long_steps, float(target))
        balanced = _pad_weights(balanced, long_token_count)
        condition_specs = {
            "long_full": (long_completion, all_long, long_trace_id),
            "long_step_balanced": (long_completion, balanced, long_trace_id),
            "long_random_mask": (
                long_completion,
                supervision_weights(
                    long_token_count, long_steps, selections["long_random_mask"]
                ),
                long_trace_id,
            ),
            "long_high_nll_mask": (
                long_completion,
                supervision_weights(
                    long_token_count, long_steps, selections["long_high_nll_mask"]
                ),
                long_trace_id,
            ),
            "long_ctv_mask": (
                long_completion,
                supervision_weights(
                    long_token_count, long_steps, selections["long_ctv_mask"]
                ),
                long_trace_id,
            ),
            "long_answer_only": (long_completion, answer_only, long_trace_id),
            "same_source_pruned_ctv": _pruned_spec(
                long_steps, selections["long_ctv_mask"], tokenizer, long_trace_id
            ),
            "short_band_full": (
                short_completion,
                [1.0] * short_token_count,
                str(short_trace["trace_id"]),
            ),
            "answer_only_pruned": _pruned_spec(
                long_steps, [], tokenizer, long_trace_id
            ),
        }
        for condition, (
            completion,
            weights,
            source_trace_id,
        ) in condition_specs.items():
            selected_reasoning_tokens = _reasoning_weight_count(
                weights,
                completion,
                tokenizer,
                int(config["segmentation"]["minimum_step_tokens"]),
            )
            selected_by_condition[condition].append(
                {
                    "problem_id": problem_id,
                    "trace_id": f"{source_trace_id}:{condition}",
                    "source_trace_id": source_trace_id,
                    "long_context_trace_id": long_trace_id,
                    "prompt": prompts[problem_id],
                    "completion": completion,
                    "completion_loss_weights": weights,
                    "condition": condition,
                    "completion_token_count": len(weights),
                    "paired_short_reasoning_token_target": target,
                    "supervised_reasoning_weight": selected_reasoning_tokens,
                    "uses_identical_long_context": condition.startswith("long_"),
                }
            )
        audit_rows.append(
            {
                "problem_id": problem_id,
                "long_trace_id": long_trace_id,
                "short_trace_id": short_trace["trace_id"],
                "target_reasoning_tokens": target,
                "steps": [
                    dict(row)
                    for row in sorted(
                        steps_by_problem[problem_id],
                        key=lambda value: int(value["step_index"]),
                    )
                ],
                "selected_step_indices": selections,
                "budget_gaps": {
                    condition: relative_budget_gap(
                        sum(
                            step.token_count
                            for step in long_steps
                            if step.index in indices
                        ),
                        target,
                    )
                    for condition, indices in selections.items()
                },
            }
        )
    retained = len(audit_rows)
    if retained == 0 or any(
        len(rows) != retained for rows in selected_by_condition.values()
    ):
        raise ValueError("Credit-allocation conditions lack common non-empty support.")
    support = {row["problem_id"] for row in selected_by_condition[CONDITIONS[0]]}
    if any(
        {row["problem_id"] for row in rows} != support
        for rows in selected_by_condition.values()
    ):
        raise ValueError(
            "Credit-allocation conditions have different question support."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for condition in CONDITIONS:
        path = output_dir / f"{condition}.jsonl"
        count = write_jsonl(path, selected_by_condition[condition])
        artifacts.append(
            {
                "condition": condition,
                "path": str(path),
                "sha256": file_sha256(path),
                "record_count": count,
            }
        )
    audit_count = int(config["segmentation"]["manual_audit_questions"])
    audit_sample = sorted(
        audit_rows,
        key=lambda row: hashlib.sha256(
            f"audit|{row['problem_id']}".encode()
        ).hexdigest(),
    )[:audit_count]
    audit_path = output_dir / "step_audit_sample.jsonl"
    write_jsonl(audit_path, audit_sample)
    dropped_path = output_dir / "dropped_questions.jsonl"
    write_jsonl(dropped_path, dropped)
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "utility_config_hash": canonical_sha256(utility_config),
        "utility_config_sha256": file_sha256(utility_config_path),
        "gate1_evidence": gate1,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "step_score_manifests": step_manifests,
        "input_question_count": len(train_ids),
        "retained_common_support_count": retained,
        "dropped_question_count": len(dropped),
        "drop_policy": config["mask_budget"]["gap_policy"],
        "maximum_relative_budget_gap": maximum_gap,
        "identical_long_context_conditions": [
            condition for condition in CONDITIONS if condition.startswith("long_")
        ],
        "artifacts": artifacts,
        "manual_audit_path": str(audit_path),
        "manual_audit_sha256": file_sha256(audit_path),
        "dropped_path": str(dropped_path),
        "dropped_sha256": file_sha256(dropped_path),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "credit_allocation_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/credit_allocation.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "credit_allocation_data_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "DATA_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        f"retained_common_support_count={retained}\n",
        encoding="utf-8",
    )


def _load_step_scores(root: Path):
    rows, manifests = [], []
    for path in sorted(root.glob("shard_*/step_score_manifest.json")):
        manifest = read_json(path)
        rows_path = Path(manifest["rows_path"])
        if (
            manifest.get("status") != "complete"
            or file_sha256(rows_path) != manifest["rows_sha256"]
        ):
            raise ValueError(f"Invalid step-score shard: {path}")
        rows.extend(dict(row) for row in read_jsonl(rows_path))
        manifests.append({"path": str(path), "sha256": file_sha256(path)})
    if not manifests:
        raise FileNotFoundError(f"No step-score shards under {root}")
    return rows, manifests


def _restore_steps(rows):
    return [
        ReasoningStep(
            index=int(row["step_index"]),
            text=str(row["step_text"]),
            char_start=int(row["char_start"]),
            char_end=int(row["char_end"]),
            token_start=int(row["token_start"]),
            token_end=int(row["token_end"]),
            is_answer=bool(row["is_answer"]),
        )
        for row in sorted(rows, key=lambda value: int(value["step_index"]))
    ]


def _pruned_spec(steps, selected, tokenizer, source_trace_id):
    selected_set = set(selected)
    text = "\n".join(
        step.text.strip()
        for step in steps
        if step.is_answer or step.index in selected_set
    )
    weights = [1.0] * len(tokenizer.encode(text, add_special_tokens=False))
    return text, weights, source_trace_id


def _pad_weights(weights, length):
    if len(weights) > length:
        raise ValueError("Loss-weight vector exceeds completion length.")
    return list(weights) + [0.0] * (length - len(weights))


def _reasoning_weight_count(weights, completion, tokenizer, minimum_step_tokens):
    steps = segment_reasoning_steps(
        completion, tokenizer, minimum_step_tokens=minimum_step_tokens
    )
    answer_positions = {
        position
        for step in steps
        if step.is_answer
        for position in range(step.token_start, step.token_end)
    }
    return float(
        sum(
            weight
            for index, weight in enumerate(weights)
            if index not in answer_positions
        )
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
