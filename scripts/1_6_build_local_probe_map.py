#!/usr/bin/env python3
"""Compute base-student difficulty and select 32 nearest held-out probes per pool question."""

from __future__ import annotations

import argparse
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
    batched_probe_losses,
    load_lora_student,
    tokenize_prompt_completion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--input-dir", default="results/phase1_teaching_utility_v1/formal/ctv_inputs"
    )
    parser.add_argument(
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/local_probes"
    )
    parser.add_argument("--anchor-adapter", default=None)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    input_dir = _resolve(args.input_dir)
    questions_dir = _resolve(args.questions_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    model, tokenizer, _, model_evidence = load_lora_student(
        config["student"], adapter_path=args.anchor_adapter
    )
    question_rows = list(read_jsonl(questions_dir / "pool_questions.jsonl")) + list(
        read_jsonl(questions_dir / "global_probe_questions.jsonl")
    )
    pre_features = {
        row["problem_id"]: dict(row)
        for row in read_jsonl(input_dir / "difficulty_features_pre_nll.jsonl")
    }
    max_length = int(config["utility"]["max_length"])
    encoded = [
        tokenize_prompt_completion(
            tokenizer,
            record_id=row["problem_id"],
            prompt=row["student_prompt"],
            completion=row["official_completion"],
            max_length=max_length,
        )
        for row in question_rows
    ]
    import torch

    nll_by_id = {}
    with torch.inference_mode():
        for start in range(0, len(encoded), 4):
            batch = encoded[start : start + 4]
            losses = (
                batched_probe_losses(model, batch, answer_only=False)
                .detach()
                .cpu()
                .tolist()
            )
            for item, loss in zip(batch, losses):
                nll_by_id[item.record_id] = float(loss)
    features = []
    for problem_id in sorted(pre_features):
        row = pre_features[problem_id]
        row["base_student_official_rationale_nll"] = nll_by_id[problem_id]
        features.append(row)
    fields = config["utility"]["local_difficulty_features"]
    centers = {
        field: statistics.fmean(float(row[field]) for row in features)
        for field in fields
    }
    scales = {
        field: max(statistics.pstdev(float(row[field]) for row in features), 1e-12)
        for field in fields
    }
    normalized = {
        row["problem_id"]: [
            (float(row[field]) - centers[field]) / scales[field] for field in fields
        ]
        for row in features
    }
    pool_ids = {
        row["problem_id"] for row in read_jsonl(questions_dir / "pool_questions.jsonl")
    }
    probe_ids = {
        row["problem_id"]
        for row in read_jsonl(questions_dir / "global_probe_questions.jsonl")
    }
    local_count = int(config["utility"]["local_probe_questions"])
    mapping = []
    for problem_id in sorted(pool_ids):
        vector = normalized[problem_id]
        nearest = sorted(
            probe_ids,
            key=lambda probe_id: (
                sum(
                    (left - right) ** 2
                    for left, right in zip(vector, normalized[probe_id])
                ),
                probe_id,
            ),
        )[:local_count]
        mapping.append({"problem_id": problem_id, "local_probe_problem_ids": nearest})
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / "difficulty_features.jsonl"
    map_path = output_dir / "local_probe_map.jsonl"
    write_jsonl(feature_path, features)
    write_jsonl(map_path, mapping)
    report = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "anchor_adapter": args.anchor_adapter,
        "model_evidence": model_evidence,
        "feature_fields": fields,
        "feature_centers": centers,
        "feature_scales": scales,
        "pool_question_count": len(pool_ids),
        "probe_question_count": len(probe_ids),
        "local_probe_count": local_count,
        "difficulty_path": str(feature_path),
        "difficulty_sha256": file_sha256(feature_path),
        "local_probe_map_path": str(map_path),
        "local_probe_map_sha256": file_sha256(map_path),
        "runtime": runtime_metadata(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "teaching_utility_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/teaching_utility.py"
        ),
        "formal_claim_allowed": False,
    }
    report_path = output_dir / "local_probe_report.json"
    write_json_exclusive(report_path, report)
    (output_dir / "LOCAL_PROBES_COMPLETE").write_text(
        f"status=complete\nconfig_hash={report['config_hash']}\nreport_sha256={file_sha256(report_path)}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
