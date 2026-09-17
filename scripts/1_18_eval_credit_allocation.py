#!/usr/bin/env python3
"""Evaluate one credit-allocation adapter on an explicit Phase-1 split."""

from __future__ import annotations

import argparse
import hashlib
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
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.student_evaluation import (
    evaluate_explicit_questions,
    load_student_for_evaluation,
    summarize_predictions,
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
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path, utility_path = _resolve(args.config), _resolve(args.utility_config)
    output_dir, adapter_path = _resolve(args.output_dir), _resolve(args.adapter_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config, utility = read_json(config_path), read_json(utility_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    questions_dir = _resolve(args.questions_dir)
    split_path = questions_dir / "question_splits.json"
    split_ids = set(read_json(split_path)[args.split])
    questions_path = questions_dir / "pool_questions.jsonl"
    questions = sorted(
        (
            dict(row)
            for row in read_jsonl(questions_path)
            if row["problem_id"] in split_ids
        ),
        key=lambda row: row["problem_id"],
    )
    if {row["problem_id"] for row in questions} != split_ids:
        raise ValueError("Credit-allocation evaluation support is incomplete.")
    marker = adapter_path / "TRAIN_COMPLETE"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    training_evidence = validated_training_artifacts(adapter_path)
    bundle = load_student_for_evaluation(
        utility["student"], adapter_path=str(adapter_path)
    )
    evaluation = config["evaluation"]
    rows = evaluate_explicit_questions(
        bundle,
        questions,
        batch_size=int(evaluation["batch_size"]),
        max_new_tokens=int(evaluation["max_new_tokens"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    support_hash = hashlib.sha256(
        ("\n".join(sorted(split_ids)) + "\n").encode()
    ).hexdigest()
    summary = {
        **summarize_predictions(rows),
        "status": "complete",
        "condition": args.condition,
        "seed": args.seed,
        "split": args.split,
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "utility_config_hash": canonical_sha256(utility),
        "utility_config_sha256": file_sha256(utility_path),
        "gate1_evidence": gate1,
        "adapter_path": str(adapter_path),
        "adapter_marker_sha256": file_sha256(marker),
        "adapter_config_sha256": file_sha256(adapter_path / "adapter_config.json"),
        "adapter_model_sha256": file_sha256(adapter_path / "adapter_model.safetensors"),
        "training_metrics_sha256": training_evidence["hashes"]["training_metrics"],
        "questions_sha256": file_sha256(questions_path),
        "question_splits_sha256": file_sha256(split_path),
        "problem_ids_sha256": support_hash,
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "runtime": runtime_metadata(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "evaluation_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/student_evaluation.py"
        ),
        "formal_claim_allowed": False,
    }
    summary_path = output_dir / "evaluation_summary.json"
    write_json_exclusive(summary_path, summary)
    (output_dir / "EVALUATION_COMPLETE").write_text(
        f"status=complete\nconfig_hash={summary['config_hash']}\nsummary_sha256={file_sha256(summary_path)}\n"
        f"predictions_sha256={summary['predictions_sha256']}\nproblem_ids_sha256={support_hash}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
