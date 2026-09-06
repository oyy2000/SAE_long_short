#!/usr/bin/env python3
"""Prepare role-balanced exact-CTV inputs and all-candidate gradient-score shards."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
    operation_count,
    select_candidate_roles,
    stable_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--candidate-manifest",
        default="results/phase1_teaching_utility_v1/formal/candidate_pool/candidate_pool_manifest.json",
    )
    parser.add_argument(
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/ctv_inputs"
    )
    parser.add_argument("--score-shards", type=int, default=3)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    candidate_manifest_path = _resolve(args.candidate_manifest)
    candidate_manifest = read_json(candidate_manifest_path)
    candidate_path = Path(candidate_manifest["pool_path"])
    if file_sha256(candidate_path) != candidate_manifest["pool_sha256"]:
        raise ValueError("Candidate pool hash mismatch.")
    questions_dir = _resolve(args.questions_dir)
    pool_questions = {
        row["problem_id"]: row
        for row in read_jsonl(questions_dir / "pool_questions.jsonl")
    }
    probe_questions = list(read_jsonl(questions_dir / "global_probe_questions.jsonl"))
    splits = read_json(questions_dir / "question_splits.json")
    from transformers import AutoTokenizer

    student = config["student"]
    tokenizer = AutoTokenizer.from_pretrained(
        student["model_name"],
        revision=student["revision"],
        cache_dir=student["cache_dir"],
        local_files_only=True,
    )
    grouped = defaultdict(list)
    for source in read_jsonl(candidate_path):
        row = dict(source)
        row["solution_token_count"] = len(
            tokenizer.encode(str(row["solution"]), add_special_tokens=False)
        )
        row["student_prompt"] = pool_questions[row["problem_id"]]["student_prompt"]
        grouped[row["problem_id"]].append(row)
    role_seed = int(config["dataset"]["question_split"]["seed"])
    dev_ids = set(splits["dev"])
    exact_rows = _role_candidates(grouped, dev_ids, role_seed)
    expected_exact = int(config["exact_utility"]["question_count"]) * int(
        config["exact_utility"]["candidates_per_question"]
    )
    if len(exact_rows) != expected_exact:
        raise ValueError("Exact-CTV input cardinality mismatch.")
    calibration_count = int(config["calibration"]["question_count"])
    train_ids = set(splits["train"])
    calibration_ids = set(
        sorted(
            train_ids,
            key=lambda value: stable_hash(role_seed, "calibration", value),
        )[:calibration_count]
    )
    calibration_rows = _role_candidates(grouped, calibration_ids, role_seed)
    if calibration_ids & dev_ids:
        raise AssertionError(
            "Eta calibration questions overlap exact-CTV dev questions."
        )

    difficulty_rows = []
    for row in [*pool_questions.values(), *probe_questions]:
        difficulty_rows.append(
            {
                "problem_id": row["problem_id"],
                "source_index": row["source_index"],
                "official_solution_tokens": len(
                    tokenizer.encode(
                        row["official_completion"], add_special_tokens=False
                    )
                ),
                "operation_count": operation_count(row["official_completion"]),
                "base_student_official_rationale_nll": None,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for name, rows in (
        ("exact_ctv_candidates.jsonl", exact_rows),
        ("calibration_candidates.jsonl", calibration_rows),
        ("difficulty_features_pre_nll.jsonl", difficulty_rows),
        ("global_probes.jsonl", probe_questions),
    ):
        path = output_dir / name
        count = write_jsonl(path, rows)
        artifacts.append(_artifact(path, count))
    ordered_ids = sorted(grouped)
    for shard in range(args.score_shards):
        shard_ids = {
            problem_id
            for index, problem_id in enumerate(ordered_ids)
            if index % args.score_shards == shard
        }
        rows = [row for problem_id in sorted(shard_ids) for row in grouped[problem_id]]
        path = (
            output_dir
            / "score_shards"
            / f"shard_{shard:02d}_of_{args.score_shards:02d}.jsonl"
        )
        count = write_jsonl(path, rows)
        artifacts.append(_artifact(path, count))
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "dev_question_count": len(dev_ids),
        "exact_candidate_count": len(exact_rows),
        "calibration_question_count": len(calibration_ids),
        "calibration_candidate_count": len(calibration_rows),
        "calibration_split": "train",
        "exact_utility_split": "dev",
        "calibration_exact_question_overlap": 0,
        "global_probe_count": len(probe_questions),
        "score_shards": args.score_shards,
        "artifacts": artifacts,
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "ctv_input_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "CTV_INPUTS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )


def _artifact(path: Path, records: int) -> dict:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "record_count": records,
        "size_bytes": path.stat().st_size,
    }


def _role_candidates(grouped, problem_ids, seed):
    rows = []
    for problem_id in sorted(problem_ids):
        roles = select_candidate_roles(grouped[problem_id], seed)
        for role, row in roles.items():
            rows.append({**dict(row), "candidate_role": role})
    return rows


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
