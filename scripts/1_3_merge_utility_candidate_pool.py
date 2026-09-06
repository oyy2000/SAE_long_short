#!/usr/bin/env python3
"""Merge the sealed 881-question parent pool with Phase-1 extension shards."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.ranked_sampling import (
    build_length_agnostic_teacher_prompt,
    normalized_completion_key,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument("--extension-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/phase1_teaching_utility_v1/formal/candidate_pool",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    questions_dir = _resolve(args.questions_dir)
    extension_root = _resolve(args.extension_root)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    question_manifest = read_json(questions_dir / "question_manifest.json")
    if question_manifest.get("status") != "complete" or question_manifest.get(
        "config_hash"
    ) != canonical_sha256(config):
        raise ValueError(
            "Question manifest is incomplete or bound to another protocol."
        )
    splits = read_json(questions_dir / "question_splits.json")
    split_by_id = {
        problem_id: split
        for split, problem_ids in splits.items()
        for problem_id in problem_ids
    }
    parent_project = Path(config["source_pool"]["parent_config_path"]).parent.parent
    parent_audit_path = Path(config["source_pool"]["parent_generation_audit_path"])
    if (
        file_sha256(parent_audit_path)
        != config["source_pool"]["parent_generation_audit_sha256"]
    ):
        raise ValueError("Parent audit hash mismatch.")
    parent_audit = read_json(parent_audit_path)
    rows = []
    input_artifacts = []
    for item in parent_audit["input_manifests"]:
        manifest_path = _parent_path(parent_project, item["path"])
        if file_sha256(manifest_path) != item["sha256"]:
            raise ValueError(f"Parent shard manifest hash mismatch: {manifest_path}")
        manifest = read_json(manifest_path)
        raw = manifest["raw"]
        raw_path = _parent_path(parent_project, raw["path"])
        if file_sha256(raw_path) != raw["sha256"]:
            raise ValueError(f"Parent raw hash mismatch: {raw_path}")
        for source in read_jsonl(raw_path):
            row = dict(source)
            row["pool_source"] = "sealed_parent_881"
            rows.append(row)
        input_artifacts.append(
            {
                "path": str(raw_path),
                "sha256": raw["sha256"],
                "record_count": raw["record_count"],
            }
        )
    shard_count = int(config["teacher"]["generation_shards"])
    for shard in range(shard_count):
        shard_dir = extension_root / f"shard_{shard:02d}_of_{shard_count:02d}"
        manifest_path = shard_dir / "generation_manifest.json"
        marker_path = shard_dir / "GENERATION_COMPLETE"
        if not marker_path.is_file():
            raise FileNotFoundError(marker_path)
        manifest = read_json(manifest_path)
        raw_path = Path(manifest["raw_path"])
        if file_sha256(raw_path) != manifest["raw_sha256"]:
            raise ValueError(f"Extension raw hash mismatch: {raw_path}")
        for source in read_jsonl(raw_path):
            row = dict(source)
            row["pool_source"] = "phase1_extension_1119"
            rows.append(row)
        input_artifacts.append(
            {
                "path": str(raw_path),
                "sha256": manifest["raw_sha256"],
                "record_count": manifest["candidate_count"],
            }
        )
    expected_raw = int(config["source_pool"]["target_problem_count"]) * int(
        config["teacher"]["num_rollouts"]
    )
    if (
        len(rows) != expected_raw
        or len({str(row["trace_id"]) for row in rows}) != expected_raw
    ):
        raise ValueError("Merged raw pool cardinality or trace identity mismatch.")
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        expected_prompt = build_length_agnostic_teacher_prompt(str(row["question"]))
        if str(row.get("prompt", "")) != expected_prompt:
            raise ValueError(
                f"Teacher-prompt drift in merged pool: {row.get('trace_id')}"
            )
        predicted = extract_final_answer(str(row["solution"]))
        row["predicted_answer"] = predicted
        row["is_correct"] = verify_answer(predicted, str(row["answer"]))
        row["question_split"] = split_by_id[str(row["problem_id"])]
        grouped[str(row["problem_id"])].append(row)
    eligible = []
    counts = {}
    for problem_id in sorted(grouped):
        seen = set()
        correct = []
        for row in sorted(
            grouped[problem_id],
            key=lambda value: (int(value["candidate_index"]), str(value["trace_id"])),
        ):
            key = normalized_completion_key(str(row["solution"]))
            if bool(row["is_correct"]) and key not in seen:
                seen.add(key)
                correct.append(row)
        counts[problem_id] = len(correct)
        if len(correct) < 4:
            raise ValueError(
                f"Problem {problem_id} has only {len(correct)} unique correct traces."
            )
        eligible.extend(correct)
    output_dir.mkdir(parents=True, exist_ok=False)
    pool_path = output_dir / "unique_correct_candidates.jsonl"
    count = write_jsonl(pool_path, eligible)
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "question_manifest_sha256": file_sha256(
            questions_dir / "question_manifest.json"
        ),
        "raw_record_count": len(rows),
        "problem_count": len(grouped),
        "unique_correct_record_count": count,
        "minimum_unique_correct": min(counts.values()),
        "maximum_unique_correct": max(counts.values()),
        "split_problem_counts": {name: len(values) for name, values in splits.items()},
        "teacher_prompt_version": "sealed_parent_length_agnostic_v1",
        "teacher_prompt_drift_count": 0,
        "pool_path": str(pool_path),
        "pool_sha256": file_sha256(pool_path),
        "input_artifacts": input_artifacts,
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "prompt_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/ranked_sampling.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "candidate_pool_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "CANDIDATE_POOL_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\npool_sha256={manifest['pool_sha256']}\n",
        encoding="utf-8",
    )


def _parent_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
