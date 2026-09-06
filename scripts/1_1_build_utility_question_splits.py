#!/usr/bin/env python3
"""Build disjoint Phase-1 pool, extension shards, and held-out probe questions."""

from __future__ import annotations

import argparse
import re
import sys
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
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.student_prompts import build_student_math_prompt
from length_budget_distill.teaching_utility import deterministic_question_split
from length_budget_distill.verifiers import extract_final_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    dataset = dict(config["dataset"])
    from datasets import load_dataset

    loaded = load_dataset(
        dataset["dataset_name"], dataset.get("dataset_config"), split=dataset["split"]
    )
    probe_stop = int(dataset["probe_index_stop"])
    if len(loaded) < probe_stop:
        raise ValueError(
            "GSM8K train split is smaller than the registered pool and probe range."
        )
    pool = [
        _question_row(loaded[index], index)
        for index in range(
            int(dataset["pool_index_start"]), int(dataset["pool_index_stop"])
        )
    ]
    probes = [
        _question_row(loaded[index], index)
        for index in range(int(dataset["probe_index_start"]), probe_stop)
    ]
    if {row["problem_id"] for row in pool} & {row["problem_id"] for row in probes}:
        raise AssertionError("Pool and probe questions overlap.")

    phase0_selected = (
        PROJECT_ROOT
        / "results/trace_length_observation_gate0_v1/formal/data/selected_traces.jsonl"
    )
    existing_rows = list(read_jsonl(phase0_selected))
    existing_ids = {str(row["problem_id"]) for row in existing_rows}
    expected_existing = int(config["source_pool"]["existing_problem_count"])
    if len(existing_ids) != expected_existing:
        raise ValueError(
            f"Expected {expected_existing} Phase-0 problems, observed {len(existing_ids)}."
        )
    pool_ids = {row["problem_id"] for row in pool}
    if not existing_ids <= pool_ids:
        raise ValueError(
            "The existing 881-question pool is not a subset of GSM8K train[:2000]."
        )
    extension = [row for row in pool if row["problem_id"] not in existing_ids]
    if len(extension) != int(config["source_pool"]["extension_problem_count"]):
        raise ValueError("Extension cardinality is not the registered 1119 questions.")

    split_config = dict(dataset["question_split"])
    splits = deterministic_question_split(
        sorted(pool_ids),
        train_count=int(split_config["train"]),
        dev_count=int(split_config["dev"]),
        test_count=int(split_config["test"]),
        seed=int(split_config["seed"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for name, rows in (
        ("pool_questions.jsonl", pool),
        ("global_probe_questions.jsonl", probes),
    ):
        path = output_dir / name
        count = write_jsonl(path, rows)
        artifacts.append(_artifact(path, count))
    extension_shards = int(config["teacher"]["generation_shards"])
    for shard in range(extension_shards):
        path = (
            output_dir
            / "extension_shards"
            / f"shard_{shard:02d}_of_{extension_shards:02d}.jsonl"
        )
        rows = [
            row
            for index, row in enumerate(extension)
            if index % extension_shards == shard
        ]
        count = write_jsonl(path, rows)
        artifacts.append(_artifact(path, count))
    split_path = output_dir / "question_splits.json"
    write_json_exclusive(split_path, splits)
    artifacts.append(_artifact(split_path, None))
    manifest = {
        "status": "complete",
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config_hash": canonical_sha256(config),
        "gate_evidence": gate,
        "phase0_selected_path": str(phase0_selected),
        "phase0_selected_sha256": file_sha256(phase0_selected),
        "pool_problem_count": len(pool),
        "existing_problem_count": len(existing_ids),
        "extension_problem_count": len(extension),
        "probe_problem_count": len(probes),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "pool_probe_overlap": 0,
        "split_overlap": 0,
        "artifacts": artifacts,
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "question_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "QUESTIONS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )


def _question_row(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    raw_answer = str(item["answer"])
    gold = extract_final_answer(raw_answer)
    if gold is None:
        raise ValueError(f"Could not extract GSM8K answer at source index {index}.")
    rationale = re.sub(r"<<[^<>]*>>", "", raw_answer)
    rationale = re.sub(r"(?m)^####\s*[^\n]+\s*$", f"Answer: {gold}", rationale).strip()
    return {
        "problem_id": f"hf-{index:06d}",
        "source_index": index,
        "question": str(item["question"]),
        "answer": gold,
        "official_completion": rationale,
        "student_prompt": build_student_math_prompt(str(item["question"])),
    }


def _artifact(path: Path, records: int | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if records is not None:
        result["record_count"] = records
    return result


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
