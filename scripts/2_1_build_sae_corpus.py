#!/usr/bin/env python3
"""Build the question-split, label-mixed SAE trajectory corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.sae_data import (
    build_mixed_trace_corpus,
    stable_question_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument(
        "--output-dir", default="results/phase2_sae_pilot_v1/formal/corpus"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    source = config["source_pool"]
    audit_path = Path(source["generation_audit_path"])
    if file_sha256(audit_path) != source["generation_audit_sha256"]:
        raise ValueError("Generation audit does not match the frozen protocol.")
    audit = read_json(audit_path)
    if audit.get("status") != "passed":
        raise ValueError("Parent trajectory generation audit did not pass.")
    rows = []
    raw_evidence = []
    for shard in source["raw_shards"]:
        path = Path(shard["path"])
        observed_hash = file_sha256(path)
        shard_rows = list(read_jsonl(path))
        if observed_hash != shard["sha256"] or len(shard_rows) != int(shard["records"]):
            raise ValueError(f"Raw-shard evidence mismatch: {path}")
        rows.extend(shard_rows)
        raw_evidence.append(
            {"path": str(path), "sha256": observed_hash, "records": len(shard_rows)}
        )
    if len(rows) != int(source["trajectory_count"]):
        raise ValueError("Unexpected total raw trajectory count.")
    per_question = Counter(str(row["problem_id"]) for row in rows)
    if len(per_question) != int(source["question_count"]) or set(
        per_question.values()
    ) != {int(source["rollouts_per_question"])}:
        raise ValueError(
            "Parent pool is not the registered rectangular question matrix."
        )
    split_config = config["question_split"]
    assignments = stable_question_split(
        per_question,
        train_count=int(split_config["train"]),
        dev_count=int(split_config["dev"]),
        test_count=int(split_config["test"]),
        seed=int(split_config["seed"]),
    )
    corpus, summary = build_mixed_trace_corpus(
        rows,
        question_splits=assignments,
        minimum_correct=int(source["minimum_correct_per_question"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    corpus_path = output_dir / "mixed_trajectories.jsonl"
    write_jsonl(corpus_path, corpus)
    split_path = output_dir / "question_split.json"
    write_json_exclusive(
        split_path,
        {
            "status": "complete",
            "method": split_config["method"],
            "seed": int(split_config["seed"]),
            "assignments": assignments,
        },
    )
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "generation_audit_path": str(audit_path),
        "generation_audit_sha256": file_sha256(audit_path),
        "raw_shards": raw_evidence,
        "corpus_path": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
        "question_split_path": str(split_path),
        "question_split_sha256": file_sha256(split_path),
        **summary,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "corpus_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "CORPUS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\n"
        f"corpus_sha256={manifest['corpus_sha256']}\n"
        f"question_split_sha256={manifest['question_split_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
