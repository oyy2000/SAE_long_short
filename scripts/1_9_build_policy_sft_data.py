#!/usr/bin/env python3
"""Select one correct training trace per question for each Phase-1 screen policy."""

from __future__ import annotations

import argparse
import hashlib
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
from length_budget_distill.teaching_utility import select_candidate_roles
from length_budget_distill.trace_baselines import select_scas_lowest_group


POLICIES = (
    "ctv",
    "rsr",
    "scas",
    "lark_style",
    "low_nll",
    "absolute_shortest",
    "quantile_short",
    "random",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--candidate-manifest",
        default="results/phase1_teaching_utility_v1/formal/candidate_pool/candidate_pool_manifest.json",
    )
    parser.add_argument("--score-root", required=True)
    parser.add_argument(
        "--questions-dir", default="results/phase1_teaching_utility_v1/formal/questions"
    )
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/policy_data"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    candidate_manifest = read_json(_resolve(args.candidate_manifest))
    candidate_path = Path(candidate_manifest["pool_path"])
    trace_by_id = {row["trace_id"]: dict(row) for row in read_jsonl(candidate_path)}
    question_by_id = {
        row["problem_id"]: dict(row)
        for row in read_jsonl(_resolve(args.questions_dir) / "pool_questions.jsonl")
    }
    scores = []
    score_manifests = []
    for manifest_path in sorted(
        _resolve(args.score_root).glob("shard_*/score_manifest.json")
    ):
        manifest = read_json(manifest_path)
        score_path = Path(manifest["score_path"])
        if file_sha256(score_path) != manifest["score_sha256"]:
            raise ValueError(f"Score shard hash mismatch: {score_path}")
        scores.extend(dict(row) for row in read_jsonl(score_path))
        score_manifests.append(
            {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}
        )
    if set(row["trace_id"] for row in scores) != set(trace_by_id):
        raise ValueError("CTV scores and candidate pool have different trace support.")
    train_ids = set(
        read_json(_resolve(args.questions_dir) / "question_splits.json")["train"]
    )
    grouped = defaultdict(list)
    for score in scores:
        if score["problem_id"] in train_ids:
            grouped[score["problem_id"]].append(score)
    if set(grouped) != train_ids:
        raise ValueError("Policy selection is missing training questions.")
    seed = int(config["policy_sft"]["screen_seed"])
    selected_by_policy = {policy: [] for policy in POLICIES}
    for problem_id in sorted(grouped):
        rows = grouped[problem_id]
        choices = {
            "ctv": max(
                rows,
                key=lambda row: (
                    float(row["ctv_gradient_global_token_mean"]),
                    row["trace_id"],
                ),
            ),
            "rsr": min(
                rows, key=lambda row: (float(row["rsr_rank_clip_100"]), row["trace_id"])
            ),
            "scas": select_scas_lowest_group(
                rows, score_field="scas_score", groups=5, seed=seed
            ),
            "lark_style": max(
                rows, key=lambda row: (float(row["lark_g_hat"]), row["trace_id"])
            ),
            "low_nll": min(
                rows, key=lambda row: (float(row["student_nll_mean"]), row["trace_id"])
            ),
            "absolute_shortest": min(
                rows,
                key=lambda row: (int(row["solution_token_count"]), row["trace_id"]),
            ),
            "quantile_short": select_candidate_roles(rows, seed)["quantile_short"],
            "random": min(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{seed}|{row['trace_id']}".encode()
                ).hexdigest(),
            ),
        }
        for policy, score in choices.items():
            trace = trace_by_id[score["trace_id"]]
            selected_by_policy[policy].append(
                {
                    "prompt": question_by_id[problem_id]["student_prompt"],
                    "completion": trace["solution"],
                    "problem_id": problem_id,
                    "trace_id": trace["trace_id"],
                    "selection_policy": policy,
                    "selection_score": _policy_score(policy, score),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    support = None
    for policy in POLICIES:
        rows = selected_by_policy[policy]
        current = {row["problem_id"] for row in rows}
        support = current if support is None else support
        if current != support or len(rows) != len(train_ids):
            raise ValueError(
                f"Policy {policy} does not have identical one-trace-per-question support."
            )
        path = output_dir / f"{policy}.jsonl"
        count = write_jsonl(path, rows)
        artifacts.append(
            {
                "policy": policy,
                "path": str(path),
                "sha256": file_sha256(path),
                "record_count": count,
            }
        )
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "policies": list(POLICIES),
        "problem_count": len(train_ids),
        "records_per_policy": len(train_ids),
        "identical_question_support": True,
        "one_trace_per_question": True,
        "score_manifests": score_manifests,
        "artifacts": artifacts,
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "policy_data_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "POLICY_DATA_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )


def _policy_score(policy, row):
    fields = {
        "ctv": "ctv_gradient_global_token_mean",
        "rsr": "rsr_rank_clip_100",
        "scas": "scas_score",
        "lark_style": "lark_g_hat",
        "low_nll": "student_nll_mean",
        "absolute_shortest": "solution_token_count",
        "quantile_short": "solution_token_count",
        "random": "trace_id",
    }
    return row[fields[policy]]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
