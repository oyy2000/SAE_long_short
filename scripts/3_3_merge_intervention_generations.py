#!/usr/bin/env python3
"""Audit and merge the main common-prefix intervention generation shards."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase3_sae_intervention_distillation_pilot_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    result_root = _resolve(config["outputs"]["result_root"])
    output_dir = result_root / "main_generation"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    shard_count = int(config["main_generation"]["generation_shards"])
    records, evidence = _load_shards(
        result_root / "main_generation_shards", shard_count, config
    )
    conditions = list(config["main_generation"]["conditions"])
    expected_ids = set(config["question_cohorts"]["main_problem_ids"])
    candidates = int(config["main_generation"]["candidates_per_question"])
    identities = Counter(
        (row["condition"], row["problem_id"], int(row["candidate_index"]))
        for row in records
    )
    expected_count = len(expected_ids) * len(conditions) * candidates
    if len(records) != expected_count or any(value != 1 for value in identities.values()):
        raise ValueError("Main generation has missing or duplicate condition cells.")
    if {row["problem_id"] for row in records} != expected_ids:
        raise ValueError("Main generation question cohort mismatch.")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["problem_id"])].append(row)
    selected = {}
    coverage = {}
    summaries = []
    no_steering_by_pair = {
        row["pair_id"]: row for row in records if row["condition"] == "no_steering"
    }
    for condition in conditions:
        condition_rows = [row for row in records if row["condition"] == condition]
        correct_rows = [row for row in condition_rows if bool(row["is_correct"])]
        covered_ids = []
        for problem_id in sorted(expected_ids):
            correct = sorted(
                (
                    row
                    for row in grouped[(condition, problem_id)]
                    if bool(row["is_correct"])
                ),
                key=lambda row: int(row["candidate_index"]),
            )
            if correct:
                selected[(condition, problem_id)] = correct[0]
                covered_ids.append(problem_id)
        coverage[condition] = set(covered_ids)
        total_tokens = sum(int(row["output_token_count"]) for row in condition_rows)
        changed_from_no_steering = sum(
            row["response_token_ids"]
            != no_steering_by_pair[row["pair_id"]]["response_token_ids"]
            for row in condition_rows
        )
        paired_length_differences = [
            int(row["output_token_count"])
            - int(no_steering_by_pair[row["pair_id"]]["output_token_count"])
            for row in condition_rows
        ]
        diagnostics = [row["intervention_diagnostics"] for row in condition_rows]
        summaries.append(
            {
                "condition": condition,
                "generated_trace_count": len(condition_rows),
                "unconditional_correct_count": len(correct_rows),
                "unconditional_accuracy": len(correct_rows) / len(condition_rows),
                "question_coverage_count": len(covered_ids),
                "question_coverage": len(covered_ids) / len(expected_ids),
                "total_generated_tokens": total_tokens,
                "mean_output_tokens": statistics.fmean(
                    int(row["output_token_count"]) for row in condition_rows
                ),
                "mean_correct_output_tokens": statistics.fmean(
                    int(row["output_token_count"]) for row in correct_rows
                )
                if correct_rows
                else None,
                "correct_usable_traces_per_million_tokens": len(correct_rows)
                / max(total_tokens, 1)
                * 1_000_000,
                "changed_from_no_steering_fraction": changed_from_no_steering
                / len(condition_rows),
                "mean_paired_output_token_delta_vs_no_steering": statistics.fmean(
                    paired_length_differences
                ),
                "mean_modified_continuation_fraction": statistics.fmean(
                    float(row["modified_tokens"])
                    / max(int(row["continuation_forward_tokens"]), 1)
                    for row in diagnostics
                ),
                "mean_delta_to_hidden_norm_fraction": statistics.fmean(
                    float(row["mean_delta_to_hidden_norm_fraction"])
                    for row in diagnostics
                ),
                "max_delta_to_hidden_norm_fraction": max(
                    float(row["max_delta_to_hidden_norm_fraction"])
                    for row in diagnostics
                ),
            }
        )
    common_ids = sorted(set.intersection(*(coverage[name] for name in conditions)))
    if not common_ids:
        raise RuntimeError("No all-condition correct common support remains.")
    output_dir.mkdir(parents=True, exist_ok=False)
    merged_path = output_dir / "merged_generations.jsonl"
    with merged_path.open("x", encoding="utf-8") as handle:
        for row in sorted(
            records,
            key=lambda value: (
                value["problem_id"],
                int(value["candidate_index"]),
                value["condition"],
            ),
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selected_path = output_dir / "common_support_selected_traces.jsonl"
    with selected_path.open("x", encoding="utf-8") as handle:
        for problem_id in common_ids:
            for condition in conditions:
                handle.write(
                    json.dumps(selected[(condition, problem_id)], ensure_ascii=False)
                    + "\n"
                )
    metrics_path = output_dir / "generation_condition_metrics.csv"
    _write_csv(metrics_path, summaries)
    common_ids_path = output_dir / "common_support_problem_ids.json"
    write_json_exclusive(common_ids_path, {"problem_ids": common_ids})
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "expected_question_count": len(expected_ids),
        "condition_count": len(conditions),
        "candidates_per_question": candidates,
        "expected_record_count": expected_count,
        "observed_record_count": len(records),
        "duplicate_cell_count": sum(value > 1 for value in identities.values()),
        "missing_cell_count": expected_count - len(identities),
        "condition_metrics": summaries,
        "common_support_question_count": len(common_ids),
        "common_support_fraction": len(common_ids) / len(expected_ids),
        "input_shards": evidence,
        "artifacts": {},
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    for path in (merged_path, selected_path, metrics_path, common_ids_path):
        manifest["artifacts"][path.name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    manifest_path = output_dir / "generation_merge_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "MAIN_GENERATION_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        f"record_count={len(records)}\ncommon_support_question_count={len(common_ids)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _load_shards(
    root: Path, shard_count: int, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    evidence = []
    source_hashes = set()
    library_hashes = set()
    for shard_index in range(shard_count):
        directory = root / f"shard_{shard_index:02d}_of_{shard_count:02d}"
        marker_path = directory / "GENERATION_SHARD_COMPLETE"
        manifest_path = directory / "generation_manifest.json"
        marker = read_key_value_marker(marker_path)
        manifest = read_json(manifest_path)
        if marker.get("status") != "complete" or marker.get("stage") != "main":
            raise ValueError(f"Incomplete main generation shard: {directory}")
        if marker.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError(f"Main generation manifest hash mismatch: {directory}")
        if manifest["config_hash"] != canonical_sha256(config):
            raise ValueError(f"Main generation config hash mismatch: {directory}")
        records_path = Path(manifest["records_path"])
        if file_sha256(records_path) != manifest["records_sha256"]:
            raise ValueError(f"Main generation record hash mismatch: {directory}")
        source_hashes.add(manifest["source_code_sha256"])
        library_hashes.add(manifest["library_code_sha256"])
        with records_path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
        evidence.append(
            {
                "shard_index": shard_index,
                "problem_count": int(manifest["problem_count"]),
                "pair_count": int(manifest["pair_count"]),
                "marker_path": str(marker_path),
                "marker_sha256": file_sha256(marker_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "records_path": str(records_path),
                "records_sha256": manifest["records_sha256"],
            }
        )
    if len(source_hashes) != 1 or len(library_hashes) != 1:
        raise ValueError("Main generation shards used inconsistent source hashes.")
    return records, evidence


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
