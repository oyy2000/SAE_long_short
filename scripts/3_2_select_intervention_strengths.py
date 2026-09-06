#!/usr/bin/env python3
"""Audit calibration shards and select short-enhance and long-suppress strengths."""

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
from length_budget_distill.sae_intervention_sweep import (
    load_calibration_shards as _load_shards,
    summarize_condition as _condition_summary,
)


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
    output_dir = result_root / "calibration"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    shard_root = result_root / "calibration_shards"
    shard_count = int(config["calibration"]["generation_shards"])
    records, evidence = _load_shards(shard_root, shard_count, config)
    expected_ids = set(config["question_cohorts"]["calibration_problem_ids"])
    observed_ids = {row["problem_id"] for row in records}
    if observed_ids != expected_ids:
        raise ValueError("Calibration question coverage differs from the frozen cohort.")
    conditions = sorted({row["condition"] for row in records})
    expected_per_condition = len(expected_ids) * int(
        config["calibration"]["candidates_per_question"]
    )
    counts = Counter(row["condition"] for row in records)
    if any(counts[name] != expected_per_condition for name in conditions):
        raise ValueError("Calibration condition coverage is incomplete.")
    summaries = [_condition_summary(name, records) for name in conditions]
    by_name = {row["condition"]: row for row in summaries}
    baseline = by_name["no_steering"]
    maximum_drop = float(
        config["calibration"]["maximum_accuracy_drop_from_no_steering"]
    )
    selected = {}
    for family, prefix in (
        ("short_feature_enhance", "short_feature_enhance__"),
        ("long_feature_suppress", "long_feature_suppress__"),
    ):
        candidates = [row for row in summaries if row["condition"].startswith(prefix)]
        eligible = [
            row
            for row in candidates
            if float(row["accuracy"])
            >= float(baseline["accuracy"]) - maximum_drop
        ]
        pool = eligible if eligible else candidates
        chosen = sorted(
            pool,
            key=lambda row: (
                -float(row["correct_usable_per_million_tokens"]),
                -float(row["accuracy"]),
                float(row["mean_correct_output_tokens"]),
                float(row["strength"]),
            ),
        )[0]
        selected[family] = {
            **chosen,
            "passed_accuracy_guard": bool(chosen in eligible),
            "accuracy_guard": float(baseline["accuracy"]) - maximum_drop,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    merged_path = output_dir / "calibration_generations.jsonl"
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
    metrics_path = output_dir / "calibration_metrics.csv"
    _write_csv(metrics_path, summaries)
    selection = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "selection_metric": config["calibration"]["selection_metric"],
        "baseline": baseline,
        "selected": selected,
        "condition_metrics": summaries,
        "input_shards": evidence,
        "artifacts": {
            merged_path.name: {
                "path": str(merged_path),
                "sha256": file_sha256(merged_path),
            },
            metrics_path.name: {
                "path": str(metrics_path),
                "sha256": file_sha256(metrics_path),
            },
        },
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    selection_path = output_dir / "selected_strengths.json"
    write_json_exclusive(selection_path, selection)
    (output_dir / "CALIBRATION_COMPLETE").write_text(
        f"status=complete\nconfig_hash={selection['config_hash']}\n"
        f"selection_sha256={file_sha256(selection_path)}\n"
        f"short_strength={selected['short_feature_enhance']['strength']}\n"
        f"long_strength={selected['long_feature_suppress']['strength']}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2), flush=True)


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
