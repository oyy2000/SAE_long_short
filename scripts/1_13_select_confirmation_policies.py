#!/usr/bin/env python3
"""Choose the strongest non-CTV policy from the registered seed-17 dev screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument("--screen-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/phase1_teaching_utility_v1/formal/confirmation_plan",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    root = _resolve(args.screen_root)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    expected = set(config["policy_sft"]["screen_policies"])
    summaries = []
    for path in sorted(root.glob("*/evaluation_summary.json")):
        row = read_json(path)
        if row.get("status") != "complete" or row.get("split") != "dev":
            raise ValueError(f"Invalid screening evaluation: {path}")
        if int(row["seed"]) != int(config["policy_sft"]["screen_seed"]):
            raise ValueError(f"Screening evaluation uses the wrong seed: {path}")
        if file_sha256(Path(row["predictions_path"])) != row["predictions_sha256"]:
            raise ValueError(f"Screening prediction hash mismatch: {path}")
        summaries.append(
            {**row, "summary_path": str(path), "summary_sha256": file_sha256(path)}
        )
    observed = {str(row["policy"]) for row in summaries}
    if observed != expected or len(summaries) != len(expected):
        raise ValueError(
            f"Expected screening policies {sorted(expected)}, observed {sorted(observed)}"
        )
    supports = {str(row["problem_ids_sha256"]) for row in summaries}
    if len(supports) != 1:
        raise ValueError("Screening evaluations do not share question support.")
    eligible = [row for row in summaries if row["policy"] != "ctv"]
    strongest = max(
        eligible,
        key=lambda row: (
            float(row["accuracy"]),
            -float(row["mean_output_tokens"]),
            str(row["policy"]),
        ),
    )
    labels = {
        "ctv": "ctv",
        "strongest_non_ctv": str(strongest["policy"]),
        "quantile_short": "quantile_short",
        "random": "random",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "screen_seed": int(config["policy_sft"]["screen_seed"]),
        "selection_split": "dev",
        "selection_rule": "maximum_accuracy_then_minimum_mean_output_tokens_then_policy_name",
        "strongest_non_ctv_policy": str(strongest["policy"]),
        "confirmation_label_to_policy": labels,
        "unique_confirmation_policies": sorted(set(labels.values())),
        "confirmation_seeds": list(config["policy_sft"]["confirm_seeds"]),
        "screening_summaries": summaries,
        "problem_ids_sha256": next(iter(supports)),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    path = output_dir / "confirmation_plan.json"
    write_json_exclusive(path, plan)
    (output_dir / "CONFIRMATION_PLAN_COMPLETE").write_text(
        f"status=complete\nconfig_hash={plan['config_hash']}\nplan_sha256={file_sha256(path)}\n"
        f"strongest_non_ctv_policy={plan['strongest_non_ctv_policy']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
