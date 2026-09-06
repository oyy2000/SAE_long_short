#!/usr/bin/env python3
"""Build the exact first-half random-policy stream for the mid-SFT CTV anchor."""

from __future__ import annotations

import argparse
import math
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
from length_budget_distill.records import read_jsonl, write_jsonl
from trace_length_observation.controlled_sft import deterministic_training_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--random-policy-data",
        default="results/phase1_teaching_utility_v1/formal/policy_data/random.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1_teaching_utility_v1/formal/mid_sft_anchor_data",
    )
    args = parser.parse_args()
    config_path, source_path, output_dir = (
        _resolve(args.config),
        _resolve(args.random_policy_data),
        _resolve(args.output_dir),
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    seed = int(config["policy_sft"]["screen_seed"])
    rows = deterministic_training_order(list(read_jsonl(source_path)), seed)
    batch_size = int(config["policy_sft"]["training"]["per_device_train_batch_size"])
    full_steps = math.ceil(len(rows) / batch_size)
    half_steps = math.ceil(0.5 * full_steps)
    selected = rows[: min(len(rows), half_steps * batch_size)]
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "random_correct_first_half.jsonl"
    count = write_jsonl(path, selected)
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "seed": seed,
        "batch_size": batch_size,
        "full_optimizer_steps": full_steps,
        "mid_optimizer_steps": half_steps,
        "mid_step_fraction": half_steps / full_steps,
        "record_count": count,
        "data_path": str(path),
        "data_sha256": file_sha256(path),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "mid_sft_anchor_data_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "DATA_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        f"data_sha256={manifest['data_sha256']}\nmid_optimizer_steps={half_steps}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
