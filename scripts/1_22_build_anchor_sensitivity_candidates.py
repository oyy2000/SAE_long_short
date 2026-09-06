#!/usr/bin/env python3
"""Select the registered 100-by-4 dev subset for mid-SFT anchor sensitivity."""

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
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.teaching_utility import stable_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--exact-candidates",
        default="results/phase1_teaching_utility_v1/formal/ctv_inputs/exact_ctv_candidates.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1_teaching_utility_v1/formal/anchor_sensitivity_inputs",
    )
    args = parser.parse_args()
    config_path, source_path, output_dir = (
        _resolve(args.config),
        _resolve(args.exact_candidates),
        _resolve(args.output_dir),
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    rows = list(read_jsonl(source_path))
    ids = sorted(
        {row["problem_id"] for row in rows},
        key=lambda value: stable_hash(
            int(config["dataset"]["question_split"]["seed"]), "mid-anchor", value
        ),
    )[: int(config["anchors"]["full_sensitivity_sample_questions"])]
    selected = [dict(row) for row in rows if row["problem_id"] in set(ids)]
    expected = len(ids) * int(config["anchors"]["candidates_per_question"])
    if len(selected) != expected:
        raise ValueError(
            f"Anchor sensitivity cardinality mismatch: {len(selected)} != {expected}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "mid_sft_exact_candidates.jsonl"
    write_jsonl(path, selected)
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "question_count": len(ids),
        "candidate_count": len(selected),
        "candidate_path": str(path),
        "candidate_sha256": file_sha256(path),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "anchor_sensitivity_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "ANCHOR_SENSITIVITY_INPUTS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        f"candidate_sha256={manifest['candidate_sha256']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
