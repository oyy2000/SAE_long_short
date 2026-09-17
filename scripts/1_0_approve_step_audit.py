#!/usr/bin/env python3
"""Record an explicit reviewer approval after inspecting the 100-question step sample."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json
from length_budget_distill.factorial import file_sha256, read_key_value_marker
from length_budget_distill.records import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase1_5_credit_allocation_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument(
        "--data-manifest",
        default="results/phase1_5_credit_allocation_v1/formal/data/credit_allocation_data_manifest.json",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    if any("\n" in value or "\r" in value for value in (args.reviewer, args.notes)):
        raise ValueError("Reviewer and notes must each fit on one marker line.")
    config_path, manifest_path = _resolve(args.config), _resolve(args.data_manifest)
    config, manifest = read_json(config_path), read_json(manifest_path)
    data_marker_path = manifest_path.parent / "DATA_COMPLETE"
    data_marker = read_key_value_marker(data_marker_path)
    if (
        manifest.get("status") != "complete"
        or data_marker.get("status") != "complete"
        or data_marker.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("Credit-allocation data is not marker-bound.")
    sample_path = Path(manifest["manual_audit_path"])
    if file_sha256(sample_path) != manifest["manual_audit_sha256"]:
        raise ValueError("Manual-audit sample hash mismatch.")
    observed = sum(1 for _ in read_jsonl(sample_path))
    expected = min(
        int(config["segmentation"]["manual_audit_questions"]),
        int(manifest["retained_common_support_count"]),
    )
    if observed != expected:
        raise ValueError(
            f"Manual-audit sample count mismatch: {observed} != {expected}"
        )
    marker_path = _resolve(config["segmentation"]["approval_marker_path"])
    if marker_path.exists():
        raise FileExistsError(marker_path)
    marker_path.write_text(
        f"status=approved\nreviewer={args.reviewer}\nreviewed_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"config_sha256={file_sha256(config_path)}\ndata_manifest_sha256={file_sha256(manifest_path)}\n"
        f"sample_sha256={file_sha256(sample_path)}\nsample_count={observed}\nnotes={args.notes}\n",
        encoding="utf-8",
    )


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
