#!/usr/bin/env python3
"""Inspect the nonblocking Phase-0 robustness and claim-scope gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--output",
        default="results/trace_length_observation_gate0_v1/GATE0_CLAIM_SCOPE.json",
    )
    parser.add_argument(
        "--authorize",
        action="store_true",
        help="Write --output after a pass; the default is read-only.",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        gate = require_passed_gate(PROJECT_ROOT, config["phase0_robustness_audit"])
    except RuntimeError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2))
        raise SystemExit(2) from None
    payload = {
        "status": "claim_scope_authorized",
        "phase": "phase0_robustness_audit",
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config_hash": canonical_sha256(config),
        "gate_evidence": gate,
        "authorized_claim": "short advantage survives registered budget and normalization controls",
    }
    if args.authorize:
        write_json_exclusive(_resolve(args.output), payload)
    else:
        print(json.dumps(payload, indent=2))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
