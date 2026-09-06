#!/usr/bin/env python3
"""Submit Phase-1.5 confirmation on a passed pilot or record a blocked state."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    validated_gate_decision,
    write_text_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        default="results/phase1_5_credit_allocation_v1/formal/pilot_analysis/pilot_decision.json",
    )
    args = parser.parse_args()
    decision_path = _resolve(args.decision)
    marker_path = decision_path.parent / "PILOT_ANALYSIS_COMPLETE"
    evidence = validated_gate_decision(
        PROJECT_ROOT,
        gate_name="pilot",
        decision_path=decision_path,
        completion_marker_path=marker_path,
    )
    if evidence["status"] == "passed":
        subprocess.run(
            [sys.executable, "scripts/1_0_submit_phase1_5_confirmation.py"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        return
    path = (
        PROJECT_ROOT
        / "results/phase1_5_credit_allocation_v1/submissions/CONFIRMATION_BLOCKED_BY_PILOT"
    )
    write_text_exclusive(
        path,
        f"status=blocked\npilot_decision_sha256={evidence['decision_sha256']}\n"
        "pilot_analysis_marker_sha256="
        f"{evidence['completion_marker_sha256']}\n",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
