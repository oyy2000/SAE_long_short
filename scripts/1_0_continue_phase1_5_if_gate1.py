#!/usr/bin/env python3
"""Submit Phase-1.5 preparation on passed Gate 1 or record a blocked state."""

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
from length_budget_distill.factorial import read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        default="results/phase1_teaching_utility_v1/formal/analysis/gate1_decision.json",
    )
    args = parser.parse_args()
    decision_path = _resolve(args.decision)
    marker_path = decision_path.parent / "ANALYSIS_COMPLETE"
    evidence = validated_gate_decision(
        PROJECT_ROOT,
        gate_name="gate1",
        decision_path=decision_path,
        completion_marker_path=marker_path,
    )
    completion = read_key_value_marker(decision_path.parents[2] / "PHASE1_COMPLETE")
    if (
        completion.get("status") != "complete"
        or completion.get("gate1_status") != evidence["status"]
        or completion.get("gate1_decision_sha256") != evidence["decision_sha256"]
    ):
        raise ValueError("Phase-1 completion marker is not bound to Gate 1.")
    if evidence["status"] == "passed":
        subprocess.run(
            [sys.executable, "scripts/1_0_submit_phase1_5_prep.py"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        return
    path = (
        PROJECT_ROOT
        / "results/phase1_5_credit_allocation_v1/submissions/PHASE1_5_BLOCKED_BY_GATE1"
    )
    write_text_exclusive(
        path,
        f"status=blocked\ngate1_decision_sha256={evidence['decision_sha256']}\n"
        "gate1_analysis_marker_sha256="
        f"{evidence['completion_marker_sha256']}\n",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
