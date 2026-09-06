#!/usr/bin/env python3
"""Submit Phase 1 on a passed Gate 0, otherwise record a clean blocked state."""

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
        default="results/trace_length_observation_gate0_v1/formal/analysis/gate0_decision.json",
    )
    args = parser.parse_args()
    decision_path = _resolve(args.decision)
    marker_path = decision_path.parent / "ANALYSIS_COMPLETE"
    evidence = validated_gate_decision(
        PROJECT_ROOT,
        gate_name="gate0",
        decision_path=decision_path,
        completion_marker_path=marker_path,
    )
    completion = read_key_value_marker(decision_path.parents[2] / "PHASE0_COMPLETE")
    if (
        completion.get("status") != "complete"
        or completion.get("gate0_status") != evidence["status"]
        or completion.get("gate0_decision_sha256") != evidence["decision_sha256"]
    ):
        raise ValueError("Phase-0 completion marker is not bound to Gate 0.")
    if evidence["status"] == "passed":
        subprocess.run(
            [
                sys.executable,
                "scripts/1_0_submit_phase1_utility_core.py",
                "--submit-policy-continuation",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        return
    path = (
        PROJECT_ROOT
        / "results/phase1_teaching_utility_v1/submissions/PHASE1_BLOCKED_BY_GATE0"
    )
    write_text_exclusive(
        path,
        f"status=blocked\ngate0_decision_sha256={evidence['decision_sha256']}\n"
        "gate0_analysis_marker_sha256="
        f"{evidence['completion_marker_sha256']}\n",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
