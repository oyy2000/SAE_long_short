#!/usr/bin/env python3
"""Submit Phase-1.5 confirmation after both registered gates pass."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, require_passed_gate
from length_budget_distill.factorial import file_sha256, read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config",
        default="results/phase1_5_credit_allocation_v1/formal/protocol/frozen_protocol.json",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    pilot = require_passed_gate(PROJECT_ROOT, config["pilot_gate_dependency"])
    _validate_step_audit(config)
    if not args.dry_run:
        _validate_checkpoint_root()
    exports = (
        f"ALL,PHASE1_5_CONFIG={config_path},"
        "PHASE1_CONFIG=results/phase1_teaching_utility_v1/formal/protocol/"
        "frozen_protocol.json"
    )
    jobs = []
    confirm_train = _submit(
        "a5000ada",
        "c32",
        None,
        exports,
        "scripts/slurm/1_20_phase1_5_train_confirm.sh",
        args.dry_run,
        "CONFIRM_TRAIN",
    )
    jobs.append({"stage": "confirmation_training", "job_id": confirm_train})
    confirm_eval = _submit(
        "a5000ada",
        "c49",
        confirm_train,
        exports,
        "scripts/slurm/1_21_phase1_5_eval_confirm.sh",
        args.dry_run,
        "CONFIRM_EVAL",
    )
    jobs.append({"stage": "confirmation_evaluation", "job_id": confirm_eval})
    final = _submit(
        "a6000",
        "c31",
        confirm_eval,
        exports,
        "scripts/slurm/1_22_phase1_5_final_analysis.sh",
        args.dry_run,
        "FINAL",
    )
    jobs.append({"stage": "gate1_5_analysis", "job_id": final})
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "gate1_evidence": gate1,
        "pilot_evidence": pilot,
        "jobs": jobs,
        "terminal_job_id": final,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    root = PROJECT_ROOT / "results/phase1_5_credit_allocation_v1/submissions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"confirmation_submission_{final}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {"status": "submitted", "terminal_job_id": final, "manifest": str(path)},
            indent=2,
        )
    )


def _validate_step_audit(config: dict) -> None:
    approval_path = _resolve(config["segmentation"]["approval_marker_path"])
    approval = read_key_value_marker(approval_path)
    if approval.get("status") != "approved":
        raise RuntimeError("Step audit has not been explicitly approved.")
    manifest_path = _resolve(
        "results/phase1_5_credit_allocation_v1/formal/data/"
        "credit_allocation_data_manifest.json"
    )
    if approval.get("data_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Step-audit approval is bound to another data manifest.")


def _validate_checkpoint_root() -> None:
    project = PROJECT_ROOT / "checkpoints/phase1_5_credit_allocation_v1"
    expected = Path(
        "/mnt/beegfs/youyang7/projects/SAE_long_short/"
        "phase1_5_credit_allocation_v1/checkpoints"
    )
    if not project.is_symlink() or project.resolve() != expected:
        raise ValueError("Phase-1.5 checkpoint root is not the registered BeeGFS link.")


def _submit(partition, node, dependency, exports, script, dry_run, label):
    command = ["sbatch", "--parsable", f"--partition={partition}", f"--nodelist={node}"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.extend([f"--export={exports}", script])
    if dry_run:
        return label
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"sbatch failed: {result.stderr!r}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
