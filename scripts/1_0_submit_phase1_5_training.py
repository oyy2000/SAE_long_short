#!/usr/bin/env python3
"""Submit the Phase-1.5 pilot and its decision-gated continuation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from length_budget_distill.experiment_io import read_json, require_passed_gate
from length_budget_distill.factorial import file_sha256, read_key_value_marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config",
        default="results/phase1_5_credit_allocation_v1/formal/protocol/frozen_protocol.json",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    approval = _resolve(config["segmentation"]["approval_marker_path"])
    approval_fields = read_key_value_marker(approval) if approval.is_file() else {}
    if approval_fields.get("status") != "approved":
        raise RuntimeError("Step audit has not been explicitly approved.")
    data_manifest = _resolve(
        "results/phase1_5_credit_allocation_v1/formal/data/"
        "credit_allocation_data_manifest.json"
    )
    if approval_fields.get("data_manifest_sha256") != file_sha256(data_manifest):
        raise ValueError("Step-audit approval is bound to another data manifest.")
    if not args.dry_run:
        _prepare_checkpoint_root()
    exports = f"ALL,PHASE1_5_CONFIG={config_path},PHASE1_CONFIG=results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json"
    jobs = []
    pilot_train = _submit(
        "a5000ada",
        "c32",
        None,
        exports,
        "scripts/slurm/1_17_phase1_5_train_pilot.sh",
        args.dry_run,
        "PILOT_TRAIN",
    )
    jobs.append({"stage": "pilot_training", "job_id": pilot_train})
    pilot_eval = _submit(
        "a5000ada",
        "c49",
        pilot_train,
        exports,
        "scripts/slurm/1_18_phase1_5_eval_pilot.sh",
        args.dry_run,
        "PILOT_EVAL",
    )
    jobs.append({"stage": "pilot_evaluation", "job_id": pilot_eval})
    pilot_analysis = _submit(
        "a6000",
        "c31",
        pilot_eval,
        exports,
        "scripts/slurm/1_19_phase1_5_analyze_pilot.sh",
        args.dry_run,
        "PILOT_ANALYSIS",
    )
    jobs.append({"stage": "pilot_analysis", "job_id": pilot_analysis})
    continuation = _submit(
        "a6000",
        "c31",
        pilot_analysis,
        exports,
        "scripts/slurm/1_0_phase1_5_pilot_to_confirm.sh",
        args.dry_run,
        "PILOT_CONTINUATION",
    )
    jobs.append(
        {
            "stage": "submit_confirmation_if_pilot_passes",
            "job_id": continuation,
        }
    )
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "jobs": jobs,
        "terminal_job_id": continuation,
        "next_stage": "confirmation is submitted only by the pilot-gate continuation",
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    root = PROJECT_ROOT / "results/phase1_5_credit_allocation_v1/submissions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"pilot_submission_{continuation}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "submitted",
                "terminal_job_id": continuation,
                "manifest": str(path),
            },
            indent=2,
        )
    )


def _submit(partition, node, dependency, exports, script, dry, label):
    command = ["sbatch", "--parsable", f"--partition={partition}", f"--nodelist={node}"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.extend([f"--export={exports}", script])
    if dry:
        return label
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    job = result.stdout.strip().split(";", 1)[0]
    if not job.isdigit():
        raise ValueError(result.stdout)
    return job


def _prepare_checkpoint_root():
    project = PROJECT_ROOT / "checkpoints/phase1_5_credit_allocation_v1"
    beegfs = Path(
        "/mnt/beegfs/youyang7/projects/SAE_long_short/phase1_5_credit_allocation_v1/checkpoints"
    )
    beegfs.mkdir(parents=True, exist_ok=True)
    if project.is_symlink():
        if project.resolve() != beegfs:
            raise ValueError("Checkpoint symlink target mismatch.")
    elif project.exists():
        raise FileExistsError(project)
    else:
        project.parent.mkdir(parents=True, exist_ok=True)
        project.symlink_to(beegfs, target_is_directory=True)


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
