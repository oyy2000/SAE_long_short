#!/usr/bin/env python3
"""Submit Gate-1-guarded Phase-1.5 step scoring and data construction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from length_budget_distill.experiment_io import read_json, require_passed_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config", default="configs/phase1_5_credit_allocation_v1.json"
    )
    args = parser.parse_args()
    source = _resolve(args.config)
    config = read_json(source)
    require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    if not args.dry_run:
        _prepare_formal_result_root()
    frozen = (
        PROJECT_ROOT
        / "results/phase1_5_credit_allocation_v1/formal/protocol/frozen_protocol.json"
    )
    if not args.dry_run and not frozen.is_file():
        subprocess.run(
            [
                sys.executable,
                "scripts/1_0_freeze_phase1_5_protocol.py",
                "--config",
                str(source),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    selected = frozen if not args.dry_run else source
    exports = f"ALL,PHASE1_5_CONFIG={selected},PHASE1_CONFIG=results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json"
    step = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c32",
            f"--export={exports}",
            "scripts/slurm/1_15_phase1_5_step_score.sh",
        ],
        args.dry_run,
        "STEP",
    )
    data = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--dependency=afterok:{step}",
            f"--export={exports}",
            "scripts/slurm/1_16_phase1_5_build_data.sh",
        ],
        args.dry_run,
        "DATA",
    )
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "jobs": [
            {"stage": "step_score", "job_id": step},
            {"stage": "build_data", "job_id": data},
        ],
        "terminal_job_id": data,
        "next_required_action": "inspect step_audit_sample.jsonl and record STEP_AUDIT_APPROVED",
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    root = PROJECT_ROOT / "results/phase1_5_credit_allocation_v1/submissions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"prep_submission_{data}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {"status": "submitted", "terminal_job_id": data, "manifest": str(path)},
            indent=2,
        )
    )


def _submit(command, dry, label):
    if dry:
        return label
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    job = result.stdout.strip().split(";", 1)[0]
    if not job.isdigit():
        raise ValueError(result.stdout)
    return job


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _prepare_formal_result_root():
    project = PROJECT_ROOT / "results/phase1_5_credit_allocation_v1/formal"
    beegfs = Path(
        "/mnt/beegfs/youyang7/projects/SAE_long_short/phase1_5_credit_allocation_v1/results/formal"
    )
    beegfs.mkdir(parents=True, exist_ok=True)
    project.parent.mkdir(parents=True, exist_ok=True)
    if project.is_symlink():
        if project.resolve() != beegfs:
            raise ValueError(
                f"Formal-result symlink target mismatch: {project.resolve()}"
            )
    elif project.exists():
        raise FileExistsError(f"Formal result root must be a BeeGFS symlink: {project}")
    else:
        project.symlink_to(beegfs, target_is_directory=True)


if __name__ == "__main__":
    main()
