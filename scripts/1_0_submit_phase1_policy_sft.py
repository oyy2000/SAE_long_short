#!/usr/bin/env python3
"""Submit the Phase-1 policy SFT, anchor-sensitivity, and Gate-1 DAG."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-phase1-5-continuation", action="store_true")
    parser.add_argument(
        "--config",
        default="results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    for required in (
        "results/phase1_teaching_utility_v1/formal/ctv_analysis/CTV_ANALYSIS_COMPLETE",
        "results/phase1_teaching_utility_v1/formal/policy_data/POLICY_DATA_COMPLETE",
        "results/phase1_teaching_utility_v1/formal/mid_sft_anchor_data/DATA_COMPLETE",
    ):
        if not _resolve(required).is_file():
            raise FileNotFoundError(required)
    if not args.dry_run:
        _prepare_checkpoint_root()
    exports = f"ALL,PHASE1_CONFIG={config_path}"
    jobs = []
    screen_train = _submit(
        "a5000ada",
        "c32",
        None,
        exports,
        "scripts/slurm/1_8_phase1_train_screen.sh",
        args.dry_run,
        "SCREEN_TRAIN",
    )
    jobs.append(_entry("screen_training_and_mid_anchor", screen_train))
    screen_eval = _submit(
        "a5000ada",
        "c49",
        screen_train,
        exports,
        "scripts/slurm/1_9_phase1_eval_screen.sh",
        args.dry_run,
        "SCREEN_EVAL",
    )
    jobs.append(_entry("screen_dev_evaluation", screen_eval))
    select = _submit(
        "a6000",
        "c31",
        screen_eval,
        exports,
        "scripts/slurm/1_10_phase1_select_confirm.sh",
        args.dry_run,
        "SELECT",
    )
    jobs.append(_entry("confirmation_policy_selection", select))
    confirm_train = _submit(
        "a5000ada",
        "c32",
        select,
        exports,
        "scripts/slurm/1_11_phase1_train_confirm.sh",
        args.dry_run,
        "CONFIRM_TRAIN",
    )
    jobs.append(_entry("confirmation_training", confirm_train))
    confirm_eval = _submit(
        "a5000ada",
        "c49",
        confirm_train,
        exports,
        "scripts/slurm/1_12_phase1_eval_confirm.sh",
        args.dry_run,
        "CONFIRM_EVAL",
    )
    jobs.append(_entry("locked_test_evaluation", confirm_eval))
    sensitivity = _submit(
        "a5000ada",
        "c32",
        screen_train,
        exports,
        "scripts/slurm/1_13_phase1_mid_sensitivity.sh",
        args.dry_run,
        "SENSITIVITY",
    )
    jobs.append(_entry("mid_sft_anchor_sensitivity", sensitivity))
    final_dependency = f"{confirm_eval}:{sensitivity}"
    final = _submit(
        "a6000",
        "c31",
        final_dependency,
        exports,
        "scripts/slurm/1_14_phase1_final_analysis.sh",
        args.dry_run,
        "FINAL",
    )
    jobs.append(_entry("gate1_analysis", final))
    continuation = None
    if args.submit_phase1_5_continuation:
        continuation = _submit(
            "a6000",
            "c31",
            final,
            exports,
            "scripts/slurm/1_0_phase1_to_phase1_5.sh",
            args.dry_run,
            "PHASE1_5_CONTINUATION",
        )
        jobs.append(_entry("submit_phase1_5_prep_continuation", continuation))
    terminal_job = continuation or final
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "experiment_name": config["experiment_name"],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
        "terminal_job_id": terminal_job,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    root = PROJECT_ROOT / "results/phase1_teaching_utility_v1/submissions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"policy_submission_{terminal_job}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "submitted",
                "terminal_job_id": terminal_job,
                "manifest": str(path),
            },
            indent=2,
        )
    )


def _submit(partition, node, dependency, exports, script, dry_run, label):
    command = ["sbatch", "--parsable", f"--partition={partition}", f"--nodelist={node}"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.extend([f"--export={exports}", script])
    if dry_run:
        return label
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {command!r} stderr={result.stderr!r}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def _prepare_checkpoint_root():
    project = PROJECT_ROOT / "checkpoints/phase1_teaching_utility_v1"
    beegfs = Path(
        "/mnt/beegfs/youyang7/projects/SAE_long_short/phase1_teaching_utility_v1/checkpoints"
    )
    beegfs.mkdir(parents=True, exist_ok=True)
    if project.is_symlink():
        if project.resolve() != beegfs:
            raise ValueError(f"Checkpoint symlink target mismatch: {project.resolve()}")
    elif project.exists():
        raise FileExistsError(f"Checkpoint path must be a BeeGFS symlink: {project}")
    else:
        project.parent.mkdir(parents=True, exist_ok=True)
        project.symlink_to(beegfs, target_is_directory=True)


def _entry(stage, job_id):
    return {"stage": stage, "job_id": job_id}


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
