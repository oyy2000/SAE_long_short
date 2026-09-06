#!/usr/bin/env python3
"""Submit the entry-authorized Phase-1 candidate, CTV, and exact-utility DAG."""

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
    parser.add_argument("--submit-policy-continuation", action="store_true")
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    if not args.dry_run:
        _prepare_formal_result_root()
    frozen = (
        PROJECT_ROOT
        / "results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json"
    )
    if not args.dry_run and not frozen.is_file():
        subprocess.run(
            [
                sys.executable,
                "scripts/1_0_freeze_phase1_protocol.py",
                "--config",
                str(config_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    config_arg = str(frozen if not args.dry_run else config_path)
    exports = f"ALL,PHASE1_CONFIG={config_arg}"
    jobs = []
    questions = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--export={exports}",
            "scripts/slurm/1_1_phase1_questions.sh",
        ],
        args.dry_run,
        "QUESTIONS",
    )
    jobs.append(_entry("questions", questions))
    generate = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c32",
            f"--dependency=afterok:{questions}",
            f"--export={exports}",
            "scripts/slurm/1_2_phase1_generate_extension.sh",
        ],
        args.dry_run,
        "GENERATE",
    )
    jobs.append(_entry("teacher_extension", generate))
    prepare = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--dependency=afterok:{generate}",
            f"--export={exports}",
            "scripts/slurm/1_3_phase1_prepare.sh",
        ],
        args.dry_run,
        "PREPARE",
    )
    jobs.append(_entry("candidate_pool_and_inputs", prepare))
    calibrate = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c49",
            f"--dependency=afterok:{prepare}",
            f"--export={exports}",
            "scripts/slurm/1_4_phase1_calibrate_local.sh",
        ],
        args.dry_run,
        "CALIBRATE",
    )
    jobs.append(_entry("calibration_and_local_probes", calibrate))
    score = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c32",
            f"--dependency=afterok:{calibrate}",
            f"--export={exports}",
            "scripts/slurm/1_5_phase1_score.sh",
        ],
        args.dry_run,
        "SCORE",
    )
    exact = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c49",
            f"--dependency=afterok:{calibrate}",
            f"--export={exports}",
            "scripts/slurm/1_6_phase1_exact.sh",
        ],
        args.dry_run,
        "EXACT",
    )
    jobs.extend([_entry("all_candidate_scores", score), _entry("exact_dev_ctv", exact)])
    analyze = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--dependency=afterok:{score}:{exact}",
            f"--export={exports}",
            "scripts/slurm/1_7_phase1_analyze_prepare.sh",
        ],
        args.dry_run,
        "ANALYZE",
    )
    jobs.append(_entry("ctv_analysis_and_policy_data", analyze))
    continuation = None
    if args.submit_policy_continuation:
        continuation = _submit(
            [
                "sbatch",
                "--parsable",
                "--partition=a6000",
                "--nodelist=c31",
                f"--dependency=afterok:{analyze}",
                f"--export={exports}",
                "scripts/slurm/1_0_phase1_core_to_policy.sh",
            ],
            args.dry_run,
            "POLICY_CONTINUATION",
        )
        jobs.append(_entry("submit_policy_sft_continuation", continuation))
    terminal_job = continuation or analyze
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "experiment_name": config["experiment_name"],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_config": config_arg,
        "jobs": jobs,
        "terminal_job_id": terminal_job,
        "next_stage": (
            "policy SFT continuation registered"
            if continuation
            else "submit policy SFT only after this DAG completes"
        ),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    output = PROJECT_ROOT / "results/phase1_teaching_utility_v1/submissions"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"core_submission_{terminal_job}.json"
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


def _submit(command, dry_run, label):
    if dry_run:
        return label
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {command!r} stderr={result.stderr!r}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def _entry(stage, job_id):
    return {"stage": stage, "job_id": job_id}


def _prepare_formal_result_root():
    project = PROJECT_ROOT / "results/phase1_teaching_utility_v1/formal"
    beegfs = Path(
        "/mnt/beegfs/youyang7/projects/SAE_long_short/phase1_teaching_utility_v1/results/formal"
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


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
