#!/usr/bin/env python3
"""Submit the dependency-ordered Phase-0 data, training, audit, and evaluation DAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import file_sha256, read_key_value_marker
from trace_length_observation.trace_observation import protocol_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing-data-job", default=None)
    parser.add_argument(
        "--existing-training-job",
        action="append",
        default=[],
        metavar="SHARD=JOB_ID",
        help="Reuse an already-submitted shard after a partial submission.",
    )
    parser.add_argument("--result-root", default="results/trace_length_observation_gate0_v1")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/trace_length_observation_gate0_v1_formal",
    )
    parser.add_argument(
        "--beegfs-checkpoint-root",
        default=(
            "/mnt/beegfs/youyang7/projects/SAE_long_short/"
            "trace_length_observation_gate0_v1/checkpoints/trace_length_observation_gate0_v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.existing_data_job is not None and not str(args.existing_data_job).isdigit():
        raise ValueError("--existing-data-job must be a numeric Slurm job ID.")
    existing_training_jobs: Dict[int, str] = {}
    for value in args.existing_training_job:
        shard_text, separator, job_id = value.partition("=")
        if not separator or not shard_text.isdigit() or not job_id.isdigit():
            raise ValueError("--existing-training-job must use SHARD=JOB_ID.")
        existing_training_jobs[int(shard_text)] = job_id
    os.chdir(PROJECT_ROOT)
    result_root = _resolve(args.result_root)
    checkpoint_root = _resolve(args.checkpoint_root)
    beegfs_checkpoint_root = Path(args.beegfs_checkpoint_root).resolve()
    required = [
        PROJECT_ROOT / "configs/trace_length_observation_gate0_v1.json",
        PROJECT_ROOT / "data/parent_dependencies.json",
        PROJECT_ROOT / "scripts/slurm/24_1_build_trace_observation_data.sh",
        PROJECT_ROOT / "scripts/slurm/24_3_train_trace_observation.sh",
        PROJECT_ROOT / "scripts/slurm/24_4_audit_trace_observation_training.sh",
        PROJECT_ROOT / "scripts/slurm/24_5_eval_analyze_trace_observation.sh",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase-0 submission inputs: {missing}")
    if (result_root / "PHASE0_COMPLETE").exists():
        raise FileExistsError(f"Phase 0 is already complete: {result_root}")
    if not args.dry_run:
        _require_frozen_protocol(result_root)
        _prepare_checkpoint_root(checkpoint_root, beegfs_checkpoint_root)
        (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    exports = (
        f"ALL,RESULT_ROOT={args.result_root},CHECKPOINT_ROOT={args.checkpoint_root},"
        "SFT_ENV=/mnt/beegfs/youyang7/.conda/envs/sft,"
        "FACT_ENV=/mnt/beegfs/youyang7/.conda/envs/fact"
    )
    submissions: List[Dict[str, Any]] = []
    data_command = [
        "sbatch",
        "--parsable",
        "--partition=a6000",
        "--nodelist=c31",
        f"--export={exports}",
        "scripts/slurm/24_1_build_trace_observation_data.sh",
    ]
    data_job = str(args.existing_data_job) if args.existing_data_job else _submit(
        data_command, args.dry_run, "DATA_JOB"
    )
    submissions.append(
        {
            "stage": "data",
            "job_id": data_job,
            "command": data_command,
            "submission_status": "reused_existing" if args.existing_data_job else "submitted",
        }
    )
    # C30 is currently saturated by an unrelated four-GPU process and C31 hosts
    # the user's persistent parser allocation plus unrelated GPU workloads.
    # Use C32 and C49; Slurm serializes the two C32 shards by CPU availability.
    nodes = (("c32", "a5000ada"), ("c49", "a5000ada"), ("c32", "a5000ada"))
    training_jobs: List[str] = []
    for shard, (node, partition) in enumerate(nodes):
        shard_exports = f"{exports},LAUNCHER_SHARDS=3,LAUNCHER_SHARD_INDEX={shard}"
        command = [
            "sbatch",
            "--parsable",
            f"--partition={partition}",
            f"--nodelist={node}",
            f"--dependency=afterok:{data_job}",
            f"--export={shard_exports}",
            "scripts/slurm/24_3_train_trace_observation.sh",
        ]
        if shard in existing_training_jobs:
            job_id = existing_training_jobs[shard]
            submission_status = "reused_existing"
        else:
            job_id = _submit(command, args.dry_run, f"TRAIN_{shard}_JOB")
            submission_status = "submitted"
        training_jobs.append(job_id)
        submissions.append(
            {
                "stage": "training",
                "shard": shard,
                "node": node,
                "job_id": job_id,
                "command": command,
                "submission_status": submission_status,
            }
        )
    audit_dependency = ":".join(training_jobs)
    audit_command = [
        "sbatch",
        "--parsable",
        "--partition=a6000",
        "--nodelist=c31",
        f"--dependency=afterok:{audit_dependency}",
        f"--export={exports}",
        "scripts/slurm/24_4_audit_trace_observation_training.sh",
    ]
    audit_job = _submit(audit_command, args.dry_run, "AUDIT_JOB")
    submissions.append({"stage": "training_audit", "job_id": audit_job, "command": audit_command})
    eval_command = [
        "sbatch",
        "--parsable",
        "--partition=a5000ada",
        "--nodelist=c32",
        f"--dependency=afterok:{audit_job}",
        f"--export={exports}",
        "scripts/slurm/24_5_eval_analyze_trace_observation.sh",
    ]
    eval_job = _submit(eval_command, args.dry_run, "EVAL_JOB")
    submissions.append({"stage": "evaluation_analysis_audit", "job_id": eval_job, "command": eval_command})
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "experiment_name": "trace_length_observation_gate0_v1",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": submissions,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    submission_dir = result_root / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    path = submission_dir / f"submission_{eval_job}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "submitted", "final_job": eval_job, "manifest": str(path)}, indent=2))


def _submit(command: List[str], dry_run: bool, dry_id: str) -> str:
    if dry_run:
        return dry_id
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch failed returncode={completed.returncode} command={command!r} "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch output: {completed.stdout!r}")
    return job_id


def _prepare_checkpoint_root(project_path: Path, beegfs_path: Path) -> None:
    beegfs_path.mkdir(parents=True, exist_ok=True)
    if project_path.is_symlink():
        if project_path.resolve() != beegfs_path:
            raise ValueError(f"Checkpoint symlink targets {project_path.resolve()}, expected {beegfs_path}")
        return
    if project_path.exists():
        if project_path.resolve() != beegfs_path:
            raise ValueError(f"Checkpoint path is not the registered BeeGFS root: {project_path}")
        return
    project_path.symlink_to(beegfs_path, target_is_directory=True)


def _require_frozen_protocol(result_root: Path) -> None:
    source_path = PROJECT_ROOT / "configs/trace_length_observation_gate0_v1.json"
    frozen_path = result_root / "formal/protocol/frozen_protocol.json"
    marker_path = frozen_path.parent / "PROTOCOL_FROZEN"
    if not frozen_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(
            "Freeze the Phase-0 protocol before submission with "
            "scripts/24_0_freeze_trace_observation_protocol.py"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    marker = read_key_value_marker(marker_path)
    expected_hash = protocol_hash(source)
    if frozen != source or protocol_hash(frozen) != expected_hash:
        raise ValueError("Frozen protocol differs from the current registered source config.")
    if marker.get("status") != "frozen" or marker.get("config_hash") != expected_hash:
        raise ValueError("Frozen protocol marker is not bound to the current protocol.")
    if marker.get("frozen_protocol_sha256") != file_sha256(frozen_path):
        raise ValueError("Frozen protocol file hash differs from its marker.")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
