#!/usr/bin/env python3
"""Freeze and submit the exploratory SAE pilot as a resumable Slurm DAG."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_sae_pilot_v1.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    if not args.dry_run:
        _prepare_storage()
    frozen = (
        PROJECT_ROOT
        / "results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json"
    )
    if not args.dry_run and not frozen.is_file():
        subprocess.run(
            [
                sys.executable,
                "scripts/2_0_freeze_sae_pilot_protocol.py",
                "--config",
                str(config_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    config_arg = str(frozen if not args.dry_run else config_path)
    exports = f"ALL,PHASE2_CONFIG={config_arg}"
    jobs = []
    corpus = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--export={exports}",
            "scripts/slurm/2_1_build_sae_corpus.sh",
        ],
        args.dry_run,
        "CORPUS",
    )
    jobs.append({"stage": "corpus", "job_id": corpus})
    extraction = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c32",
            f"--dependency=afterok:{corpus}",
            f"--export={exports}",
            "scripts/slurm/2_2_extract_sae_activations.sh",
        ],
        args.dry_run,
        "EXTRACTION",
    )
    jobs.append({"stage": "activation_extraction", "job_id": extraction})
    sampling = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--dependency=afterok:{extraction}",
            f"--export={exports}",
            "scripts/slurm/2_3_sample_sae_tokens.sh",
        ],
        args.dry_run,
        "SAMPLING",
    )
    jobs.append({"stage": "token_sampling", "job_id": sampling})
    training = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a5000ada",
            "--nodelist=c32",
            f"--dependency=afterok:{sampling}",
            f"--export={exports}",
            "scripts/slurm/2_4_train_sae_matrix.sh",
        ],
        args.dry_run,
        "TRAINING",
    )
    jobs.append({"stage": "sae_training", "job_id": training})
    audit = _submit(
        [
            "sbatch",
            "--parsable",
            "--partition=a6000",
            "--nodelist=c31",
            f"--dependency=afterok:{training}",
            f"--export={exports}",
            "scripts/slurm/2_5_audit_sae_pilot.sh",
        ],
        args.dry_run,
        "AUDIT",
    )
    jobs.append({"stage": "audit", "job_id": audit})
    payload = {
        "status": "dry_run" if args.dry_run else "submitted",
        "experiment_name": config["experiment_name"],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_arg,
        "config_sha256": file_sha256(frozen if not args.dry_run else config_path),
        "source_config_path": str(config_path),
        "source_config_sha256": file_sha256(config_path),
        "jobs": jobs,
        "terminal_job_id": audit,
        "formal_claim_allowed": False,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    output = PROJECT_ROOT / "results/phase2_sae_pilot_v1/submissions"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"submission_{audit}.json"
    write_json_exclusive(path, payload)
    print(
        json.dumps({"status": "submitted", "manifest": str(path), **payload}, indent=2)
    )


def _prepare_storage() -> None:
    targets = {
        PROJECT_ROOT / "results/phase2_sae_pilot_v1/formal": Path(
            "/mnt/beegfs/youyang7/projects/SAE_long_short/phase2_sae_pilot_v1/results/formal"
        ),
        PROJECT_ROOT / "checkpoints/phase2_sae_pilot_v1": Path(
            "/mnt/beegfs/youyang7/projects/SAE_long_short/phase2_sae_pilot_v1/checkpoints"
        ),
    }
    for link, target in targets.items():
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Storage link mismatch: {link} -> {link.resolve()}")
        elif link.exists():
            raise FileExistsError(f"Expected a storage symlink, found: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)


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


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
