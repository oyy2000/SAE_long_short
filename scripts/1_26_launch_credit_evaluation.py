#!/usr/bin/env python3
"""Launch the Phase-1.5 pilot or confirmation evaluation matrix across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from length_budget_distill.experiment_io import read_json, require_passed_gate
from length_budget_distill.factorial import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase1_5_credit_allocation_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument(
        "--utility-config",
        default="results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument("--stage", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument(
        "--checkpoint-root", default="checkpoints/phase1_5_credit_allocation_v1"
    )
    parser.add_argument(
        "--output-root",
        default="results/phase1_5_credit_allocation_v1/formal/evaluation",
    )
    args = parser.parse_args()
    config_path, utility_path = _resolve(args.config), _resolve(args.utility_config)
    config = read_json(config_path)
    require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    checkpoint_root, output_root = (
        _resolve(args.checkpoint_root),
        _resolve(args.output_root),
    )
    if args.stage == "pilot":
        split = "dev"
        pairs = [
            (condition, int(config["pilot"]["seed"]), "pilot")
            for condition in config["pilot"]["conditions"]
        ]
    else:
        require_passed_gate(PROJECT_ROOT, config["pilot_gate_dependency"])
        split = "test"
        pilot_seed = int(config["pilot"]["seed"])
        pairs = [
            (condition, int(seed), "pilot" if int(seed) == pilot_seed else "confirm")
            for condition in config["confirmation"]["conditions"]
            for seed in config["confirmation"]["seeds"]
        ]
    tasks = [
        {
            "condition": condition,
            "seed": seed,
            "split": split,
            "adapter": checkpoint_root / source / condition / f"seed_{seed}",
            "output": output_root / args.stage / condition / f"seed_{seed}",
            "status": "prepared",
        }
        for condition, seed, source in pairs
    ]
    failures = _run(
        tasks,
        [value for value in args.gpu_ids.split(",") if value],
        config_path,
        utility_path,
    )
    manifest = {
        "status": "failed" if failures else "complete",
        "stage": args.stage,
        "config_sha256": file_sha256(config_path),
        "task_count": len(tasks),
        "tasks": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in row.items()
            }
            for row in tasks
        ],
        "failures": failures,
        "source_sha256": file_sha256(Path(__file__).resolve()),
    }
    path = (
        output_root
        / f"{args.stage}_evaluation_manifest_{os.environ.get('SLURM_JOB_ID', os.getpid())}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    if failures:
        raise SystemExit(f"Credit evaluation failures: {failures}")
    (output_root / f"{args.stage.upper()}_EVALUATION_COMPLETE").write_text(
        f"status=complete\nmanifest={path}\nmanifest_sha256={file_sha256(path)}\ntask_count={len(tasks)}\n",
        encoding="utf-8",
    )


def _run(tasks, gpus, config, utility):
    if not gpus:
        raise ValueError("No GPU IDs supplied.")
    pending, available, running, failures = list(tasks), list(gpus), [], []
    while pending or running:
        while pending and available:
            task, gpu = pending.pop(0), available.pop(0)
            if (task["output"] / "EVALUATION_COMPLETE").is_file():
                task["status"] = "skipped_complete"
                continue
            if not (task["adapter"] / "TRAIN_COMPLETE").is_file():
                raise FileNotFoundError(task["adapter"] / "TRAIN_COMPLETE")
            task["output"].parent.mkdir(parents=True, exist_ok=True)
            log_path = task["output"].parent / f"{task['output'].name}.log"
            log = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                "scripts/1_18_eval_credit_allocation.py",
                "--config",
                str(config),
                "--utility-config",
                str(utility),
                "--adapter-path",
                str(task["adapter"]),
                "--condition",
                task["condition"],
                "--seed",
                str(task["seed"]),
                "--split",
                task["split"],
                "--output-dir",
                str(task["output"]),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            task.update(
                {
                    "status": "running",
                    "gpu_id": gpu,
                    "pid": process.pid,
                    "log_path": log_path,
                }
            )
            running.append((process, task, log, gpu))
        next_running = []
        for process, task, log, gpu in running:
            code = process.poll()
            if code is None:
                next_running.append((process, task, log, gpu))
                continue
            log.close()
            available.append(gpu)
            task["returncode"] = code
            if code == 0 and (task["output"] / "EVALUATION_COMPLETE").is_file():
                task["status"] = "complete"
            else:
                task["status"] = "failed"
                failures.append(f"{task['condition']}:{task['seed']}")
        running = next_running
        if running:
            time.sleep(5)
    return failures


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
