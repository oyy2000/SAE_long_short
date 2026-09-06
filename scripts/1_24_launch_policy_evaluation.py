#!/usr/bin/env python3
"""Launch registered Phase-1 screen or confirmation evaluations across GPUs."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument(
        "--checkpoint-root", default="checkpoints/phase1_teaching_utility_v1"
    )
    parser.add_argument(
        "--output-root",
        default="results/phase1_teaching_utility_v1/formal/policy_evaluation",
    )
    parser.add_argument(
        "--confirmation-plan",
        default="results/phase1_teaching_utility_v1/formal/confirmation_plan/confirmation_plan.json",
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    gpu_ids = [value for value in args.gpu_ids.split(",") if value]
    checkpoint_root, output_root = (
        _resolve(args.checkpoint_root),
        _resolve(args.output_root),
    )
    tasks = []
    if args.stage == "screen":
        seed = int(config["policy_sft"]["screen_seed"])
        for policy in config["policy_sft"]["screen_policies"]:
            tasks.append(
                _task(
                    policy,
                    seed,
                    "dev",
                    checkpoint_root / "screen" / policy / f"seed_{seed}",
                    output_root / "screen" / policy,
                )
            )
    else:
        plan = read_json(_resolve(args.confirmation_plan))
        policies = sorted(set(plan["confirmation_label_to_policy"].values()))
        screen_seed = int(config["policy_sft"]["screen_seed"])
        for policy in policies:
            for seed in config["policy_sft"]["confirm_seeds"]:
                base = "screen" if int(seed) == screen_seed else "confirm"
                adapter = checkpoint_root / base / policy / f"seed_{seed}"
                tasks.append(
                    _task(
                        policy,
                        int(seed),
                        "test",
                        adapter,
                        output_root / "confirm" / policy / f"seed_{seed}",
                    )
                )
    failures = _run(tasks, gpu_ids, config_path)
    manifest = {
        "status": "failed" if failures else "complete",
        "stage": args.stage,
        "config_sha256": file_sha256(config_path),
        "task_count": len(tasks),
        "tasks": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in task.items()
            }
            for task in tasks
        ],
        "failures": failures,
        "source_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest_path = output_root / f"{args.stage}_evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    if failures:
        raise SystemExit(f"Evaluation failures: {failures}")
    (manifest_path.parent / f"{args.stage.upper()}_EVALUATION_COMPLETE").write_text(
        f"status=complete\nmanifest_sha256={file_sha256(manifest_path)}\ntask_count={len(tasks)}\n",
        encoding="utf-8",
    )


def _task(policy, seed, split, adapter, output):
    return {
        "policy": policy,
        "seed": seed,
        "split": split,
        "adapter": adapter,
        "output": output,
        "status": "prepared",
    }


def _run(tasks, gpu_ids, config_path):
    if not gpu_ids:
        raise ValueError("No GPU IDs were supplied.")
    pending, available, running, failures = list(tasks), list(gpu_ids), [], []
    while pending or running:
        while pending and available:
            task, gpu = pending.pop(0), available.pop(0)
            if (task["output"] / "EVALUATION_COMPLETE").is_file():
                task["status"] = "skipped_complete"
                continue
            if not (task["adapter"] / "TRAIN_COMPLETE").is_file():
                raise FileNotFoundError(task["adapter"] / "TRAIN_COMPLETE")
            command = [
                sys.executable,
                "scripts/1_12_eval_utility_policy.py",
                "--config",
                str(config_path),
                "--adapter-path",
                str(task["adapter"]),
                "--policy",
                task["policy"],
                "--seed",
                str(task["seed"]),
                "--split",
                task["split"],
                "--output-dir",
                str(task["output"]),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            task["output"].parent.mkdir(parents=True, exist_ok=True)
            log_path = task["output"].parent / f"{task['output'].name}.log"
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
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
                failures.append(f"{task['policy']}:{task['seed']}")
        running = next_running
        if running:
            time.sleep(5)
    return failures


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
