#!/usr/bin/env python3
"""Launch a registered Phase-1 policy-SFT screen or confirmation matrix."""

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
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument(
        "--checkpoint-root", default="checkpoints/phase1_teaching_utility_v1"
    )
    parser.add_argument(
        "--policy-data-root",
        default="results/phase1_teaching_utility_v1/formal/policy_data",
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
    if not gpu_ids:
        raise ValueError("No GPU IDs were supplied.")
    checkpoint_root = _resolve(args.checkpoint_root)
    runtime_root = Path(args.runtime_root).resolve()
    policy_root = _resolve(args.policy_data_root)
    tasks = []
    if args.stage == "screen":
        seed = int(config["policy_sft"]["screen_seed"])
        for policy in config["policy_sft"]["screen_policies"]:
            tasks.append(
                _task(
                    policy,
                    seed,
                    policy_root / f"{policy}.jsonl",
                    checkpoint_root / "screen" / policy / f"seed_{seed}",
                    runtime_root,
                )
            )
        mid_data = (
            PROJECT_ROOT
            / "results/phase1_teaching_utility_v1/formal/mid_sft_anchor_data/random_correct_first_half.jsonl"
        )
        tasks.append(
            _task(
                "mid_sft_anchor",
                seed,
                mid_data,
                checkpoint_root / "anchors/mid_sft_random_50pct",
                runtime_root,
            )
        )
    else:
        plan = read_json(_resolve(args.confirmation_plan))
        policies = sorted(set(plan["confirmation_label_to_policy"].values()))
        screen_seed = int(config["policy_sft"]["screen_seed"])
        for policy in policies:
            for seed in config["policy_sft"]["confirm_seeds"]:
                if int(seed) == screen_seed:
                    continue
                tasks.append(
                    _task(
                        policy,
                        int(seed),
                        policy_root / f"{policy}.jsonl",
                        checkpoint_root / "confirm" / policy / f"seed_{seed}",
                        runtime_root,
                    )
                )
    for task in tasks:
        if not task["data_path"].is_file():
            raise FileNotFoundError(task["data_path"])
    failures = _run(tasks, gpu_ids, config_path)
    manifest = {
        "status": "failed" if failures else "complete",
        "stage": args.stage,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "task_count": len(tasks),
        "tasks": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in task.items()
                if key != "command"
            }
            for task in tasks
        ],
        "failures": failures,
        "source_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest_path = checkpoint_root / f"{args.stage}_training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    if failures:
        raise SystemExit(f"Policy training failures: {failures}")
    (manifest_path.parent / f"{args.stage.upper()}_TRAINING_COMPLETE").write_text(
        f"status=complete\nmanifest_sha256={file_sha256(manifest_path)}\ntask_count={len(tasks)}\n",
        encoding="utf-8",
    )


def _task(policy, seed, data_path, publish_dir, runtime_root):
    run = f"{policy}__seed_{seed}"
    return {
        "policy": policy,
        "seed": seed,
        "data_path": data_path,
        "publish_dir": publish_dir,
        "runtime_dir": runtime_root / run,
        "log_path": runtime_root / "logs" / f"{run}.log",
        "status": "prepared",
    }


def _run(tasks, gpu_ids, config_path):
    pending, available, running, failures = list(tasks), list(gpu_ids), [], []
    while pending or running:
        while pending and available:
            task, gpu = pending.pop(0), available.pop(0)
            marker = task["publish_dir"] / "TRAIN_COMPLETE"
            if marker.is_file():
                task["status"] = "skipped_complete"
                continue
            task["log_path"].parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "scripts/1_10_train_utility_policy_sft.py",
                "--config",
                str(config_path),
                "--policy-data",
                str(task["data_path"]),
                "--policy",
                str(task["policy"]),
                "--seed",
                str(task["seed"]),
                "--runtime-output-dir",
                str(task["runtime_dir"]),
                "--publish-output-dir",
                str(task["publish_dir"]),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log = task["log_path"].open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            task.update({"status": "running", "gpu_id": gpu, "pid": process.pid})
            running.append((process, task, log, gpu))
        next_running = []
        for process, task, log, gpu in running:
            returncode = process.poll()
            if returncode is None:
                next_running.append((process, task, log, gpu))
                continue
            log.close()
            available.append(gpu)
            task["returncode"] = returncode
            if returncode == 0 and (task["publish_dir"] / "TRAIN_COMPLETE").is_file():
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
