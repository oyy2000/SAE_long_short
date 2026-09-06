#!/usr/bin/env python3
"""Launch disjoint activation-extraction shards across available GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    shard_count = int(config["activation_extraction"]["trajectory_shards"])
    runs = deque(
        shard
        for shard in range(shard_count)
        if not (args.skip_complete and _complete_marker(shard, shard_count).is_file())
    )
    active = {}
    completed = []
    while runs or active:
        for gpu_id in gpu_ids:
            if gpu_id in active or not runs:
                continue
            shard = runs.popleft()
            output_dir = (
                PROJECT_ROOT
                / f"results/phase2_sae_pilot_v1/formal/activations/shard_{shard:02d}_of_{shard_count:02d}"
            )
            log_path = (
                PROJECT_ROOT
                / f"logs/phase2_extract_shard_{shard:02d}-{os.environ.get('SLURM_JOB_ID', 'local')}.log"
            )
            log_handle = log_path.open("x", encoding="utf-8")
            command = [
                sys.executable,
                "scripts/2_2_extract_residual_activations.py",
                "--config",
                str(config_path),
                "--shard-index",
                str(shard),
                "--shard-count",
                str(shard_count),
                "--output-dir",
                str(output_dir),
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu_id] = (process, log_handle, shard, command)
            print(f"launched shard={shard} gpu={gpu_id} pid={process.pid}")
        time.sleep(2)
        for gpu_id, payload in list(active.items()):
            process, log_handle, shard, command = payload
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[gpu_id]
            if return_code != 0:
                for other, other_payload in active.items():
                    other_payload[0].terminate()
                    other_payload[1].close()
                    print(f"terminated sibling process on gpu={other}", file=sys.stderr)
                raise RuntimeError(
                    f"Activation extraction failed shard={shard} command={command!r}"
                )
            completed.append({"shard_index": shard, "gpu_id": gpu_id})
            print(f"completed shard={shard} gpu={gpu_id}")
    print(json.dumps({"status": "complete", "runs": completed}, indent=2))


def _complete_marker(shard: int, shard_count: int) -> Path:
    return (
        PROJECT_ROOT
        / f"results/phase2_sae_pilot_v1/formal/activations/shard_{shard:02d}_of_{shard_count:02d}/ACTIVATIONS_COMPLETE"
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
