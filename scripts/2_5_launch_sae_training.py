#!/usr/bin/env python3
"""Launch the registered layer-by-k SAE matrix across disjoint GPUs."""

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
    parser.add_argument("--matrix-shard-index", type=int, default=0)
    parser.add_argument("--matrix-shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.matrix_shard_index < args.matrix_shard_count:
        raise ValueError("Invalid SAE matrix shard topology.")
    config_path = _resolve(args.config)
    config = read_json(config_path)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    full_matrix = [
        (int(layer), int(k))
        for layer in config["activation_extraction"]["layer_indices_zero_based"]
        for k in config["sae"]["k_values"]
    ]
    runs = deque(
        (layer, k)
        for index, (layer, k) in enumerate(full_matrix)
        if index % args.matrix_shard_count == args.matrix_shard_index
        and not (args.skip_complete and _complete_marker(layer, k).is_file())
    )
    active = {}
    completed = []
    while runs or active:
        for gpu_id in gpu_ids:
            if gpu_id in active or not runs:
                continue
            layer, k = runs.popleft()
            output_dir = (
                PROJECT_ROOT
                / f"results/phase2_sae_pilot_v1/formal/sae_training/layer_{layer:02d}_k_{k:03d}"
            )
            checkpoint_dir = (
                PROJECT_ROOT
                / f"checkpoints/phase2_sae_pilot_v1/layer_{layer:02d}_k_{k:03d}"
            )
            log_path = (
                PROJECT_ROOT
                / f"logs/phase2_sae_train_layer_{layer:02d}_k_{k:03d}-{os.environ.get('SLURM_JOB_ID', 'local')}.log"
            )
            log_handle = log_path.open("x", encoding="utf-8")
            command = [
                sys.executable,
                "scripts/2_4_train_topk_sae.py",
                "--config",
                str(config_path),
                "--layer-index",
                str(layer),
                "--k",
                str(k),
                "--output-dir",
                str(output_dir),
                "--checkpoint-dir",
                str(checkpoint_dir),
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
            active[gpu_id] = (process, log_handle, layer, k, command)
            print(f"launched layer={layer} k={k} gpu={gpu_id} pid={process.pid}")
        time.sleep(2)
        for gpu_id, payload in list(active.items()):
            process, log_handle, layer, k, command = payload
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
                    f"SAE training failed layer={layer} k={k} command={command!r}"
                )
            completed.append({"layer_index": layer, "k": k, "gpu_id": gpu_id})
            print(f"completed layer={layer} k={k} gpu={gpu_id}")
    print(
        json.dumps(
            {
                "status": "complete",
                "matrix_shard_index": args.matrix_shard_index,
                "matrix_shard_count": args.matrix_shard_count,
                "runs": completed,
            },
            indent=2,
        )
    )


def _complete_marker(layer: int, k: int) -> Path:
    return (
        PROJECT_ROOT
        / f"results/phase2_sae_pilot_v1/formal/sae_training/layer_{layer:02d}_k_{k:03d}/SAE_TRAINING_COMPLETE"
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
