#!/usr/bin/env python3
"""Launch disjoint short-versus-long SAE scoring runs across selected GPUs."""

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
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--matrix-shard-index", type=int, required=True)
    parser.add_argument("--matrix-shard-count", type=int, required=True)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.matrix_shard_index < args.matrix_shard_count:
        raise ValueError("Invalid matrix-shard topology.")
    config_path = _resolve(args.config)
    config = read_json(config_path)
    parent_config = read_json(_resolve(config["parent_sae"]["config_path"]))
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU is required.")
    matrix = [
        (int(layer), int(k))
        for layer in parent_config["activation_extraction"]["layer_indices_zero_based"]
        for k in parent_config["sae"]["k_values"]
    ]
    result_root = _resolve(config["outputs"]["result_root"]) / "feature_scores"
    pending = deque()
    for index, (layer, k) in enumerate(matrix):
        if index % args.matrix_shard_count != args.matrix_shard_index:
            continue
        output_dir = result_root / f"layer_{layer:02d}_k_{k:03d}"
        if args.skip_complete and (output_dir / "FEATURE_SCORING_COMPLETE").is_file():
            continue
        pending.append((layer, k, output_dir))
    active = {}
    completed = []
    while pending or active:
        for gpu_id in gpu_ids:
            if gpu_id in active or not pending:
                continue
            layer, k, output_dir = pending.popleft()
            log_path = (
                PROJECT_ROOT
                / "logs"
                / f"phase2_feature_score_layer_{layer:02d}_k_{k:03d}-{os.environ.get('SLURM_JOB_ID', 'local')}.log"
            )
            log_handle = log_path.open("x", encoding="utf-8")
            command = [
                sys.executable,
                "scripts/2_8_score_short_long_features.py",
                "--config",
                str(config_path),
                "--layer-index",
                str(layer),
                "--k",
                str(k),
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
                for sibling in active.values():
                    sibling[0].terminate()
                    sibling[1].close()
                raise RuntimeError(
                    f"Feature scoring failed layer={layer} k={k}: {command!r}"
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


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
