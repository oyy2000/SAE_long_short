#!/usr/bin/env python3
"""Launch a disjoint shard of the intervention SFT matrix across local GPUs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, validated_training_artifacts
from length_budget_distill.factorial import canonical_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--launcher-shards", type=int, required=True)
    parser.add_argument("--launcher-shard-index", type=int, required=True)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = _resolve(args.config)
    config = read_json(config_path)
    config_hash = canonical_sha256(config)
    manifest_path = _resolve(args.dataset_manifest)
    data_manifest = read_json(manifest_path)
    if data_manifest.get("status") != "complete" or data_manifest.get("config_hash") != config_hash:
        raise ValueError("SFT dataset manifest is incomplete or belongs to another protocol.")
    cells = sorted(
        data_manifest["datasets"],
        key=lambda row: (row["budget_regime"], row["condition"]),
    )
    runs = []
    for cell in cells:
        for seed in config["student_sft"]["seeds"]:
            runs.append({**cell, "seed": int(seed)})
    selected = [
        row for index, row in enumerate(runs)
        if index % args.launcher_shards == args.launcher_shard_index
    ]
    if not selected:
        raise ValueError("Launcher shard contains no runs.")
    gpu_ids = [part for part in args.gpu_ids.split(",") if part]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")
    work_dir = _resolve(args.work_dir)
    checkpoint_root = _resolve(args.checkpoint_root)
    runtime_root = Path(args.runtime_root).resolve()
    config_dir, log_dir = work_dir / "run_configs", work_dir / "logs"
    for directory in (config_dir, log_dir, checkpoint_root, runtime_root):
        directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "training": file_sha256(PROJECT_ROOT / "src/trace_length_observation/controlled_sft.py"),
        "launcher": file_sha256(Path(__file__).resolve()),
        "entrypoint": file_sha256(PROJECT_ROOT / "scripts/4_1_train_intervention_sft.py"),
    }
    entries = []
    for cell in selected:
        run_name = f"{cell['budget_regime']}__{cell['condition']}__seed_{cell['seed']}"
        train_path = _resolve(cell["path"])
        if file_sha256(train_path) != cell["sha256"]:
            raise ValueError(f"Dataset hash mismatch for {run_name}")
        output_dir = checkpoint_root / run_name
        run_config_path = config_dir / f"{run_name}.json"
        run_config = {
            "run_name": run_name,
            "frozen_protocol_path": str(config_path),
            "config_hash": config_hash,
            "sft_data_manifest_path": str(manifest_path),
            "sft_data_manifest_sha256": file_sha256(manifest_path),
            "condition": {
                "seed": cell["seed"],
                "budget_regime": cell["budget_regime"],
                "intervention_condition": cell["condition"],
                "loss_normalization": config["student_sft"]["loss_normalization"],
                "loss_mask": config["student_sft"]["loss_mask"],
            },
            "data": {
                "train_path": str(train_path),
                "train_sha256": cell["sha256"],
                "record_count": cell["record_count"],
                "unique_problem_count": cell["unique_problem_count"],
                "completion_token_count": cell["completion_token_count"],
            },
            "student": deepcopy(config["student"]),
            "training": deepcopy(config["student_sft"]),
            "source_hashes": sources,
            "output_dir": str(output_dir.resolve()),
        }
        run_config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
        if run_config_path.exists():
            if run_config_path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"Existing run config differs: {run_config_path}")
        else:
            run_config_path.write_text(payload, encoding="utf-8")
        entries.append(
            {
                "run_name": run_name,
                "run_config_path": str(run_config_path),
                "run_config_sha256": file_sha256(run_config_path),
                "output_dir": str(output_dir.resolve()),
                "runtime_output_dir": str(runtime_root / run_name),
                "log_path": str(log_dir / f"{run_name}.log"),
                "seed": cell["seed"],
                "budget_regime": cell["budget_regime"],
                "condition": cell["condition"],
                "train_sha256": cell["sha256"],
                "status": "prepared",
            }
        )
    launch_manifest_path = work_dir / f"training_launcher_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    launch_manifest = {
        "status": "running",
        "config_hash": config_hash,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "source_hashes": sources,
        "runs": entries,
    }
    _write_replace(launch_manifest_path, launch_manifest)
    failures = _run(entries, gpu_ids, launch_manifest, launch_manifest_path, args.skip_complete)
    launch_manifest["status"] = "failed" if failures else "complete"
    _write_replace(launch_manifest_path, launch_manifest)
    if failures:
        raise SystemExit(f"SFT failures={failures}")


def _run(entries: list[dict[str, Any]], gpu_ids: list[str], manifest: dict[str, Any], manifest_path: Path, skip_complete: bool) -> int:
    pending = list(entries)
    available = list(gpu_ids)
    running: list[tuple[subprocess.Popen[Any], dict[str, Any], Any, str]] = []
    failures = 0
    while pending or running:
        while pending and available:
            entry = pending.pop(0)
            try:
                evidence = validated_training_artifacts(entry["output_dir"])
            except (FileNotFoundError, ValueError):
                evidence = None
            if evidence is not None:
                if not skip_complete:
                    raise FileExistsError(f"Completed run exists: {entry['output_dir']}")
                entry["status"] = "skipped_complete"
                _write_replace(manifest_path, manifest)
                continue
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            command = [
                sys.executable,
                "scripts/4_1_train_intervention_sft.py",
                "--run-config", entry["run_config_path"],
                "--runtime-output-dir", entry["runtime_output_dir"],
                "--publish-output-dir", entry["output_dir"],
            ]
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            entry.update({"status": "running", "gpu_id": gpu_id, "pid": process.pid})
            running.append((process, entry, log_handle, gpu_id))
            _write_replace(manifest_path, manifest)
        next_running = []
        for process, entry, handle, gpu_id in running:
            code = process.poll()
            if code is None:
                next_running.append((process, entry, handle, gpu_id))
                continue
            handle.close()
            available.append(gpu_id)
            entry["returncode"] = code
            entry["status"] = "complete" if code == 0 else "failed"
            failures += int(code != 0)
            _write_replace(manifest_path, manifest)
        running = next_running
        if running:
            time.sleep(5)
    return failures


def _write_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
