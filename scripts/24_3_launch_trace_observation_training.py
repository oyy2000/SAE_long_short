#!/usr/bin/env python3
"""Launch one process per GPU for a Phase-0 controlled-SFT shard."""

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
from typing import Any, Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import (
    file_sha256,
    select_launcher_shard_runs,
    validated_adapter_evidence,
)
from trace_length_observation.trace_observation import (
    protocol_hash,
    validate_trace_observation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--launcher-shards", type=int, required=True)
    parser.add_argument("--launcher-shard-index", type=int, required=True)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    manifest_path = _resolve(args.dataset_manifest)
    work_dir = _resolve(args.work_dir)
    checkpoint_root = _resolve(args.checkpoint_root)
    runtime_root = Path(args.runtime_root).resolve()
    config = _read_json(config_path)
    validate_trace_observation_config(config)
    config_hash = protocol_hash(config)
    dataset_manifest = _read_json(manifest_path)
    _validate_dataset_manifest(dataset_manifest, manifest_path, config_hash)
    all_runs = [dict(run) for run in dataset_manifest["runs"]]
    selected_runs = select_launcher_shard_runs(
        all_runs,
        launcher_shards=args.launcher_shards,
        launcher_shard_index=args.launcher_shard_index,
    )
    if not selected_runs:
        raise ValueError("No Phase-0 runs were assigned to this launcher shard.")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU.")
    max_parallel = args.max_parallel or len(gpu_ids)
    if max_parallel <= 0 or max_parallel > len(gpu_ids):
        raise ValueError("--max-parallel must be positive and no larger than GPU count.")
    config_dir = work_dir / "configs"
    log_dir = work_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "training": file_sha256(SRC_ROOT / "trace_length_observation/controlled_sft.py"),
        "masking": file_sha256(SRC_ROOT / "trace_length_observation/trace_observation.py"),
        "launcher": file_sha256(Path(__file__).resolve()),
        "entrypoint": file_sha256(PROJECT_ROOT / "scripts/24_2_train_trace_observation_sft.py"),
    }
    entries: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for run in selected_runs:
        entry, command = _prepare_run(
            run,
            config,
            config_path,
            config_hash,
            manifest_path,
            source_hashes,
            config_dir,
            log_dir,
            checkpoint_root,
            runtime_root,
        )
        entries.append(entry)
        commands.append(command)
    launch_manifest_path = work_dir / (
        f"training_manifest_shard_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    )
    if launch_manifest_path.exists() and not args.skip_complete:
        raise FileExistsError(f"Refusing to overwrite training manifest: {launch_manifest_path}")
    launch_manifest = {
        "status": "prepared" if args.dry_run else "running",
        "stage": str(dataset_manifest["stage"]),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "run_count": len(entries),
        "source_hashes": source_hashes,
        "runs": entries,
    }
    _write_json(launch_manifest_path, launch_manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    failures = _run_commands(
        entries,
        commands,
        gpu_ids,
        max_parallel,
        launch_manifest,
        launch_manifest_path,
        skip_complete=args.skip_complete,
    )
    launch_manifest["status"] = "failed" if failures else "complete"
    _write_json(launch_manifest_path, launch_manifest)
    if failures:
        raise SystemExit(f"Controlled SFT failures={len(failures)}")
    logging.info("phase0_training_shard_complete manifest=%s", launch_manifest_path)


def _prepare_run(
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    config_hash: str,
    dataset_manifest_path: Path,
    source_hashes: Mapping[str, str],
    config_dir: Path,
    log_dir: Path,
    checkpoint_root: Path,
    runtime_root: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    run_name = str(run["run_name"])
    train_path = _resolve(str(run["train_path"]))
    if file_sha256(train_path) != str(run["train_sha256"]):
        raise ValueError(f"Training-data hash mismatch for {run_name}")
    output_dir = (checkpoint_root / run_name).resolve()
    run_config_path = config_dir / f"{run_name}.json"
    run_config = {
        "run_name": run_name,
        "protocol_path": str(config_path),
        "protocol_hash": config_hash,
        "protocol_file_sha256": file_sha256(config_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "condition": {
            key: run[key]
            for key in (
                "budget_regime",
                "loss_normalization",
                "loss_mask",
                "length_rank",
                "seed",
            )
        },
        "data": {
            "train_path": str(train_path),
            "train_sha256": str(run["train_sha256"]),
            "record_count": int(run["record_count"]),
            "unique_problem_count": int(run["unique_problem_count"]),
            "budget_token_basis": str(run["budget_token_basis"]),
            "target_budget_tokens": int(run["target_budget_tokens"]),
            "actual_budget_tokens": int(run["actual_budget_tokens"]),
            "actual_completion_tokens": int(run["actual_completion_tokens"]),
            "actual_model_input_tokens": int(run["actual_model_input_tokens"]),
        },
        "student": deepcopy(dict(config["student"])),
        "training": deepcopy(dict(config["training"])),
        "source_hashes": dict(source_hashes),
        "output_dir": str(output_dir),
    }
    if run_config_path.exists():
        existing = _read_json(run_config_path)
        if existing != run_config:
            raise ValueError(f"Existing run config differs from registered content: {run_config_path}")
    else:
        _write_json(run_config_path, run_config)
    entry = {
        **dict(run),
        "config_path": str(run_config_path),
        "run_config_sha256": file_sha256(run_config_path),
        "output_dir": str(output_dir),
        "runtime_output_dir": str(runtime_root / run_name),
        "log_path": str(log_dir / f"{run_name}.log"),
        "source_hashes": dict(source_hashes),
        "status": "prepared",
    }
    command = [
        sys.executable,
        "scripts/24_2_train_trace_observation_sft.py",
        "--run-config",
        str(run_config_path),
        "--runtime-output-dir",
        entry["runtime_output_dir"],
        "--publish-output-dir",
        str(output_dir),
    ]
    return entry, command


def _run_commands(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    manifest: Dict[str, Any],
    manifest_path: Path,
    *,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    available = list(gpu_ids)
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
    failures: List[Dict[str, Any]] = []
    while pending or running:
        while pending and available and len(running) < max_parallel:
            entry, command = pending.pop(0)
            completed = _completed_adapter(entry)
            if completed is not None:
                if not skip_complete:
                    raise FileExistsError(f"Completed adapter already exists: {entry['output_dir']}")
                entry.update(completed)
                entry["status"] = "skipped_complete"
                _write_json(manifest_path, manifest)
                continue
            gpu_id = available.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            entry["status"] = "running"
            entry["gpu_id"] = gpu_id
            entry["pid"] = process.pid
            running.append((process, entry, log_handle, gpu_id))
            _write_json(manifest_path, manifest)
        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
        for process, entry, log_handle, gpu_id in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle, gpu_id))
                continue
            log_handle.close()
            available.append(gpu_id)
            entry["returncode"] = return_code
            completed = _completed_adapter(entry) if return_code == 0 else None
            if completed is None:
                entry["status"] = "failed"
                failures.append(entry)
            else:
                entry.update(completed)
                entry["status"] = "complete"
            _write_json(manifest_path, manifest)
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _completed_adapter(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    evidence = validated_adapter_evidence(entry["output_dir"])
    if evidence is None:
        return None
    expected = {
        "run_name": str(entry["run_name"]),
        "seed": str(entry["seed"]),
        "train_sha256": str(entry["train_sha256"]),
        "run_config_sha256": str(entry["run_config_sha256"]),
        "training_source_sha256": str(entry["source_hashes"]["training"]),
        "launcher_source_sha256": str(entry["source_hashes"]["launcher"]),
        "masking_source_sha256": str(entry["source_hashes"]["masking"]),
    }
    if any(str(evidence.get(key)) != value for key, value in expected.items()):
        return None
    metrics_path = Path(str(entry["output_dir"])) / "training_metrics.json"
    if not metrics_path.is_file():
        return None
    return {
        "adapter_config_sha256": evidence["adapter_config_sha256"],
        "adapter_model_sha256": evidence["adapter_model_sha256"],
        "training_metrics_path": str(metrics_path),
        "training_metrics_sha256": file_sha256(metrics_path),
    }


def _validate_dataset_manifest(
    manifest: Mapping[str, Any], path: Path, config_hash: str
) -> None:
    if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
        raise ValueError("Dataset manifest is incomplete or bound to another protocol.")
    if int(manifest.get("run_count", -1)) != len(manifest.get("runs", [])):
        raise ValueError("Dataset manifest run count is inconsistent.")
    marker = path.parent / "DATA_COMPLETE"
    if not marker.is_file():
        raise FileNotFoundError(marker)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
