#!/usr/bin/env python3
"""Train and publish one explicitly normalized Phase-0 LoRA adapter."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trace_length_observation.controlled_sft import run_controlled_lora_sft
from length_budget_distill.factorial import file_sha256
from trace_length_observation.trace_observation import (
    protocol_hash,
    validate_trace_observation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--runtime-output-dir", required=True)
    parser.add_argument("--publish-output-dir", required=True)
    parser.add_argument("--publish-attempts", type=int, default=10)
    parser.add_argument("--publish-wait-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_config_path = _resolve(args.run_config)
    runtime_output_dir = Path(args.runtime_output_dir).resolve()
    publish_output_dir = _resolve(args.publish_output_dir)
    if runtime_output_dir.exists():
        raise FileExistsError(f"Runtime output already exists: {runtime_output_dir}")
    if publish_output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite adapter evidence: {publish_output_dir}")
    run_config = _read_json(run_config_path)
    _validate_run_config(run_config, run_config_path, publish_output_dir)
    metrics = run_controlled_lora_sft(run_config, runtime_output_dir=runtime_output_dir)
    metrics.update(
        {
            "protocol_hash": str(run_config["protocol_hash"]),
            "run_config_path": str(run_config_path),
            "run_config_sha256": file_sha256(run_config_path),
            "train_path": str(run_config["data"]["train_path"]),
            "train_sha256": str(run_config["data"]["train_sha256"]),
        }
    )
    runtime_metrics = runtime_output_dir / "training_metrics.json"
    _write_json(runtime_metrics, metrics)
    _publish(
        runtime_output_dir,
        publish_output_dir,
        attempts=args.publish_attempts,
        wait_seconds=args.publish_wait_seconds,
    )
    evidence = {
        filename: file_sha256(publish_output_dir / filename)
        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "training_metrics.json",
        )
    }
    marker = publish_output_dir / "TRAIN_COMPLETE"
    condition = dict(run_config["condition"])
    marker.write_text(
        f"run_name={run_config['run_name']}\n"
        f"seed={condition['seed']}\n"
        f"budget_regime={condition['budget_regime']}\n"
        f"loss_normalization={condition['loss_normalization']}\n"
        f"loss_mask={condition['loss_mask']}\n"
        f"length_rank={condition['length_rank']}\n"
        f"protocol_hash={run_config['protocol_hash']}\n"
        f"train_sha256={run_config['data']['train_sha256']}\n"
        f"run_config_sha256={file_sha256(run_config_path)}\n"
        f"training_source_sha256={run_config['source_hashes']['training']}\n"
        f"launcher_source_sha256={run_config['source_hashes']['launcher']}\n"
        f"masking_source_sha256={run_config['source_hashes']['masking']}\n"
        f"adapter_config_sha256={evidence['adapter_config.json']}\n"
        f"adapter_model_sha256={evidence['adapter_model.safetensors']}\n"
        f"training_metrics_sha256={evidence['training_metrics.json']}\n",
        encoding="utf-8",
    )
    logging.info("controlled_adapter_complete run=%s output=%s", run_config["run_name"], publish_output_dir)


def _validate_run_config(
    run_config: Mapping[str, Any], run_config_path: Path, publish_output_dir: Path
) -> None:
    protocol_path = _resolve(str(run_config["protocol_path"]))
    protocol = _read_json(protocol_path)
    validate_trace_observation_config(protocol)
    if protocol_hash(protocol) != str(run_config["protocol_hash"]):
        raise ValueError("Run config protocol hash mismatch.")
    if file_sha256(protocol_path) != str(run_config["protocol_file_sha256"]):
        raise ValueError("Run config protocol file hash mismatch.")
    train_path = _resolve(str(run_config["data"]["train_path"]))
    if file_sha256(train_path) != str(run_config["data"]["train_sha256"]):
        raise ValueError("Run config training-data hash mismatch.")
    if Path(str(run_config["output_dir"])).resolve() != publish_output_dir.resolve():
        raise ValueError("Requested publish path differs from the run config.")
    if int(run_config["condition"]["seed"]) not in (17, 42, 73):
        raise ValueError("Unexpected training seed.")
    source_hashes = dict(run_config["source_hashes"])
    expected_training = file_sha256(
        SRC_ROOT / "trace_length_observation/controlled_sft.py"
    )
    expected_launcher = file_sha256(PROJECT_ROOT / "scripts/24_3_launch_trace_observation_training.py")
    expected_masking = file_sha256(SRC_ROOT / "trace_length_observation/trace_observation.py")
    expected_entrypoint = file_sha256(Path(__file__).resolve())
    if source_hashes.get("training") != expected_training:
        raise ValueError("Controlled training source hash mismatch.")
    if source_hashes.get("launcher") != expected_launcher:
        raise ValueError("Training launcher source hash mismatch.")
    if source_hashes.get("masking") != expected_masking:
        raise ValueError("Training masking source hash mismatch.")
    if source_hashes.get("entrypoint") != expected_entrypoint:
        raise ValueError("Training entrypoint source hash mismatch.")


def _publish(source: Path, destination: Path, *, attempts: int, wait_seconds: float) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("adapter_config.json", "adapter_model.safetensors", "training_metrics.json"):
        source_path = source / filename
        expected = file_sha256(source_path)
        target = destination / filename
        partial = destination / f".{filename}.partial-{os.getpid()}"
        for attempt in range(1, attempts + 1):
            try:
                shutil.copyfile(source_path, partial)
                if file_sha256(partial) != expected:
                    raise OSError(f"Post-copy hash mismatch: {partial}")
                os.replace(partial, target)
                if file_sha256(target) != expected:
                    raise OSError(f"Published hash mismatch: {target}")
                break
            except OSError:
                if partial.exists():
                    partial.unlink()
                if attempt == attempts:
                    raise
                logging.warning("publish retry %d/%d file=%s", attempt, attempts, filename)
                time.sleep(wait_seconds)


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
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
