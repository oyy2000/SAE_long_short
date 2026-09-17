#!/usr/bin/env python3
"""Train and hash-publish one LoRA student on an intervention dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    publish_files_hash_verified,
    read_json,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker
from trace_length_observation.controlled_sft import run_controlled_lora_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--runtime-output-dir", required=True)
    parser.add_argument("--publish-output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_config_path = _resolve(args.run_config)
    run_config = read_json(run_config_path)
    runtime_dir = Path(args.runtime_output_dir).resolve()
    publish_dir = _resolve(args.publish_output_dir)
    if runtime_dir.exists():
        raise FileExistsError(f"Runtime output exists: {runtime_dir}")
    if publish_dir.exists():
        raise FileExistsError(f"Published output exists: {publish_dir}")
    _validate(run_config, run_config_path, publish_dir)

    metrics = run_controlled_lora_sft(run_config, runtime_output_dir=runtime_dir)
    metrics.update(
        {
            "config_hash": run_config["config_hash"],
            "frozen_protocol_path": run_config["frozen_protocol_path"],
            "run_config_path": str(run_config_path),
            "run_config_sha256": file_sha256(run_config_path),
            "train_path": run_config["data"]["train_path"],
            "train_sha256": run_config["data"]["train_sha256"],
            "budget_regime": run_config["condition"]["budget_regime"],
            "intervention_condition": run_config["condition"]["intervention_condition"],
            "formal_claim_allowed": False,
        }
    )
    write_json_exclusive(runtime_dir / "training_metrics.json", metrics)
    hashes = publish_files_hash_verified(
        runtime_dir,
        publish_dir,
        ("adapter_config.json", "adapter_model.safetensors", "training_metrics.json"),
    )
    condition = run_config["condition"]
    marker = publish_dir / "TRAIN_COMPLETE"
    marker.write_text(
        "status=complete\n"
        f"run_name={run_config['run_name']}\n"
        f"seed={condition['seed']}\n"
        f"budget_regime={condition['budget_regime']}\n"
        f"intervention_condition={condition['intervention_condition']}\n"
        f"config_hash={run_config['config_hash']}\n"
        f"train_sha256={run_config['data']['train_sha256']}\n"
        f"run_config_sha256={file_sha256(run_config_path)}\n"
        f"training_source_sha256={run_config['source_hashes']['training']}\n"
        f"launcher_source_sha256={run_config['source_hashes']['launcher']}\n"
        f"entrypoint_source_sha256={run_config['source_hashes']['entrypoint']}\n"
        f"adapter_config_sha256={hashes['adapter_config.json']}\n"
        f"adapter_model_sha256={hashes['adapter_model.safetensors']}\n"
        f"training_metrics_sha256={hashes['training_metrics.json']}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    logging.info("intervention_sft_complete run=%s output=%s", run_config["run_name"], publish_dir)


def _validate(run: Mapping[str, Any], path: Path, publish_dir: Path) -> None:
    protocol_path = _resolve(run["frozen_protocol_path"])
    protocol = read_json(protocol_path)
    if canonical_sha256(protocol) != run["config_hash"]:
        raise ValueError("Frozen protocol canonical hash mismatch.")
    protocol_marker = protocol_path.parent / "PROTOCOL_FROZEN"
    marker = read_key_value_marker(protocol_marker)
    if marker.get("status") != "frozen" or marker.get("config_sha256") != file_sha256(protocol_path):
        raise ValueError("Frozen protocol marker is invalid.")
    manifest_path = _resolve(run["sft_data_manifest_path"])
    manifest = read_json(manifest_path)
    data_marker = read_key_value_marker(manifest_path.parent / "SFT_DATA_COMPLETE")
    if data_marker.get("status") != "complete" or data_marker.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("SFT data completion marker is invalid.")
    if manifest.get("config_hash") != run["config_hash"]:
        raise ValueError("SFT data protocol hash mismatch.")
    train_path = _resolve(run["data"]["train_path"])
    if file_sha256(train_path) != run["data"]["train_sha256"]:
        raise ValueError("Training dataset hash mismatch.")
    if publish_dir.resolve() != Path(run["output_dir"]).resolve():
        raise ValueError("Publish output differs from registered run config.")
    if int(run["condition"]["seed"]) not in (17, 42, 73):
        raise ValueError("Unexpected training seed.")
    expected_sources = {
        "training": file_sha256(PROJECT_ROOT / "src/trace_length_observation/controlled_sft.py"),
        "launcher": file_sha256(PROJECT_ROOT / "scripts/4_2_launch_intervention_sft_matrix.py"),
        "entrypoint": file_sha256(Path(__file__).resolve()),
    }
    if dict(run["source_hashes"]) != expected_sources:
        raise ValueError("Training source hashes differ from the registered run config.")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
