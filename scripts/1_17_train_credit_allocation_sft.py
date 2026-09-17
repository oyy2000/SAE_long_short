#!/usr/bin/env python3
"""Train one registered fixed-context credit-allocation LoRA condition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    publish_files_hash_verified,
    read_json,
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
)
from length_budget_distill.weighted_sft import run_weighted_lora_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase1_5_credit_allocation_v1.json"
    )
    parser.add_argument(
        "--utility-config", default="configs/phase1_teaching_utility_v1.json"
    )
    parser.add_argument(
        "--data-manifest",
        default="results/phase1_5_credit_allocation_v1/formal/data/credit_allocation_data_manifest.json",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-output-dir", required=True)
    parser.add_argument("--publish-output-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    utility_config_path = _resolve(args.utility_config)
    runtime_dir = Path(args.runtime_output_dir).resolve()
    publish_dir = _resolve(args.publish_output_dir)
    if runtime_dir.exists() or publish_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {runtime_dir} or {publish_dir}")
    config = read_json(config_path)
    utility_config = read_json(utility_config_path)
    gate1 = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    approval_path = _resolve(config["segmentation"]["approval_marker_path"])
    approval = None
    if bool(config["segmentation"]["manual_audit_required_before_training"]):
        if not approval_path.is_file():
            raise RuntimeError(
                "The registered manual step-segmentation audit is not approved."
            )
        approval = read_key_value_marker(approval_path)
        if approval.get("status") != "approved":
            raise RuntimeError(
                "The registered manual step-segmentation audit is not approved."
            )
    allowed = set(config["pilot"]["conditions"]) | set(
        config["confirmation"]["conditions"]
    )
    if args.condition not in allowed:
        raise ValueError(f"Unregistered condition: {args.condition}")
    if args.seed not in set(config["confirmation"]["seeds"]):
        raise ValueError(f"Unregistered seed: {args.seed}")
    if args.seed != int(config["pilot"]["seed"]):
        if args.condition not in set(config["confirmation"]["conditions"]):
            raise ValueError(
                "Non-pilot seeds are restricted to confirmation conditions."
            )
        require_passed_gate(PROJECT_ROOT, config["pilot_gate_dependency"])
    manifest_path = _resolve(args.data_manifest)
    manifest = read_json(manifest_path)
    if approval is not None and approval.get("data_manifest_sha256") != file_sha256(
        manifest_path
    ):
        raise ValueError("Step-audit approval is bound to another data manifest.")
    artifact = next(
        (row for row in manifest["artifacts"] if row["condition"] == args.condition),
        None,
    )
    if artifact is None:
        raise ValueError(f"Condition is absent from data manifest: {args.condition}")
    train_path = Path(artifact["path"])
    if file_sha256(train_path) != artifact["sha256"]:
        raise ValueError(f"Training data hash mismatch: {train_path}")
    run_name = f"{args.condition}__seed_{args.seed}"
    run_config = {
        "run_name": run_name,
        "condition": {"name": args.condition, "seed": args.seed},
        "data": {"train_path": str(train_path), "train_sha256": artifact["sha256"]},
        "student": utility_config["student"],
        "training": config["training"],
    }
    metrics = run_weighted_lora_sft(run_config, runtime_output_dir=runtime_dir)
    metrics.update(
        {
            "config_hash": canonical_sha256(config),
            "config_sha256": file_sha256(config_path),
            "utility_config_hash": canonical_sha256(utility_config),
            "utility_config_sha256": file_sha256(utility_config_path),
            "gate1_evidence": gate1,
            "data_manifest_sha256": file_sha256(manifest_path),
            "training_data_sha256": artifact["sha256"],
            "training_source_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/weighted_sft.py"
            ),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        }
    )
    write_json_exclusive(runtime_dir / "training_metrics.json", metrics)
    hashes = publish_files_hash_verified(
        runtime_dir,
        publish_dir,
        ("adapter_config.json", "adapter_model.safetensors", "training_metrics.json"),
    )
    (publish_dir / "TRAIN_COMPLETE").write_text(
        f"status=complete\nrun_name={run_name}\ncondition={args.condition}\nseed={args.seed}\n"
        f"config_hash={metrics['config_hash']}\ntraining_data_sha256={artifact['sha256']}\n"
        f"adapter_config_sha256={hashes['adapter_config.json']}\n"
        f"adapter_model_sha256={hashes['adapter_model.safetensors']}\n"
        f"training_metrics_sha256={hashes['training_metrics.json']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
