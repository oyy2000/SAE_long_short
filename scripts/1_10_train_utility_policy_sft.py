#!/usr/bin/env python3
"""Train one full-completion LoRA on a registered Phase-1 trace-selection policy."""

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
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl
from length_budget_distill.weighted_sft import run_weighted_lora_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument("--policy-data", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-output-dir", required=True)
    parser.add_argument("--publish-output-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    policy_path = _resolve(args.policy_data)
    runtime_dir = Path(args.runtime_output_dir).resolve()
    publish_dir = _resolve(args.publish_output_dir)
    if runtime_dir.exists() or publish_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {runtime_dir} or {publish_dir}")
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    if args.seed not in (17, 42, 73):
        raise ValueError("Unexpected training seed.")
    rows = list(read_jsonl(policy_path))
    if len(rows) != len({row["problem_id"] for row in rows}):
        raise ValueError("Policy data is not one trace per question.")
    run_name = f"{args.policy}__seed_{args.seed}"
    training = dict(config["policy_sft"]["training"])
    training.update(
        {
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "completion_separator": "",
        }
    )
    mid_anchor_evidence = None
    if args.policy == "mid_sft_anchor":
        manifest_path = policy_path.parent / "mid_sft_anchor_data_manifest.json"
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") != "complete"
            or manifest.get("data_sha256") != file_sha256(policy_path)
            or int(manifest["record_count"]) != len(rows)
        ):
            raise ValueError("Mid-SFT anchor data is not bound to its manifest.")
        training.update(
            {
                "preserve_input_order": True,
                "scheduler_total_optimizer_steps": int(
                    manifest["full_optimizer_steps"]
                ),
            }
        )
        mid_anchor_evidence = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "full_optimizer_steps": int(manifest["full_optimizer_steps"]),
            "mid_optimizer_steps": int(manifest["mid_optimizer_steps"]),
            "mid_step_fraction": float(manifest["mid_step_fraction"]),
            "preserves_full_run_prefix_order": True,
            "uses_full_run_scheduler_horizon": True,
        }
    student = dict(config["student"])
    student["tokenizer_name"] = student["model_name"]
    run_config = {
        "run_name": run_name,
        "condition": {"name": args.policy, "seed": args.seed},
        "data": {
            "train_path": str(policy_path),
            "train_sha256": file_sha256(policy_path),
        },
        "student": student,
        "training": training,
    }
    metrics = run_weighted_lora_sft(run_config, runtime_output_dir=runtime_dir)
    if mid_anchor_evidence is not None:
        if (
            int(metrics["optimizer_steps"])
            != mid_anchor_evidence["mid_optimizer_steps"]
            or int(metrics["scheduler_total_optimizer_steps"])
            != mid_anchor_evidence["full_optimizer_steps"]
            or not bool(metrics["preserved_input_order"])
        ):
            raise RuntimeError(
                "Mid-SFT anchor did not reproduce the registered prefix."
            )
    metrics.update(
        {
            "policy": args.policy,
            "gate_evidence": gate,
            "config_hash": canonical_sha256(config),
            "config_sha256": file_sha256(config_path),
            "policy_data_sha256": file_sha256(policy_path),
            "training_source_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/weighted_sft.py"
            ),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "mid_sft_anchor_evidence": mid_anchor_evidence,
        }
    )
    metrics_path = runtime_dir / "training_metrics.json"
    write_json_exclusive(metrics_path, metrics)
    hashes = publish_files_hash_verified(
        runtime_dir,
        publish_dir,
        ("adapter_config.json", "adapter_model.safetensors", "training_metrics.json"),
    )
    marker = publish_dir / "TRAIN_COMPLETE"
    marker.write_text(
        f"status=complete\nrun_name={run_name}\npolicy={args.policy}\nseed={args.seed}\n"
        f"config_hash={metrics['config_hash']}\npolicy_data_sha256={metrics['policy_data_sha256']}\n"
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
