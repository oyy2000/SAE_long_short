#!/usr/bin/env python3
"""Audit all Phase-0 controlled-SFT adapters and budget accounting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import (
    file_sha256,
    read_key_value_marker,
    validated_adapter_evidence,
)
from trace_length_observation.trace_observation import (
    BUDGET_REGIMES,
    LENGTH_RANKS,
    protocol_hash,
    validate_trace_observation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    dataset_path = _resolve(args.dataset_manifest)
    training_dir = _resolve(args.training_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite training audit: {output_dir}")
    config = _read_json(config_path)
    validate_trace_observation_config(config)
    config_hash = protocol_hash(config)
    dataset = _read_json(dataset_path)
    expected_runs = {str(run["run_name"]): dict(run) for run in dataset.get("runs", [])}
    errors: List[str] = []
    _expect(dataset.get("status") == "complete", "Dataset manifest is incomplete.", errors)
    _expect(dataset.get("config_hash") == config_hash, "Dataset config hash mismatch.", errors)
    _expect(
        len(expected_runs) == int(config["training"]["expected_run_count"]),
        "Dataset run matrix has the wrong cardinality.",
        errors,
    )
    launcher_shards = int(config["training"]["launcher_shards"])
    manifest_paths = sorted(training_dir.glob("training_manifest_shard_*_of_*.json"))
    _expect(len(manifest_paths) == launcher_shards, "Training shard manifest count mismatch.", errors)
    observed_runs: Dict[str, Dict[str, Any]] = {}
    audited_runs: List[Dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        _expect(manifest.get("status") == "complete", f"Incomplete manifest: {manifest_path}", errors)
        _expect(manifest.get("config_hash") == config_hash, f"Config mismatch: {manifest_path}", errors)
        _expect(
            manifest.get("dataset_manifest_sha256") == file_sha256(dataset_path),
            f"Dataset hash mismatch: {manifest_path}",
            errors,
        )
        for run in manifest.get("runs", []):
            run_name = str(run.get("run_name", ""))
            if run_name in observed_runs:
                errors.append(f"Duplicate training run across shards: {run_name}")
                continue
            observed_runs[run_name] = dict(run)
    _expect(set(observed_runs) == set(expected_runs), "Training run identities are incomplete.", errors)
    for run_name in sorted(expected_runs):
        expected = expected_runs[run_name]
        observed = observed_runs.get(run_name)
        if observed is None:
            continue
        _expect(
            observed.get("status") in {"complete", "skipped_complete"},
            f"Training run is incomplete: {run_name}",
            errors,
        )
        for key in (
            "budget_regime",
            "loss_normalization",
            "loss_mask",
            "length_rank",
            "seed",
            "train_path",
            "train_sha256",
            "record_count",
            "unique_problem_count",
            "actual_completion_tokens",
            "actual_model_input_tokens",
        ):
            _expect(observed.get(key) == expected.get(key), f"Run field mismatch: {run_name} {key}", errors)
        evidence = validated_adapter_evidence(str(observed.get("output_dir", "")))
        if evidence is None:
            errors.append(f"Invalid adapter evidence: {run_name}")
            continue
        marker_path = Path(str(observed["output_dir"])) / "TRAIN_COMPLETE"
        marker = read_key_value_marker(marker_path)
        _expect(
            marker.get("masking_source_sha256")
            == str(observed.get("source_hashes", {}).get("masking")),
            f"Masking source marker mismatch: {run_name}",
            errors,
        )
        metrics_path = Path(str(observed["output_dir"])) / "training_metrics.json"
        if not metrics_path.is_file():
            errors.append(f"Missing training metrics: {run_name}")
            continue
        metrics = _read_json(metrics_path)
        _expect(metrics.get("status") == "complete", f"Incomplete metrics: {run_name}", errors)
        _expect(metrics.get("run_name") == run_name, f"Metrics run mismatch: {run_name}", errors)
        _expect(
            int(metrics.get("record_count", -1)) == int(expected["record_count"]),
            f"Metrics record count mismatch: {run_name}",
            errors,
        )
        _expect(
            int(metrics.get("completion_token_updates", -1)) == int(expected["actual_completion_tokens"]),
            f"Completion-token update mismatch: {run_name}",
            errors,
        )
        _expect(
            int(metrics.get("model_input_token_updates", -1))
            == int(expected["actual_model_input_tokens"]),
            f"Model-input token update mismatch: {run_name}",
            errors,
        )
        _expect(
            marker.get("training_metrics_sha256") == file_sha256(metrics_path),
            f"Training metrics marker mismatch: {run_name}",
            errors,
        )
        audited_runs.append(
            {
                **{key: expected[key] for key in expected},
                "output_dir": str(observed["output_dir"]),
                "run_config_path": str(observed["config_path"]),
                "run_config_sha256": str(observed["run_config_sha256"]),
                "adapter_config_sha256": evidence["adapter_config_sha256"],
                "adapter_model_sha256": evidence["adapter_model_sha256"],
                "training_metrics_path": str(metrics_path),
                "training_metrics_sha256": file_sha256(metrics_path),
                "optimizer_steps": int(metrics["optimizer_steps"]),
                "completion_token_updates": int(metrics["completion_token_updates"]),
                "model_input_token_updates": int(metrics["model_input_token_updates"]),
                "padded_model_token_updates": int(metrics["padded_model_token_updates"]),
                "effective_loss_token_updates": int(metrics["effective_loss_token_updates"]),
                "total_parameter_count": int(metrics["total_parameter_count"]),
                "trainable_parameter_count": int(metrics["trainable_parameter_count"]),
                "approximate_nonpadding_training_flops": int(
                    metrics["approximate_nonpadding_training_flops"]
                ),
                "approximate_padded_training_flops": int(
                    metrics["approximate_padded_training_flops"]
                ),
                "flops_proxy_definition": str(metrics["flops_proxy_definition"]),
                "mean_train_loss": float(metrics["mean_train_loss"]),
                "elapsed_seconds": float(metrics["elapsed_seconds"]),
            }
        )
    _audit_fairness(audited_runs, config, errors)
    _expect(
        len(audited_runs) == int(config["training"]["expected_run_count"]),
        "Validated adapter count mismatch.",
        errors,
    )
    report = {
        "status": "passed" if not errors else "failed",
        "stage": dataset.get("stage"),
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "dataset_manifest_path": str(dataset_path),
        "dataset_manifest_sha256": file_sha256(dataset_path),
        "training_manifest_paths": [str(path) for path in manifest_paths],
        "training_manifest_sha256s": [file_sha256(path) for path in manifest_paths],
        "expected_run_count": int(config["training"]["expected_run_count"]),
        "validated_run_count": len(audited_runs),
        "runs": audited_runs,
        "errors": errors,
    }
    if errors:
        raise SystemExit("Phase-0 training audit failed: " + " | ".join(errors))
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "training_audit.json"
    _write_json(report_path, report)
    (output_dir / "TRAINING_COMPLETE").write_text(
        "status=passed\n"
        f"config_hash={config_hash}\n"
        f"dataset_manifest_sha256={file_sha256(dataset_path)}\n"
        f"training_audit_sha256={file_sha256(report_path)}\n"
        f"run_count={len(audited_runs)}\n",
        encoding="utf-8",
    )


def _audit_fairness(
    runs: List[Mapping[str, Any]], config: Mapping[str, Any], errors: List[str]
) -> None:
    grouped: Dict[tuple[str, str, str, int], Dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        key = (
            str(run["budget_regime"]),
            str(run["loss_normalization"]),
            str(run["loss_mask"]),
            int(run["seed"]),
        )
        grouped.setdefault(key, {})[str(run["length_rank"])] = run
    maximum_gap = int(config["training"]["maximum_token_budget_gap"])
    for key, by_rank in grouped.items():
        if set(by_rank) != set(LENGTH_RANKS):
            errors.append(f"Fairness cell is missing a length rank: {key}")
            continue
        regime = key[0]
        records = [int(by_rank[rank]["record_count"]) for rank in LENGTH_RANKS]
        completion_tokens = [
            int(by_rank[rank]["completion_token_updates"]) for rank in LENGTH_RANKS
        ]
        model_input_tokens = [
            int(by_rank[rank]["model_input_token_updates"]) for rank in LENGTH_RANKS
        ]
        unique = [int(by_rank[rank]["unique_problem_count"]) for rank in LENGTH_RANKS]
        if regime == "equal_examples":
            _expect(len(set(records)) == 1, f"Equal-example record counts differ: {key}", errors)
            _expect(len(set(unique)) == 1, f"Equal-example problem counts differ: {key}", errors)
        elif regime == "equal_target_tokens":
            _expect(
                max(completion_tokens) - min(completion_tokens) <= maximum_gap,
                f"Target-token budgets differ: {key}",
                errors,
            )
        elif regime == "equal_processed_tokens":
            _expect(
                max(model_input_tokens) - min(model_input_tokens) <= maximum_gap,
                f"Model-input token budgets differ: {key}",
                errors,
            )
        else:
            errors.append(f"Unknown fairness regime in audit: {regime}")


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


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
