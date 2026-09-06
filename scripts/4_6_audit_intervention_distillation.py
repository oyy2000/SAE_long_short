#!/usr/bin/env python3
"""Final evidence audit for the exploratory SAE-intervention distillation pilot."""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    publish_files_hash_verified,
    read_json,
    validated_training_artifacts,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    config_hash = canonical_sha256(config)
    checkpoint_root, evaluation_root = _resolve(args.checkpoint_root), _resolve(args.evaluation_root)
    analysis_dir, publish_output_dir = _resolve(args.analysis_dir), _resolve(args.output_dir)
    if publish_output_dir.exists():
        raise FileExistsError(publish_output_dir)
    stage_root_value = os.environ.get("SAE_AUDIT_OUTPUT_STAGE_ROOT")
    stage_root = Path(stage_root_value) if stage_root_value else None
    output_dir = stage_root / "final_audit" if stage_root else publish_output_dir
    if output_dir.exists():
        raise FileExistsError(output_dir)
    errors, training, evaluations = [], [], []
    result_root = _resolve(config["outputs"]["result_root"])
    _check_marker(result_root / "main_generation/MAIN_GENERATION_COMPLETE", "complete", errors)
    _check_marker(result_root / "sft_data/SFT_DATA_COMPLETE", "complete", errors)
    for shard in range(3):
        path = result_root / f"training/training_launcher_{shard:02d}_of_03.json"
        _check_json_status(path, "complete", errors)
        path = evaluation_root / f"evaluation_launcher_{shard:02d}_of_03.json"
        _check_json_status(path, "complete", errors)
    expected = []
    for budget in config["student_sft"]["budget_regimes"]:
        for condition in config["main_generation"]["conditions"]:
            for seed in config["student_sft"]["seeds"]:
                model_id = f"{budget}__{condition}__seed_{seed}"
                expected.append(model_id)
                try:
                    artifact = _remote_io_retry(
                        lambda model_id=model_id: _validated_training(checkpoint_root, model_id),
                        f"training:{model_id}",
                    )
                    if artifact["marker"].get("config_hash") != config_hash:
                        raise ValueError("training marker protocol hash mismatch")
                    training.append({"model_id": model_id, "root": artifact["root"], "used_recovery_mirror": artifact["used_recovery_mirror"], "marker_sha256": artifact["marker_sha256"], "hashes": artifact["hashes"]})
                except Exception as exc:
                    errors.append(f"training:{model_id}:{exc}")
    for model_id in expected + ["base_student"]:
        recovery_root_value = os.environ.get("SAE_EVALUATION_RECOVERY_ROOT")
        recovery = Path(recovery_root_value) / model_id if recovery_root_value else None
        root = recovery if recovery is not None and recovery.is_dir() else evaluation_root / model_id
        try:
            valid, marker, manifest_path = _remote_io_retry(
                lambda root=root: _validate_evaluation(root),
                f"evaluation:{model_id}",
            )
        except Exception as exc:
            errors.append(f"evaluation:{model_id}:{exc}")
            continue
        if not valid:
            errors.append(f"evaluation:{model_id}:invalid marker or manifest")
        else:
            evaluations.append({"model_id": model_id, "root": str(root), "used_recovery_mirror": root != evaluation_root / model_id, "manifest_sha256": file_sha256(manifest_path), "predictions_sha256": marker.get("predictions_sha256")})
    analysis_marker = read_key_value_marker(analysis_dir / "ANALYSIS_COMPLETE")
    report_path = analysis_dir / "analysis_report.json"
    if analysis_marker.get("status") != "complete" or not report_path.is_file() or analysis_marker.get("report_sha256") != file_sha256(report_path):
        errors.append("analysis:invalid marker or report")
    report = {
        "status": "passed" if not errors else "failed",
        "experiment_name": config["experiment_name"],
        "config_hash": config_hash,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "expected_training_runs": len(expected),
        "validated_training_runs": len(training),
        "expected_evaluations": len(expected) + 1,
        "validated_evaluations": len(evaluations),
        "training_evidence": training,
        "evaluation_evidence": evaluations,
        "analysis_report_path": str(report_path),
        "analysis_report_sha256": file_sha256(report_path) if report_path.is_file() else None,
        "errors": errors,
        "formal_claim_allowed": False,
    }
    output_dir.mkdir(parents=True)
    report_out = output_dir / "final_audit.json"
    write_json_exclusive(report_out, report)
    if errors:
        raise SystemExit("Final audit failed: " + " | ".join(errors))
    (output_dir / "PILOT_COMPLETE").write_text(
        "status=passed\n"
        f"config_hash={config_hash}\n"
        f"audit_sha256={file_sha256(report_out)}\n"
        f"training_run_count={len(training)}\n"
        f"evaluation_count={len(evaluations)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    if stage_root is not None:
        publish_files_hash_verified(
            output_dir,
            publish_output_dir,
            ("final_audit.json", "PILOT_COMPLETE"),
            attempts=120,
            wait_seconds=5.0,
        )
    print(json.dumps(report, indent=2), flush=True)


def _check_marker(path: Path, expected: str, errors: list[str]) -> None:
    try:
        marker = _remote_io_retry(lambda: read_key_value_marker(path), f"marker:{path}")
        if marker.get("status") != expected:
            errors.append(f"marker:{path}:status={marker.get('status')}")
    except Exception as exc:
        errors.append(f"marker:{path}:{exc}")


def _check_json_status(path: Path, expected: str, errors: list[str]) -> None:
    try:
        if _remote_io_retry(lambda: read_json(path), f"manifest:{path}").get("status") != expected:
            errors.append(f"manifest:{path}:incomplete")
    except Exception as exc:
        errors.append(f"manifest:{path}:{exc}")


def _validate_evaluation(root: Path):
    marker = read_key_value_marker(root / "EVALUATION_COMPLETE")
    manifest_path = root / "evaluation_manifest.json"
    prediction_path = root / "predictions.jsonl"
    summary_path = root / "summary.json"
    valid = (
        marker.get("status") == "complete"
        and manifest_path.is_file()
        and marker.get("manifest_sha256") == file_sha256(manifest_path)
        and marker.get("predictions_sha256") == file_sha256(prediction_path)
        and marker.get("summary_sha256") == file_sha256(summary_path)
    )
    provenance_path = root / "RECOVERY_PROVENANCE.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        observed = {
            "predictions.jsonl": file_sha256(prediction_path),
            "summary.json": file_sha256(summary_path),
            "evaluation_manifest.json": file_sha256(manifest_path),
            "EVALUATION_COMPLETE": file_sha256(root / "EVALUATION_COMPLETE"),
        }
        valid = valid and (
            provenance.get("status") == "hash_equivalent_evaluation_recovery_copy"
            and provenance.get("content_changed") is False
            and provenance.get("file_hashes") == observed
        )
    return valid, marker, manifest_path


def _remote_io_retry(function, label: str):
    for attempt in range(1, 121):
        try:
            return function()
        except OSError as exc:
            if exc.errno != errno.EREMOTEIO or attempt == 120:
                raise
            print(f"Transient remote I/O while reading {label}; retry={attempt}/120", flush=True)
            time.sleep(5)
    raise AssertionError("Unreachable remote-I/O retry state")


def _validated_training(checkpoint_root: Path, model_id: str):
    recovery_root = os.environ.get("SAE_ADAPTER_RECOVERY_ROOT")
    recovery = Path(recovery_root) / model_id if recovery_root else None
    root = recovery if recovery is not None and recovery.is_dir() else checkpoint_root / model_id
    artifact = validated_training_artifacts(root)
    artifact["canonical_root"] = str(checkpoint_root / model_id)
    artifact["used_recovery_mirror"] = root != checkpoint_root / model_id
    if artifact["used_recovery_mirror"]:
        provenance = read_json(root / "RECOVERY_PROVENANCE.json")
        if provenance.get("status") != "hash_equivalent_recovery_copy" or provenance.get("content_changed") is not False:
            raise ValueError(f"Invalid recovery provenance: {root}")
        expected_hashes = {
            "adapter_config.json": artifact["hashes"]["adapter_config"],
            "adapter_model.safetensors": artifact["hashes"]["adapter_model"],
            "training_metrics.json": artifact["hashes"]["training_metrics"],
        }
        if provenance.get("file_hashes") != expected_hashes:
            raise ValueError(f"Recovery provenance hash mismatch: {root}")
    return artifact


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
