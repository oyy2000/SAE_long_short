#!/usr/bin/env python3
"""Seal the complete Phase-0 evidence chain without changing its Gate-0 decision."""

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

from length_budget_distill.factorial import file_sha256, nonempty_line_count, read_key_value_marker
from trace_length_observation.trace_observation import (
    protocol_hash,
    validate_trace_observation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    result_root = _resolve(args.result_root)
    stage_root = result_root / args.stage
    audit_dir = stage_root / "audit"
    root_marker = result_root / "PHASE0_COMPLETE"
    if audit_dir.exists() or root_marker.exists():
        raise FileExistsError("Refusing to overwrite Phase-0 final audit evidence.")
    config = _read_json(config_path)
    validate_trace_observation_config(config)
    config_hash = protocol_hash(config)
    errors: List[str] = []
    data_path = stage_root / "data/dataset_manifest.json"
    training_path = stage_root / "training/audit/training_audit.json"
    evaluation_path = stage_root / "evaluation/evaluation_manifest.json"
    analysis_path = stage_root / "analysis/trace_observation_analysis.json"
    gate_path = stage_root / "analysis/gate0_decision.json"
    artifact_manifest_path = stage_root / "analysis/analysis_artifact_manifest.json"
    for path in (data_path, training_path, evaluation_path, analysis_path, gate_path, artifact_manifest_path):
        _expect(path.is_file(), f"Missing evidence file: {path}", errors)
    if errors:
        raise SystemExit("Phase-0 final audit failed: " + " | ".join(errors))
    data = _read_json(data_path)
    training = _read_json(training_path)
    evaluation = _read_json(evaluation_path)
    analysis = _read_json(analysis_path)
    gate = _read_json(gate_path)
    artifacts = _read_json(artifact_manifest_path)
    _expect(data.get("status") == "complete", "Data manifest is incomplete.", errors)
    _expect(training.get("status") == "passed", "Training audit did not pass.", errors)
    _expect(evaluation.get("status") == "complete", "Evaluation manifest is incomplete.", errors)
    _expect(analysis.get("status") == "complete", "Analysis is incomplete.", errors)
    for label, payload in (
        ("data", data),
        ("training", training),
        ("evaluation", evaluation),
        ("analysis", analysis),
    ):
        _expect(payload.get("config_hash") == config_hash, f"{label} config hash mismatch.", errors)
    expected_adapters = int(config["training"]["expected_run_count"])
    expected_eval_runs = int(config["evaluation"]["expected_run_count"])
    expected_predictions = int(config["evaluation"]["limit"])
    _expect(int(data.get("run_count", -1)) == expected_adapters, "Data run count mismatch.", errors)
    _expect(
        int(training.get("validated_run_count", -1)) == expected_adapters,
        "Training adapter count mismatch.",
        errors,
    )
    _expect(int(evaluation.get("run_count", -1)) == expected_eval_runs, "Evaluation run count mismatch.", errors)
    supports: set[str] = set()
    for run in evaluation.get("runs", []):
        prediction_path = _resolve(str(run.get("prediction_path", "")))
        summary_path = _resolve(str(run.get("summary_path", "")))
        _expect(run.get("status") in {"complete", "skipped_complete"}, f"Incomplete evaluation: {run.get('model_id')}", errors)
        _expect(prediction_path.is_file(), f"Missing predictions: {prediction_path}", errors)
        _expect(summary_path.is_file(), f"Missing summary: {summary_path}", errors)
        if prediction_path.is_file():
            _expect(nonempty_line_count(prediction_path) == expected_predictions, f"Prediction count mismatch: {prediction_path}", errors)
            _expect(run.get("prediction_sha256") == file_sha256(prediction_path), f"Prediction hash mismatch: {prediction_path}", errors)
            supports.add(str(run.get("problem_ids_sha256")))
        if summary_path.is_file():
            _expect(run.get("summary_sha256") == file_sha256(summary_path), f"Summary hash mismatch: {summary_path}", errors)
    _expect(len(supports) == 1, "Evaluation problem support is not identical.", errors)
    for artifact in artifacts.get("artifacts", []):
        path = _resolve(str(artifact.get("path", "")))
        _expect(path.is_file(), f"Missing analysis artifact: {path}", errors)
        if path.is_file():
            _expect(artifact.get("sha256") == file_sha256(path), f"Analysis artifact hash mismatch: {path}", errors)
    _validate_marker(stage_root / "data/DATA_COMPLETE", config_hash, errors)
    _validate_marker(stage_root / "training/audit/TRAINING_COMPLETE", config_hash, errors)
    _validate_marker(stage_root / "evaluation/EVALUATION_COMPLETE", config_hash, errors)
    _validate_marker(stage_root / "analysis/ANALYSIS_COMPLETE", config_hash, errors)
    _expect(gate.get("status") in {"passed", "failed"}, "Gate-0 decision is invalid.", errors)
    _expect(gate.get("decision") in {"continue_to_phase1", "stop_sae_story"}, "Gate-0 action is invalid.", errors)
    report = {
        "status": "passed" if not errors else "failed",
        "stage": args.stage,
        "experiment_name": config["experiment_name"],
        "config_hash": config_hash,
        "evidence_level": "pipeline_smoke_only" if args.stage == "smoke" else config["evidence_level"],
        "formal_claim_allowed": False,
        "counts": {
            "selected_problems": int(data["selected_problem_count"]),
            "training_runs": int(training["validated_run_count"]),
            "evaluation_runs": int(evaluation["run_count"]),
            "predictions": int(evaluation["run_count"]) * expected_predictions,
            "analysis_arms": int(analysis["arm_count"]),
            "paired_contrasts": int(analysis["contrast_count"]),
        },
        "gate0": gate,
        "evidence": {
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "data_manifest": {"path": str(data_path), "sha256": file_sha256(data_path)},
            "training_audit": {"path": str(training_path), "sha256": file_sha256(training_path)},
            "evaluation_manifest": {"path": str(evaluation_path), "sha256": file_sha256(evaluation_path)},
            "analysis": {"path": str(analysis_path), "sha256": file_sha256(analysis_path)},
            "gate0_decision": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
            "analysis_artifact_manifest": {
                "path": str(artifact_manifest_path),
                "sha256": file_sha256(artifact_manifest_path),
            },
        },
        "errors": errors,
    }
    if errors:
        raise SystemExit("Phase-0 final audit failed: " + " | ".join(errors))
    audit_dir.mkdir(parents=True, exist_ok=False)
    audit_path = audit_dir / "completion_audit.json"
    _write_json(audit_path, report)
    (audit_dir / "AUDIT_COMPLETE").write_text(
        "status=passed\n"
        f"config_hash={config_hash}\n"
        f"completion_audit_sha256={file_sha256(audit_path)}\n"
        f"gate0_status={gate['status']}\n"
        f"gate0_decision={gate['decision']}\n",
        encoding="utf-8",
    )
    if args.stage == "formal":
        root_marker.write_text(
            "status=complete\n"
            f"config_hash={config_hash}\n"
            f"completion_audit_sha256={file_sha256(audit_path)}\n"
            f"gate0_status={gate['status']}\n"
            f"gate0_decision={gate['decision']}\n"
            f"training_runs={expected_adapters}\n"
            f"evaluation_runs={expected_eval_runs}\n"
            "formal_claim_allowed=false\n",
            encoding="utf-8",
        )


def _validate_marker(path: Path, config_hash: str, errors: List[str]) -> None:
    if not path.is_file():
        errors.append(f"Missing completion marker: {path}")
        return
    try:
        marker = read_key_value_marker(path)
    except (OSError, ValueError) as exc:
        errors.append(f"Invalid completion marker {path}: {exc}")
        return
    if marker.get("config_hash") != config_hash:
        errors.append(f"Completion marker config mismatch: {path}")


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
