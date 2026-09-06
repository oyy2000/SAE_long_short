#!/usr/bin/env python3
"""Audit the SAE sampling ablation and publish its completion marker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


CONDITIONS = (
    "full_token_uniform",
    "full_trace_balanced",
    "prefix64_trace_balanced",
)
FIGURE_STEMS = (
    "01_training_sampling_exposure",
    "02_reconstruction_and_feature_yield",
    "03_full_vs_first64_feature_effects",
    "04_candidate_dictionary_matching",
    "05_confirmed_feature_position_profiles",
    "06_confirmed_feature_token_signatures",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase2_sae_sampling_ablation_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    parser.add_argument("--failed-training-job", default="278985")
    parser.add_argument("--retry-training-job", default="278990")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    ablation = config["sampling_ablation"]
    result_root = _resolve(ablation["outputs"]["result_root"])
    audit_dir = result_root / "audit"
    root_marker = result_root / "SAE_SAMPLING_ABLATION_COMPLETE"
    if audit_dir.exists() or root_marker.exists():
        raise FileExistsError(f"Refusing to overwrite {audit_dir} or {root_marker}")
    errors = []
    condition_evidence = []
    try:
        protocol_marker = read_key_value_marker(config_path.parent / "PROTOCOL_FROZEN")
        if protocol_marker.get("status") != "frozen" or protocol_marker.get(
            "config_hash"
        ) != canonical_sha256(config):
            raise ValueError("Protocol marker does not bind the frozen config.")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        errors.append({"stage": "protocol", "error": str(exc)})

    for name in CONDITIONS:
        try:
            run_dir = result_root / "condition_scores" / name
            marker_path = run_dir / "CONDITION_SCORING_COMPLETE"
            summary_path = run_dir / "scoring_summary.json"
            marker = read_key_value_marker(marker_path)
            summary = read_json(summary_path)
            if marker.get("status") != "complete" or marker.get("condition") != name:
                raise ValueError("Condition marker is incomplete or misidentified.")
            if marker.get("summary_sha256") != file_sha256(summary_path):
                raise ValueError("Condition summary hash mismatch.")
            if summary["config_hash"] != canonical_sha256(config):
                raise ValueError("Condition config hash mismatch.")
            for artifact in summary["artifacts"].values():
                if file_sha256(artifact["path"]) != artifact["sha256"]:
                    raise ValueError(f"Artifact hash mismatch: {artifact['path']}")
            if int(summary["dev_question_count"]) != 132 or int(
                summary["test_question_count"]
            ) != 132:
                raise ValueError("Unexpected paired question count.")
            condition_evidence.append(
                {
                    "condition": name,
                    "summary_path": str(summary_path),
                    "summary_sha256": file_sha256(summary_path),
                    "checkpoint_path": summary["input_evidence"]["checkpoint_path"],
                    "checkpoint_sha256": summary["input_evidence"]["checkpoint_sha256"],
                    "candidate_count": int(summary["candidate_count"]),
                    "heldout_confirmed_count": int(summary["heldout_confirmed_count"]),
                }
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append({"stage": "condition", "condition": name, "error": str(exc)})

    analysis_evidence = None
    try:
        analysis_dir = result_root / "analysis"
        marker_path = analysis_dir / "ANALYSIS_COMPLETE"
        summary_path = analysis_dir / "analysis_summary.json"
        marker = read_key_value_marker(marker_path)
        summary = read_json(summary_path)
        if marker.get("status") != "complete" or marker.get(
            "analysis_summary_sha256"
        ) != file_sha256(summary_path):
            raise ValueError("Analysis marker or summary hash mismatch.")
        if summary["config_hash"] != canonical_sha256(config):
            raise ValueError("Analysis config hash mismatch.")
        for artifact in summary["artifacts"].values():
            if file_sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Analysis artifact hash mismatch: {artifact['path']}")
        figure_root = _resolve(ablation["outputs"]["figure_root"])
        required = [figure_root / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing required figure: {path}")
        analysis_evidence = {
            "summary_path": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "report_path": summary["artifacts"]["analysis_report.md"]["path"],
            "required_figure_count": len(required),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        errors.append({"stage": "analysis", "error": str(exc)})

    status = "passed" if len(condition_evidence) == len(CONDITIONS) and analysis_evidence and not errors else "failed"
    audit_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "status": status,
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "expected_condition_count": len(CONDITIONS),
        "validated_condition_count": len(condition_evidence),
        "conditions": condition_evidence,
        "analysis_evidence": analysis_evidence,
        "execution_history": {
            "failed_training_job": args.failed_training_job,
            "failed_reason": "home quota exhaustion while writing the step-1000 checkpoint; no final model was accepted",
            "retry_training_job": args.retry_training_job,
            "retry_policy": "from scratch under the identical frozen config and seed, with checkpoint output relocated to BeeGFS",
        },
        "errors": errors,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    audit_path = audit_dir / "sae_sampling_ablation_audit.json"
    write_json_exclusive(audit_path, payload)
    marker_text = (
        f"status={status}\nconfig_hash={payload['config_hash']}\n"
        f"audit_sha256={file_sha256(audit_path)}\n"
        f"validated_condition_count={len(condition_evidence)}\n"
        f"required_figure_count={analysis_evidence['required_figure_count'] if analysis_evidence else 0}\n"
        "formal_claim_allowed=false\n"
    )
    (audit_dir / "AUDIT_COMPLETE").write_text(marker_text, encoding="utf-8")
    root_marker.write_text(marker_text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "passed":
        raise SystemExit(1)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
