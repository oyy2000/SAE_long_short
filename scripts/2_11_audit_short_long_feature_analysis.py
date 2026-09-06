#!/usr/bin/env python3
"""Audit the six SAE score runs and publication-ready feature figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase2_short_long_feature_analysis_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    result_root = _resolve(config["outputs"]["result_root"])
    audit_dir = result_root / "audit"
    root_marker = result_root / "SHORT_LONG_FEATURE_ANALYSIS_COMPLETE"
    if audit_dir.exists() or root_marker.exists():
        raise FileExistsError(f"Refusing to overwrite {audit_dir} or {root_marker}")
    parent_config = read_json(_resolve(config["parent_sae"]["config_path"]))
    expected = [
        (int(layer), int(k))
        for layer in parent_config["activation_extraction"]["layer_indices_zero_based"]
        for k in parent_config["sae"]["k_values"]
    ]
    score_runs = []
    errors = []
    for layer, k in expected:
        run_dir = result_root / "feature_scores" / f"layer_{layer:02d}_k_{k:03d}"
        try:
            marker_path = run_dir / "FEATURE_SCORING_COMPLETE"
            summary_path = run_dir / "scoring_summary.json"
            marker = read_key_value_marker(marker_path)
            summary = read_json(summary_path)
            if marker.get("status") != "complete":
                raise ValueError("scoring marker is incomplete")
            if marker.get("summary_sha256") != file_sha256(summary_path):
                raise ValueError("scoring summary hash mismatch")
            if int(summary["layer_index"]) != layer or int(summary["k"]) != k:
                raise ValueError("scoring identity mismatch")
            if summary["config_hash"] != canonical_sha256(config):
                raise ValueError("analysis config hash mismatch")
            for artifact in summary["artifacts"].values():
                if file_sha256(artifact["path"]) != artifact["sha256"]:
                    raise ValueError(f"scoring artifact hash mismatch: {artifact['path']}")
            score_runs.append(
                {
                    "layer_index": layer,
                    "k": k,
                    "summary_path": str(summary_path),
                    "summary_sha256": file_sha256(summary_path),
                    "candidate_count": int(summary["candidate_count"]),
                    "heldout_confirmed_count": int(summary["heldout_confirmed_count"]),
                }
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append({"layer_index": layer, "k": k, "error": str(exc)})
    try:
        analysis_dir = result_root / "analysis"
        analysis_marker_path = analysis_dir / "ANALYSIS_COMPLETE"
        analysis_summary_path = analysis_dir / "analysis_summary.json"
        analysis_marker = read_key_value_marker(analysis_marker_path)
        analysis_summary = read_json(analysis_summary_path)
        if analysis_marker.get("status") != "complete" or analysis_marker.get(
            "analysis_summary_sha256"
        ) != file_sha256(analysis_summary_path):
            raise ValueError("analysis marker or summary hash mismatch")
        for artifact in analysis_summary["artifacts"].values():
            if file_sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"analysis artifact hash mismatch: {artifact['path']}")
        figure_root = _resolve(config["outputs"]["figure_root"])
        for filename in config["outputs"]["required_figures"]:
            path = figure_root / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing required figure: {path}")
        analysis_evidence = {
            "summary_path": str(analysis_summary_path),
            "summary_sha256": file_sha256(analysis_summary_path),
            "report_path": analysis_summary["artifacts"]["analysis_report.md"]["path"],
            "required_figure_count": len(config["outputs"]["required_figures"]),
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        errors.append({"stage": "merged_analysis", "error": str(exc)})
        analysis_evidence = None
    status = "passed" if len(score_runs) == len(expected) and not errors else "failed"
    audit_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "status": status,
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "expected_sae_count": len(expected),
        "validated_sae_count": len(score_runs),
        "score_runs": score_runs,
        "analysis_evidence": analysis_evidence,
        "errors": errors,
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
    }
    audit_path = audit_dir / "short_long_feature_analysis_audit.json"
    write_json_exclusive(audit_path, payload)
    marker_text = (
        f"status={status}\nconfig_hash={payload['config_hash']}\n"
        f"audit_sha256={file_sha256(audit_path)}\n"
        f"validated_sae_count={len(score_runs)}\n"
        f"required_figure_count={len(config['outputs']['required_figures'])}\n"
        "formal_claim_allowed=false\n"
    )
    (audit_dir / "AUDIT_COMPLETE").write_text(marker_text, encoding="utf-8")
    root_marker.write_text(marker_text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "passed":
        raise SystemExit(1)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
