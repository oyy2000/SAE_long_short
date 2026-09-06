#!/usr/bin/env python3
"""Audit all registered SAE pilot runs and issue the pilot completion marker."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument(
        "--training-root",
        default="results/phase2_sae_pilot_v1/formal/sae_training",
    )
    parser.add_argument(
        "--output-dir", default="results/phase2_sae_pilot_v1/formal/audit"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    training_root = _resolve(args.training_root)
    output_dir = _resolve(args.output_dir)
    root_marker = output_dir.parents[1] / "SAE_PILOT_COMPLETE"
    if output_dir.exists() or root_marker.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {root_marker}")
    config = read_json(config_path)
    expected = [
        (int(layer), int(k))
        for layer in config["activation_extraction"]["layer_indices_zero_based"]
        for k in config["sae"]["k_values"]
    ]
    runs = []
    errors = []
    for layer, k in expected:
        run_dir = training_root / f"layer_{layer:02d}_k_{k:03d}"
        marker_path = run_dir / "SAE_TRAINING_COMPLETE"
        metrics_path = run_dir / "training_metrics.json"
        try:
            marker = read_key_value_marker(marker_path)
            metrics = read_json(metrics_path)
            model_path = Path(metrics["model_path"])
            if marker.get("status") != "complete":
                raise ValueError("training marker is incomplete")
            if marker.get("training_metrics_sha256") != file_sha256(metrics_path):
                raise ValueError("training metrics hash mismatch")
            if marker.get("model_sha256") != file_sha256(model_path):
                raise ValueError("model hash mismatch")
            if (
                metrics.get("status") != "complete"
                or int(metrics["layer_index"]) != layer
                or int(metrics["k"]) != k
                or metrics["config_hash"] != canonical_sha256(config)
            ):
                raise ValueError("training metrics identity mismatch")
            for split in ("train", "dev", "test"):
                values = metrics["final_metrics"][split]
                for name in config["pilot_outputs"]["required_metrics"]:
                    if name not in values:
                        raise ValueError(f"missing {split} metric {name}")
            runs.append(
                {
                    "layer_index": layer,
                    "k": k,
                    "metrics_path": str(metrics_path),
                    "metrics_sha256": file_sha256(metrics_path),
                    "model_path": str(model_path),
                    "model_sha256": file_sha256(model_path),
                    "train_mse": metrics["final_metrics"]["train"]["mse"],
                    "dev_mse": metrics["final_metrics"]["dev"]["mse"],
                    "test_mse": metrics["final_metrics"]["test"]["mse"],
                    "test_explained_variance": metrics["final_metrics"]["test"][
                        "explained_variance"
                    ],
                    "test_mean_l0": metrics["final_metrics"]["test"]["mean_l0"],
                    "test_dead_feature_fraction": metrics["final_metrics"]["test"][
                        "dead_feature_fraction"
                    ],
                }
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append({"layer_index": layer, "k": k, "error": str(exc)})
    status = "passed" if len(runs) == len(expected) and not errors else "failed"
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "sae_metrics.csv"
    if runs:
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    payload = {
        "status": status,
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "expected_sae_count": int(config["pilot_outputs"]["required_sae_count"]),
        "validated_sae_count": len(runs),
        "runs": runs,
        "errors": errors,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
        "claim_boundary": "Reconstruction evidence only; utility interpretation and causal steering have not been tested.",
    }
    audit_path = output_dir / "sae_pilot_audit.json"
    write_json_exclusive(audit_path, payload)
    marker_text = (
        f"status={status}\nconfig_hash={payload['config_hash']}\n"
        f"audit_sha256={file_sha256(audit_path)}\n"
        f"validated_sae_count={len(runs)}\nformal_claim_allowed=false\n"
    )
    (output_dir / "AUDIT_COMPLETE").write_text(marker_text, encoding="utf-8")
    root_marker.write_text(marker_text, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "passed":
        raise SystemExit(1)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
