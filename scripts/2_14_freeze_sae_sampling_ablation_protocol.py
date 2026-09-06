#!/usr/bin/env python3
"""Freeze the SAE sampling-ablation overlay against the completed parent pilot."""

from __future__ import annotations

import argparse
import copy
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
        "--config", default="configs/phase2_sae_sampling_ablation_v1.json"
    )
    args = parser.parse_args()
    overlay_path = _resolve(args.config)
    overlay = read_json(overlay_path)
    parent_path = _resolve(overlay["parent_sae"]["protocol_path"])
    parent_marker_path = _resolve(
        overlay["parent_sae"]["completion_marker_path"]
    )
    marker = read_key_value_marker(parent_marker_path)
    if marker.get("status") != "passed" or marker.get("validated_sae_count") != "6":
        raise RuntimeError("The parent SAE pilot audit has not passed.")
    parent = read_json(parent_path)
    primary = overlay["primary_sae"]
    if (
        int(primary["layer_index"])
        not in parent["activation_extraction"]["layer_indices_zero_based"]
        or int(primary["k"]) not in parent["sae"]["k_values"]
        or int(primary["feature_count"]) != int(parent["sae"]["feature_count"])
    ):
        raise ValueError("Primary SAE does not match the completed parent protocol.")
    frozen = copy.deepcopy(parent)
    frozen.update(
        {
            "experiment_name": overlay["experiment_name"],
            "protocol_variant": overlay["protocol_variant"],
            "scope": overlay["scope"],
            "formal_claim_allowed": False,
            "claim_boundary": overlay["claim_boundary"],
        }
    )
    frozen["activation_extraction"]["layer_indices_zero_based"] = [
        int(primary["layer_index"])
    ]
    frozen["activation_extraction"]["nominal_depth_fractions"] = [0.65]
    frozen["sae"]["k_values"] = [int(primary["k"])]
    frozen["sae"]["seed"] = int(primary["training_seed"])
    frozen["token_sampling"] = {
        "method": "condition_specific_registered_sampling",
        "seed": int(parent["token_sampling"]["seed"]),
        "tokens_per_layer": dict(overlay["token_budget"]),
        "normalization": parent["token_sampling"]["normalization"],
    }
    frozen["token_sampling"]["tokens_per_layer"] = {
        split: int(overlay["token_budget"][split])
        for split in ("train", "dev", "test")
    }
    frozen["sampling_ablation"] = copy.deepcopy(overlay)
    frozen["sampling_ablation"]["parent_sae"] = {
        **overlay["parent_sae"],
        "protocol_sha256": file_sha256(parent_path),
        "protocol_hash": canonical_sha256(parent),
        "completion_marker_sha256": file_sha256(parent_marker_path),
    }
    result_root = _resolve(overlay["outputs"]["result_root"])
    if result_root.exists():
        raise FileExistsError(f"Refusing to overwrite {result_root}")
    protocol_dir = result_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = protocol_dir / "frozen_protocol.json"
    write_json_exclusive(frozen_path, frozen)
    manifest = {
        "status": "frozen",
        "experiment_name": overlay["experiment_name"],
        "config_hash": canonical_sha256(frozen),
        "config_path": str(frozen_path),
        "config_sha256": file_sha256(frozen_path),
        "overlay_path": str(overlay_path),
        "overlay_sha256": file_sha256(overlay_path),
        "parent_protocol_path": str(parent_path),
        "parent_protocol_sha256": file_sha256(parent_path),
        "parent_completion_marker_path": str(parent_marker_path),
        "parent_completion_marker_sha256": file_sha256(parent_marker_path),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = protocol_dir / "protocol_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (protocol_dir / "PROTOCOL_FROZEN").write_text(
        f"status=frozen\nconfig_hash={manifest['config_hash']}\n"
        f"config_sha256={manifest['config_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
