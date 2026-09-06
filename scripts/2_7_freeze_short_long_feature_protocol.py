#!/usr/bin/env python3
"""Freeze the short-versus-long SAE interpretation protocol after parent audit."""

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
        "--config", default="configs/phase2_short_long_feature_analysis_v1.json"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    parent = config["parent_sae"]
    parent_config_path = _resolve(parent["config_path"])
    parent_marker_path = _resolve(parent["completion_marker_path"])
    marker = read_key_value_marker(parent_marker_path)
    if marker.get("status") != "passed" or marker.get("validated_sae_count") != "6":
        raise RuntimeError("The six-SAE parent audit has not passed.")
    parent_config = read_json(parent_config_path)
    frozen = dict(config)
    frozen["parent_sae"] = {
        **parent,
        "config_sha256": file_sha256(parent_config_path),
        "config_hash": canonical_sha256(parent_config),
        "completion_marker_sha256": file_sha256(parent_marker_path),
    }
    output_root = _resolve(config["outputs"]["result_root"])
    protocol_dir = output_root / "protocol"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite analysis root: {output_root}")
    protocol_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = protocol_dir / "frozen_protocol.json"
    write_json_exclusive(frozen_path, frozen)
    manifest = {
        "status": "frozen",
        "experiment_name": frozen["experiment_name"],
        "config_hash": canonical_sha256(frozen),
        "config_path": str(frozen_path),
        "config_sha256": file_sha256(frozen_path),
        "source_config_path": str(config_path),
        "source_config_sha256": file_sha256(config_path),
        "parent_marker_path": str(parent_marker_path),
        "parent_marker_sha256": file_sha256(parent_marker_path),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = protocol_dir / "protocol_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (protocol_dir / "PROTOCOL_FROZEN").write_text(
        f"status=frozen\nconfig_hash={manifest['config_hash']}\n"
        f"config_sha256={manifest['config_sha256']}\n"
        f"protocol_manifest_sha256={file_sha256(manifest_path)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
