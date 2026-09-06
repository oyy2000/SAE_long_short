#!/usr/bin/env python3
"""Freeze the researcher-authorized exploratory SAE pilot protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
    runtime_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_sae_pilot_v1.json")
    parser.add_argument(
        "--output-dir", default="results/phase2_sae_pilot_v1/formal/protocol"
    )
    args = parser.parse_args()
    source = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(source)
    if config.get("formal_claim_allowed") is not False:
        raise ValueError("This bypassed-gate pilot must remain exploratory.")
    pause_marker_path = (
        PROJECT_ROOT / "results/trace_length_observation_gate0_v1/PHASE0_PAUSED"
    )
    pause_marker = read_key_value_marker(pause_marker_path)
    if (
        pause_marker.get("status") != "paused"
        or pause_marker.get("gate0_status") != "not_evaluated"
    ):
        raise ValueError(
            "The registered Phase-0 pause state is missing or inconsistent."
        )
    source_pool = config["source_pool"]
    audit_path = Path(source_pool["generation_audit_path"])
    if file_sha256(audit_path) != source_pool["generation_audit_sha256"]:
        raise ValueError("Parent generation audit hash changed.")
    for shard in source_pool["raw_shards"]:
        if file_sha256(shard["path"]) != shard["sha256"]:
            raise ValueError(f"Parent raw shard hash changed: {shard['path']}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = output_dir / "frozen_protocol.json"
    write_json_exclusive(frozen, config)
    manifest = {
        "status": "frozen",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "frozen_path": str(frozen),
        "frozen_sha256": file_sha256(frozen),
        "phase0_pause_marker_path": str(pause_marker_path),
        "phase0_pause_marker_sha256": file_sha256(pause_marker_path),
        "phase0_pause_state": pause_marker,
        "runtime": runtime_metadata(("python", "torch", "transformers", "safetensors")),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "protocol_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "PROTOCOL_FROZEN").write_text(
        f"status=frozen\nconfig_hash={manifest['config_hash']}\n"
        f"frozen_sha256={manifest['frozen_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
