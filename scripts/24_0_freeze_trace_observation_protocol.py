#!/usr/bin/env python3
"""Freeze the parent-hash-bound Phase-0 trace observation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import canonical_sha256, file_sha256
from trace_length_observation.trace_observation import validate_trace_observation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/trace_length_observation_gate0_v1.json"
    )
    parser.add_argument(
        "--output-dir",
        default="results/trace_length_observation_gate0_v1/formal/protocol",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    frozen_path = output_dir / "frozen_protocol.json"
    marker_path = output_dir / "PROTOCOL_FROZEN"
    if frozen_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol evidence: {output_dir}")
    config = _read_json(config_path)
    validate_trace_observation_config(config)
    _validate_parent(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    with frozen_path.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    config_hash = canonical_sha256(config)
    marker_path.write_text(
        "status=frozen\n"
        f"config_hash={config_hash}\n"
        f"config_file_sha256={file_sha256(config_path)}\n"
        f"frozen_protocol_sha256={file_sha256(frozen_path)}\n"
        f"parent_generation_marker_sha256={config['parent_generation']['completion_marker_sha256']}\n"
        f"expected_training_runs={config['training']['expected_run_count']}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "config_hash": config_hash,
                "frozen_protocol": str(frozen_path),
                "marker": str(marker_path),
            },
            indent=2,
        )
    )


def _validate_parent(config: Dict[str, Any]) -> None:
    parent = dict(config["parent_generation"])
    checks = (
        ("config_path", "config_file_sha256"),
        ("completion_marker_path", "completion_marker_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("generation_audit_path", "generation_audit_sha256"),
    )
    for path_key, hash_key in checks:
        path = _resolve(str(parent[path_key]))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != str(parent[hash_key]):
            raise ValueError(
                f"Parent hash mismatch for {path_key}: expected={parent[hash_key]} actual={actual}"
            )
    parent_config = _read_json(_resolve(str(parent["config_path"])))
    actual_canonical = canonical_sha256(parent_config)
    if actual_canonical != str(parent["canonical_config_sha256"]):
        raise ValueError("Parent canonical config hash mismatch.")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
