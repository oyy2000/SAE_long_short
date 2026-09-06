#!/usr/bin/env python3
"""Freeze the Phase-1 CTV protocol after its entry decision is authorized."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    require_passed_gate,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument(
        "--output-dir", default="results/phase1_teaching_utility_v1/formal/protocol"
    )
    args = parser.parse_args()
    source = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(source)
    entry = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = output_dir / "frozen_protocol.json"
    write_json_exclusive(frozen, config)
    manifest = {
        "status": "frozen",
        "config_hash": canonical_sha256(config),
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "frozen_path": str(frozen),
        "frozen_sha256": file_sha256(frozen),
        "entry_evidence": entry,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "protocol_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "PROTOCOL_FROZEN").write_text(
        f"status=frozen\nconfig_hash={manifest['config_hash']}\nfrozen_sha256={manifest['frozen_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
