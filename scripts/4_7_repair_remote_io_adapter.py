#!/usr/bin/env python3
"""Repair an unreadable canonical LoRA inode from a hash-identical local copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--backup-suffix", required=True)
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    canonical_dir = Path(args.canonical_dir)
    source = source_dir / "adapter_model.safetensors"
    canonical = canonical_dir / "adapter_model.safetensors"
    local_marker = source_dir / "TRAIN_COMPLETE"
    marker_path = local_marker if local_marker.is_file() else canonical_dir / "TRAIN_COMPLETE"
    source_marker = _read_marker(marker_path)
    expected = source_marker.get("adapter_model_sha256")
    if not expected:
        raise ValueError(f"TRAIN_COMPLETE has no adapter_model_sha256: {marker_path}")
    observed_source = _sha256(source)
    if observed_source != expected:
        raise ValueError(f"Local source hash mismatch: {source}")
    backup = canonical_dir / f"adapter_model.safetensors.{args.backup_suffix}"
    partial = canonical_dir / f".adapter_model.safetensors.repair-{os.getpid()}"
    if backup.exists() or partial.exists():
        raise FileExistsError(backup if backup.exists() else partial)
    shutil.copyfile(source, partial)
    if _sha256(partial) != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Repair partial hash mismatch: {partial}")
    os.replace(canonical, backup)
    try:
        os.replace(partial, canonical)
        if _sha256(canonical) != expected:
            raise ValueError(f"Repaired canonical hash mismatch: {canonical}")
    except Exception:
        if not canonical.exists() and backup.exists():
            os.replace(backup, canonical)
        raise
    print(
        json.dumps(
            {
                "status": "repaired",
                "canonical": str(canonical),
                "sha256": expected,
                "unreadable_inode_backup": str(backup),
            },
            indent=2,
        ),
        flush=True,
    )


def _read_marker(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
