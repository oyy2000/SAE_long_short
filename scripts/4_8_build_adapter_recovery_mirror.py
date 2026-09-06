#!/usr/bin/env python3
"""Build a hash-equivalent NFS mirror from node-local LoRA training outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


FILES = ("adapter_config.json", "adapter_model.safetensors", "training_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--run-config-root", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--destination-root", required=True)
    args = parser.parse_args()
    runtime_root = Path(args.runtime_root)
    run_config_root = Path(args.run_config_root)
    canonical_root = Path(args.canonical_root)
    destination_root = Path(args.destination_root)
    _mkdir_with_retry(destination_root, parents=True, exist_ok=True)
    completed = []
    for source in sorted(path for path in runtime_root.iterdir() if path.is_dir()):
        if not all((source / filename).is_file() for filename in FILES):
            continue
        model_id = source.name
        run_config_path = run_config_root / f"{model_id}.json"
        run = json.loads(run_config_path.read_text(encoding="utf-8"))
        destination = destination_root / model_id
        if destination.exists():
            _retry_remote_operation(lambda: _validate(destination))
            completed.append({"model_id": model_id, "status": "already_valid"})
            continue
        temporary = destination_root / f".{model_id}.partial-{os.getpid()}"
        _mkdir_with_retry(temporary, parents=False, exist_ok=False)
        for filename in FILES:
            _copy_with_retry(source / filename, temporary / filename)
        hashes = {filename: _sha256(temporary / filename) for filename in FILES}
        condition = run["condition"]
        marker_text = (
            "status=complete\n"
            f"run_name={run['run_name']}\n"
            f"seed={condition['seed']}\n"
            f"budget_regime={condition['budget_regime']}\n"
            f"intervention_condition={condition['intervention_condition']}\n"
            f"config_hash={run['config_hash']}\n"
            f"train_sha256={run['data']['train_sha256']}\n"
            f"run_config_sha256={_sha256(run_config_path)}\n"
            f"training_source_sha256={run['source_hashes']['training']}\n"
            f"launcher_source_sha256={run['source_hashes']['launcher']}\n"
            f"entrypoint_source_sha256={run['source_hashes']['entrypoint']}\n"
            f"adapter_config_sha256={hashes['adapter_config.json']}\n"
            f"adapter_model_sha256={hashes['adapter_model.safetensors']}\n"
            f"training_metrics_sha256={hashes['training_metrics.json']}\n"
            "formal_claim_allowed=false\n"
        )
        _write_text_with_retry(temporary / "TRAIN_COMPLETE", marker_text)
        provenance = {
            "status": "hash_equivalent_recovery_copy",
            "model_id": model_id,
            "node_local_source": str(source),
            "canonical_source": str(canonical_root / model_id),
            "run_config_path": str(run_config_path),
            "file_hashes": hashes,
            "marker_sha256": _sha256(temporary / "TRAIN_COMPLETE"),
            "content_changed": False,
        }
        _write_text_with_retry(
            temporary / "RECOVERY_PROVENANCE.json",
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        )
        _retry_remote_operation(lambda: _validate(temporary))
        _retry_remote_operation(lambda: os.replace(temporary, destination))
        completed.append({"model_id": model_id, "status": "mirrored", **hashes})
    print(json.dumps({"status": "complete", "models": completed}, indent=2), flush=True)


def _validate(root: Path) -> None:
    marker = _read_marker(root / "TRAIN_COMPLETE")
    if marker.get("status") != "complete":
        raise ValueError(f"Incomplete recovery marker: {root}")
    for field, filename in (
        ("adapter_config_sha256", "adapter_config.json"),
        ("adapter_model_sha256", "adapter_model.safetensors"),
        ("training_metrics_sha256", "training_metrics.json"),
    ):
        if marker.get(field) != _sha256(root / filename):
            raise ValueError(f"Recovery hash mismatch: {root / filename}")


def _read_marker(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            output[key] = value
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_retry(source: Path, destination: Path) -> None:
    expected = _sha256(source)
    for attempt in range(1, 121):
        attempt_path = destination.with_name(
            f".{destination.name}.attempt-{os.getpid()}-{attempt}"
        )
        try:
            with source.open("rb") as input_handle, attempt_path.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            if _sha256(attempt_path) != expected:
                raise OSError(f"Recovery copy hash mismatch: {attempt_path}")
            _retry_remote_operation(lambda: os.replace(attempt_path, destination))
            return
        except OSError:
            try:
                attempt_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == 120:
                raise
            time.sleep(5)


def _write_text_with_retry(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    for attempt in range(1, 121):
        attempt_path = path.with_name(f".{path.name}.attempt-{os.getpid()}-{attempt}")
        try:
            with attempt_path.open("xb") as handle:
                handle.write(payload)
            _retry_remote_operation(lambda: os.replace(attempt_path, path))
            return
        except OSError:
            try:
                attempt_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == 120:
                raise
            time.sleep(5)


def _mkdir_with_retry(path: Path, *, parents: bool, exist_ok: bool) -> None:
    _retry_remote_operation(lambda: path.mkdir(parents=parents, exist_ok=exist_ok))


def _retry_remote_operation(function):
    for attempt in range(1, 121):
        try:
            return function()
        except OSError:
            if attempt == 120:
                raise
            time.sleep(5)
    raise AssertionError("Unreachable remote operation retry state")


if __name__ == "__main__":
    main()
