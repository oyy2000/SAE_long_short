"""Hash-bound JSON artifact and gate helpers."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from .factorial import file_sha256, read_key_value_marker


def read_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return payload


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text_exclusive(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(text)


def require_passed_gate(
    project_root: Path, dependency: Mapping[str, Any]
) -> Dict[str, Any]:
    evidence = validated_gate_decision(
        project_root,
        gate_name=str(dependency["gate_name"]),
        decision_path=str(dependency["decision_path"]),
        completion_marker_path=str(dependency["completion_marker_path"]),
    )
    required = str(dependency.get("required_status", "passed"))
    if evidence["status"] != required:
        raise RuntimeError(
            f"{dependency['gate_name']} status is not {required}; dependent phase is blocked."
        )
    return evidence


def validated_gate_decision(
    project_root: Path,
    *,
    gate_name: str,
    decision_path: str | Path,
    completion_marker_path: str | Path,
) -> Dict[str, Any]:
    """Validate a passed or failed gate against its hash-bound completion marker."""

    decision_path = _resolve(project_root, str(decision_path))
    marker_path = _resolve(project_root, str(completion_marker_path))
    if not decision_path.is_file() or not marker_path.is_file():
        raise RuntimeError(
            f"{gate_name} evidence is incomplete; refusing to enter the dependent phase."
        )
    decision = read_json(decision_path)
    marker = read_key_value_marker(marker_path)
    status = str(decision.get("status"))
    if status not in {"passed", "failed"}:
        raise ValueError(f"Invalid {gate_name} decision status: {status}")
    if marker.get(f"{gate_name}_status") != status:
        raise ValueError(f"{gate_name} decision and completion marker disagree.")
    decision_sha256 = file_sha256(decision_path)
    if marker.get(f"{gate_name}_decision_sha256") != decision_sha256:
        raise ValueError(f"{gate_name} decision hash does not match its marker.")
    return {
        "gate_name": gate_name,
        "status": status,
        "decision_path": str(decision_path),
        "decision_sha256": decision_sha256,
        "completion_marker_path": str(marker_path),
        "completion_marker_sha256": file_sha256(marker_path),
    }


def publish_files_hash_verified(
    source: Path,
    destination: Path,
    filenames: tuple[str, ...],
    *,
    attempts: int = 10,
    wait_seconds: float = 5.0,
) -> Dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes: Dict[str, str] = {}
    for filename in filenames:
        source_path = source / filename
        expected = file_sha256(source_path)
        target = destination / filename
        partial = destination / f".{filename}.partial-{os.getpid()}"
        for attempt in range(1, attempts + 1):
            try:
                shutil.copyfile(source_path, partial)
                if file_sha256(partial) != expected:
                    raise OSError(f"Partial-copy hash mismatch: {partial}")
                os.replace(partial, target)
                if file_sha256(target) != expected:
                    raise OSError(f"Published hash mismatch: {target}")
                hashes[filename] = expected
                break
            except OSError:
                if partial.exists():
                    partial.unlink()
                if attempt == attempts:
                    raise
                time.sleep(wait_seconds)
    return hashes


def validated_training_artifacts(adapter_path: str | Path) -> Dict[str, Any]:
    """Validate a published LoRA marker and return its metrics plus file hashes."""

    root = Path(adapter_path)
    marker_path = root / "TRAIN_COMPLETE"
    marker = read_key_value_marker(marker_path)
    if marker.get("status") != "complete":
        raise ValueError(f"Training marker is incomplete: {marker_path}")
    paths = {
        "adapter_config": root / "adapter_config.json",
        "adapter_model": root / "adapter_model.safetensors",
        "training_metrics": root / "training_metrics.json",
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    for name, observed in hashes.items():
        expected = marker.get(f"{name}_sha256")
        if expected != observed:
            raise ValueError(f"Training marker hash mismatch: {paths[name]}")
    metrics = read_json(paths["training_metrics"])
    if metrics.get("status") != "complete":
        raise ValueError(
            f"Training metrics are incomplete: {paths['training_metrics']}"
        )
    return {
        "root": str(root),
        "marker_path": str(marker_path),
        "marker_sha256": file_sha256(marker_path),
        "marker": marker,
        "hashes": hashes,
        "metrics": metrics,
    }


def validated_artifact_marker(
    marker_path: str | Path,
    *,
    expected_status: str,
    hash_bindings: Mapping[str, str | Path],
) -> Dict[str, Any]:
    """Validate status and artifact hashes recorded by a completion marker."""

    path = Path(marker_path)
    marker = read_key_value_marker(path)
    if marker.get("status") != expected_status:
        raise ValueError(f"Unexpected marker status: {path}")
    artifacts = {}
    for field, artifact_path in hash_bindings.items():
        artifact = Path(artifact_path)
        observed = file_sha256(artifact)
        if marker.get(field) != observed:
            raise ValueError(f"Marker hash mismatch for {artifact}: {path}")
        artifacts[field] = {"path": str(artifact), "sha256": observed}
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": expected_status,
        "artifacts": artifacts,
    }


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
