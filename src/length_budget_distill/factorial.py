"""Hashing, marker, runtime, and launcher helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_line_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_key_value_marker(path: str | Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            if not separator or not key:
                raise ValueError(f"Malformed marker line in {path}: {raw_line!r}")
            values[key] = value
    return values


def runtime_metadata(packages: Sequence[str] | None = None) -> Dict[str, Any]:
    names = list(
        packages or ("python", "torch", "transformers", "datasets", "peft", "numpy")
    )
    versions: Dict[str, str | None] = {}
    for name in names:
        if name == "python":
            versions[name] = platform.python_version()
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpus = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", maxsplit=3)]
            if len(fields) == 4:
                gpus.append(
                    dict(
                        zip(
                            ("index", "name", "memory_total_mib", "driver_version"),
                            fields,
                        )
                    )
                )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": versions,
        "gpus": gpus,
    }


def select_launcher_shard_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    launcher_shards: int,
    launcher_shard_index: int,
) -> list[Dict[str, Any]]:
    if launcher_shards <= 0 or not 0 <= launcher_shard_index < launcher_shards:
        raise ValueError("Invalid launcher shard topology.")
    copied = [dict(run) for run in runs]
    declared = ["launcher_shard_index" in run for run in copied]
    if any(declared) and not all(declared):
        raise ValueError("Dataset manifest mixes declared and implicit assignments.")
    if not all(declared):
        return [
            run
            for index, run in enumerate(copied)
            if index % launcher_shards == launcher_shard_index
        ]
    for run in copied:
        if int(run.get("launcher_shards", -1)) != launcher_shards:
            raise ValueError("Declared launcher topology mismatch.")
    return [
        run
        for run in copied
        if int(run["launcher_shard_index"]) == launcher_shard_index
    ]


def validated_adapter_evidence(path: str | Path) -> Dict[str, Any] | None:
    root = Path(path)
    marker_path = root / "TRAIN_COMPLETE"
    required = (root / "adapter_config.json", root / "adapter_model.safetensors")
    if not marker_path.is_file() or not all(path.is_file() for path in required):
        return None
    try:
        marker = read_key_value_marker(marker_path)
    except ValueError:
        return None
    actual = {
        "adapter_config_sha256": file_sha256(required[0]),
        "adapter_model_sha256": file_sha256(required[1]),
    }
    if any(marker.get(key) != value for key, value in actual.items()):
        return None
    return {**marker, **actual}
