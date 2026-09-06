#!/usr/bin/env python3
"""Build a hash-equivalent mirror of completed evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


FILES = ("predictions.jsonl", "summary.json", "evaluation_manifest.json", "EVALUATION_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--replacement",
        action="append",
        default=[],
        help="Exact replacement source as model_id/filename=/absolute/path.",
    )
    args = parser.parse_args()
    evaluation_root = Path(args.evaluation_root)
    destination_root = Path(args.destination_root)
    replacements = _parse_replacements(args.replacement)
    _retry(lambda: destination_root.mkdir(parents=True, exist_ok=True))
    model_dirs = sorted(
        path for path in evaluation_root.iterdir()
        if path.is_dir() and path.name != "logs"
    )
    if len(model_dirs) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} completed evaluations, found {len(model_dirs)}")
    rows = []
    for source_root in model_dirs:
        model_id = source_root.name
        destination = destination_root / model_id
        if destination.exists():
            _retry(lambda: _validate(destination))
            rows.append({"model_id": model_id, "status": "already_valid"})
            continue
        temporary = destination_root / f".{model_id}.partial-{os.getpid()}"
        _retry(lambda: temporary.mkdir(parents=False, exist_ok=False))
        used_replacements = []
        for filename in FILES:
            replacement = replacements.get((model_id, filename))
            source = replacement if replacement is not None else source_root / filename
            _copy_with_retry(source, temporary / filename)
            if replacement is not None:
                used_replacements.append({"filename": filename, "source": str(replacement)})
        hashes = {filename: _sha256(temporary / filename) for filename in FILES}
        provenance = {
            "status": "hash_equivalent_evaluation_recovery_copy",
            "model_id": model_id,
            "canonical_source": str(source_root),
            "file_hashes": hashes,
            "replacements": used_replacements,
            "content_changed": False,
        }
        _write_text_with_retry(
            temporary / "RECOVERY_PROVENANCE.json",
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        )
        _retry(lambda: _validate(temporary))
        _retry(lambda: os.replace(temporary, destination))
        rows.append({"model_id": model_id, "status": "mirrored", "replacements": used_replacements})
    print(json.dumps({"status": "complete", "models": rows}, indent=2), flush=True)


def _validate(root: Path) -> None:
    marker = _read_marker(root / "EVALUATION_COMPLETE")
    if marker.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation recovery marker: {root}")
    bindings = {
        "predictions_sha256": "predictions.jsonl",
        "summary_sha256": "summary.json",
        "manifest_sha256": "evaluation_manifest.json",
    }
    for field, filename in bindings.items():
        if marker.get(field) != _sha256(root / filename):
            raise ValueError(f"Evaluation recovery hash mismatch: {root / filename}")


def _parse_replacements(values: list[str]) -> dict[tuple[str, str], Path]:
    output = {}
    for value in values:
        key, raw_path = value.split("=", 1)
        model_id, filename = key.split("/", 1)
        path = Path(raw_path)
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(path)
        output[(model_id, filename)] = path
    return output


def _copy_with_retry(source: Path, destination: Path) -> None:
    expected = _retry(lambda: _sha256(source))
    for attempt in range(1, 121):
        attempt_path = destination.with_name(f".{destination.name}.attempt-{os.getpid()}-{attempt}")
        try:
            with source.open("rb") as input_handle, attempt_path.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            if _sha256(attempt_path) != expected:
                raise OSError(f"Evaluation recovery copy hash mismatch: {attempt_path}")
            _retry(lambda: os.replace(attempt_path, destination))
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
            _retry(lambda: os.replace(attempt_path, path))
            return
        except OSError:
            try:
                attempt_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == 120:
                raise
            time.sleep(5)


def _read_marker(path: Path) -> dict[str, str]:
    output = {}
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


def _retry(function):
    for attempt in range(1, 121):
        try:
            return function()
        except OSError:
            if attempt == 120:
                raise
            time.sleep(5)
    raise AssertionError("Unreachable evaluation recovery retry state")


if __name__ == "__main__":
    main()
