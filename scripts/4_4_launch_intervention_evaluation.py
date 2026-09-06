#!/usr/bin/env python3
"""Launch a disjoint shard of locked intervention-student evaluations."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--launcher-shards", type=int, required=True)
    parser.add_argument("--launcher-shard-index", type=int, required=True)
    parser.add_argument("--model-id-filter", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    checkpoint_root, output_root = _resolve(args.checkpoint_root), _resolve(args.output_root)
    recovery_root_value = os.environ.get("SAE_ADAPTER_RECOVERY_ROOT")
    recovery_root = Path(recovery_root_value) if recovery_root_value else None
    tasks = []
    for budget in config["student_sft"]["budget_regimes"]:
        for condition in config["main_generation"]["conditions"]:
            for seed in config["student_sft"]["seeds"]:
                model_id = f"{budget}__{condition}__seed_{seed}"
                adapter = checkpoint_root / model_id
                admission_adapter = (
                    recovery_root / model_id
                    if recovery_root is not None and (recovery_root / model_id).is_dir()
                    else adapter
                )
                # The child stages and hash-validates each adapter before use.
                # Here only require the registered completion marker so that a
                # degraded BeeGFS does not force 24 redundant full-file hashes.
                if not (admission_adapter / "TRAIN_COMPLETE").is_file():
                    raise FileNotFoundError(admission_adapter / "TRAIN_COMPLETE")
                tasks.append({"model_id": model_id, "adapter_path": str(adapter)})
    if config["evaluation"].get("include_base_student", False):
        tasks.append({"model_id": "base_student", "adapter_path": None})
    tasks = sorted(tasks, key=lambda row: row["model_id"])
    if args.model_id_filter:
        requested = {value for value in args.model_id_filter.split(",") if value}
        tasks = [task for task in tasks if task["model_id"] in requested]
        observed = {task["model_id"] for task in tasks}
        if observed != requested:
            raise ValueError(f"Unknown model-id filter entries: {sorted(requested - observed)}")
    selected = [
        task for index, task in enumerate(tasks)
        if index % args.launcher_shards == args.launcher_shard_index
    ]
    gpu_ids = [part for part in args.gpu_ids.split(",") if part]
    if not selected or not gpu_ids:
        raise ValueError("Evaluation launcher has no tasks or GPUs.")
    _mkdir_remote_with_retry(output_root)
    log_dir = output_root / "logs"
    _mkdir_remote_with_retry(log_dir)
    publish_manifest_path = output_root / f"evaluation_launcher_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    manifest_stage_root_value = os.environ.get("SAE_EVAL_LAUNCHER_MANIFEST_STAGE_ROOT")
    if manifest_stage_root_value:
        manifest_stage_root = Path(manifest_stage_root_value)
        manifest_stage_root.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_stage_root / publish_manifest_path.name
    else:
        manifest_path = publish_manifest_path
    entries = [
        {
            **task,
            "output_dir": str(output_root / task["model_id"]),
            "log_path": str(log_dir / f"{task['model_id']}.log"),
            "status": "prepared",
        }
        for task in selected
    ]
    manifest = {
        "status": "running",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "model_id_filter": args.model_id_filter,
        "evaluation_source_sha256": file_sha256(PROJECT_ROOT / "scripts/4_3_eval_intervention_student.py"),
        "launcher_source_sha256": file_sha256(Path(__file__).resolve()),
        "runs": entries,
    }
    _write_replace(manifest_path, manifest)
    failures = _run(entries, gpu_ids, config_path, manifest, manifest_path, args.skip_complete)
    manifest["status"] = "failed" if failures else "complete"
    _write_replace(manifest_path, manifest)
    if manifest_path != publish_manifest_path:
        _publish_required(manifest_path, publish_manifest_path)
    if failures:
        raise SystemExit(f"Evaluation failures={failures}")


def _run(entries: list[dict[str, Any]], gpu_ids: list[str], config_path: Path, manifest: dict[str, Any], manifest_path: Path, skip_complete: bool) -> int:
    pending, available, running, failures = list(entries), list(gpu_ids), [], 0
    runtime_log_root_value = os.environ.get("SAE_EVAL_LAUNCHER_LOG_STAGE_ROOT")
    runtime_log_root = Path(runtime_log_root_value) if runtime_log_root_value else None
    if runtime_log_root is not None:
        runtime_log_root.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and available:
            entry = pending.pop(0)
            if _completed_evaluation(entry):
                if not skip_complete:
                    raise FileExistsError(f"Completed evaluation exists: {entry['model_id']}")
                entry["status"] = "skipped_complete"
                _write_replace(manifest_path, manifest)
                continue
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            command = [
                sys.executable, "scripts/4_3_eval_intervention_student.py",
                "--config", str(config_path), "--model-id", entry["model_id"],
                "--output-dir", entry["output_dir"],
            ]
            if entry["adapter_path"]:
                command.extend(["--adapter-path", entry["adapter_path"]])
            runtime_log_path = (
                runtime_log_root / f"{entry['model_id']}.log"
                if runtime_log_root is not None
                else Path(entry["log_path"])
            )
            log = runtime_log_path.open("w", encoding="utf-8")
            process = _popen_with_remote_io_retry(command, env, log)
            entry.update({"status": "running", "gpu_id": gpu_id, "pid": process.pid})
            running.append((process, entry, log, gpu_id, runtime_log_path))
            _write_replace(manifest_path, manifest)
        next_running = []
        for process, entry, log, gpu_id, runtime_log_path in running:
            code = process.poll()
            if code is None:
                next_running.append((process, entry, log, gpu_id, runtime_log_path))
                continue
            log.close()
            if runtime_log_path != Path(entry["log_path"]):
                entry["log_publish_status"] = _publish_log(
                    runtime_log_path, Path(entry["log_path"])
                )
            available.append(gpu_id)
            entry["returncode"] = code
            completed = _completed_evaluation(entry)
            entry["status"] = "complete" if completed else "failed"
            failures += int(not completed)
            _write_replace(manifest_path, manifest)
        running = next_running
        if running:
            time.sleep(5)
    return failures


def _completed_evaluation(entry: dict[str, Any]) -> bool:
    root = Path(entry["output_dir"])
    marker_path = root / "EVALUATION_COMPLETE"
    manifest_path = root / "evaluation_manifest.json"
    marker_manifest_valid = False
    for attempt in range(1, 11):
        try:
            marker_path.stat()
            manifest_path.stat()
            marker = read_key_value_marker(marker_path)
            marker_manifest_valid = (
                marker.get("status") == "complete"
                and marker.get("manifest_sha256") == file_sha256(manifest_path)
            )
            if not marker_manifest_valid:
                return False
            full_valid = (
                marker.get("predictions_sha256") == file_sha256(root / "predictions.jsonl")
                and marker.get("summary_sha256") == file_sha256(root / "summary.json")
            )
            if full_valid:
                entry["completion_validation"] = "full_hash"
            return full_valid
        except FileNotFoundError:
            return False
        except ValueError:
            return False
        except OSError as exc:
            if exc.errno != errno.EREMOTEIO:
                return False
            if attempt == 10 and marker_manifest_valid:
                entry["completion_validation"] = "marker_manifest_only_due_remote_io"
                return True
            time.sleep(3)
    return False


def _popen_with_remote_io_retry(command: list[str], env: dict[str, str], log: Any) -> subprocess.Popen[Any]:
    for attempt in range(1, 61):
        try:
            return subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            if exc.errno != errno.EREMOTEIO or attempt == 60:
                log.close()
                raise
            log.write(f"Transient BeeGFS remote I/O during launch; retry={attempt}/60\n")
            log.flush()
            time.sleep(3)
    raise AssertionError("Unreachable launch retry state.")


def _write_replace(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    last_error: OSError | None = None
    for attempt in range(1, 61):
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{attempt}")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if exc.errno != errno.EREMOTEIO or attempt == 60:
                raise
            time.sleep(3)
    raise last_error or OSError("Failed to publish launcher manifest")


def _publish_log(source: Path, destination: Path) -> str:
    """Best-effort publication of a non-evidence child log from node-local storage."""
    try:
        _mkdir_remote_with_retry(destination.parent)
    except OSError as exc:
        return f"not_published_parent_errno_{exc.errno}"
    for attempt in range(1, 61):
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}-{attempt}")
        try:
            temporary.write_bytes(source.read_bytes())
            os.replace(temporary, destination)
            return "published"
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if exc.errno != errno.EREMOTEIO:
                return f"not_published_errno_{exc.errno}"
            time.sleep(3)
    return "not_published_remote_io"


def _publish_required(source: Path, destination: Path) -> None:
    """Hash-verify publication of the final launcher manifest."""
    expected = file_sha256(source)
    for attempt in range(1, 121):
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}-{attempt}")
        try:
            temporary.write_bytes(source.read_bytes())
            if file_sha256(temporary) != expected:
                raise OSError("Launcher manifest partial-copy hash mismatch")
            os.replace(temporary, destination)
            if file_sha256(destination) != expected:
                raise OSError("Launcher manifest published hash mismatch")
            return
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == 120:
                raise
            time.sleep(5)


def _mkdir_remote_with_retry(path: Path) -> None:
    for attempt in range(1, 61):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except OSError as exc:
            if exc.errno != errno.EREMOTEIO or attempt == 60:
                raise
            time.sleep(3)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
