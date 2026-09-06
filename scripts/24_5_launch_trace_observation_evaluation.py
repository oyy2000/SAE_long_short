#!/usr/bin/env python3
"""Evaluate every Phase-0 adapter on the locked GSM8K cohort."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import file_sha256, nonempty_line_count
from trace_length_observation.trace_observation import (
    protocol_hash,
    validate_trace_observation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    audit_path = _resolve(args.training_audit)
    output_dir = _resolve(args.output_dir)
    config = _read_json(config_path)
    validate_trace_observation_config(config)
    config_hash = protocol_hash(config)
    audit = _read_json(audit_path)
    if audit.get("status") != "passed" or audit.get("config_hash") != config_hash:
        raise ValueError("Training audit is incomplete or bound to another protocol.")
    training_marker = audit_path.parent / "TRAINING_COMPLETE"
    if not training_marker.is_file():
        raise FileNotFoundError(training_marker)
    evaluation = dict(config["evaluation"])
    prediction_dir = output_dir / "predictions"
    summary_dir = output_dir / "summaries"
    log_dir = output_dir / "logs"
    for path in (prediction_dir, summary_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "model_id": str(run["run_name"]),
            "run_name": str(run["run_name"]),
            "adapter_path": str(run["output_dir"]),
            "budget_regime": str(run["budget_regime"]),
            "loss_normalization": str(run["loss_normalization"]),
            "loss_mask": str(run["loss_mask"]),
            "length_rank": str(run["length_rank"]),
            "seed": int(run["seed"]),
            "training_metrics_path": str(run["training_metrics_path"]),
            "training_metrics_sha256": str(run["training_metrics_sha256"]),
        }
        for run in audit["runs"]
    ]
    if bool(evaluation["include_base_model"]):
        tasks.append(
            {
                "model_id": "base",
                "run_name": "base_qwen2p5_1p5b_instruct",
                "adapter_path": None,
                "budget_regime": "base",
                "loss_normalization": None,
                "loss_mask": None,
                "length_rank": None,
                "seed": None,
            }
        )
    if len(tasks) != int(evaluation["expected_run_count"]):
        raise ValueError("Evaluation task count differs from the registered matrix.")
    entries: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for task in tasks:
        model_id = str(task["model_id"])
        entry = {
            **task,
            "prediction_path": str(prediction_dir / f"{model_id}.jsonl"),
            "summary_path": str(summary_dir / f"{model_id}.json"),
            "log_path": str(log_dir / f"{model_id}.log"),
            "status": "prepared",
        }
        entries.append(entry)
        command = [
            sys.executable,
            "scripts/24_5_eval_trace_observation_model.py",
            "--config",
            str(config_path),
            "--model-name",
            str(config["student"]["model_name"]),
            "--model-revision",
            str(config["student"]["revision"]),
            "--cache-dir",
            str(config["student"]["cache_dir"]),
            "--split",
            str(evaluation["dataset_split"]),
            "--start-index",
            str(evaluation["start_index"]),
            "--limit",
            str(evaluation["limit"]),
            "--output-jsonl",
            entry["prediction_path"],
            "--summary-json",
            entry["summary_path"],
            "--max-new-tokens",
            str(evaluation["max_new_tokens"]),
            "--temperature",
            str(evaluation["temperature"]),
            "--top-p",
            str(evaluation["top_p"]),
            "--batch-size",
            str(evaluation["batch_size"]),
            "--torch-dtype",
            str(config["student"]["torch_dtype"]),
        ]
        if task["adapter_path"]:
            command.extend(["--adapter-path", str(task["adapter_path"])])
        commands.append(command)
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists() and not args.skip_complete:
        raise FileExistsError(f"Refusing to overwrite evaluation manifest: {manifest_path}")
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "training_audit_path": str(audit_path),
        "training_audit_sha256": file_sha256(audit_path),
        "evaluation_source_sha256": file_sha256(
            PROJECT_ROOT / "scripts/24_5_eval_trace_observation_model.py"
        ),
        "launcher_source_sha256": file_sha256(Path(__file__).resolve()),
        "run_count": len(entries),
        "runs": entries,
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    max_parallel = args.max_parallel or int(evaluation["max_parallel"])
    if not gpu_ids or max_parallel <= 0 or max_parallel > len(gpu_ids):
        raise ValueError("GPU IDs/max-parallel are inconsistent.")
    failures = _run_tasks(
        entries,
        commands,
        gpu_ids,
        max_parallel,
        int(evaluation["limit"]),
        manifest,
        manifest_path,
        skip_complete=args.skip_complete,
    )
    manifest["status"] = "failed" if failures else "complete"
    _write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Evaluation failures={len(failures)}")
    (output_dir / "EVALUATION_COMPLETE").write_text(
        "status=complete\n"
        f"config_hash={config_hash}\n"
        f"training_audit_sha256={file_sha256(audit_path)}\n"
        f"evaluation_manifest_sha256={file_sha256(manifest_path)}\n"
        f"run_count={len(entries)}\n",
        encoding="utf-8",
    )


def _run_tasks(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    expected_rows: int,
    manifest: Dict[str, Any],
    manifest_path: Path,
    *,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    available = list(gpu_ids)
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
    failures: List[Dict[str, Any]] = []
    expected_problem_ids: set[str] | None = None
    while pending or running:
        while pending and available and len(running) < max_parallel:
            entry, command = pending.pop(0)
            completed = _completed(entry, expected_rows, expected_problem_ids)
            if completed is not None:
                evidence, support = completed
                if not skip_complete:
                    raise FileExistsError(f"Completed evaluation already exists: {entry['model_id']}")
                expected_problem_ids = support if expected_problem_ids is None else expected_problem_ids
                entry.update(evidence)
                entry["status"] = "skipped_complete"
                _write_json(manifest_path, manifest)
                continue
            gpu_id = available.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            entry["status"] = "running"
            entry["gpu_id"] = gpu_id
            entry["pid"] = process.pid
            running.append((process, entry, log_handle, gpu_id))
            _write_json(manifest_path, manifest)
        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
        for process, entry, log_handle, gpu_id in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle, gpu_id))
                continue
            log_handle.close()
            available.append(gpu_id)
            entry["returncode"] = return_code
            completed = _completed(entry, expected_rows, expected_problem_ids) if return_code == 0 else None
            if completed is None:
                entry["status"] = "failed"
                failures.append(entry)
            else:
                evidence, support = completed
                if expected_problem_ids is None:
                    expected_problem_ids = support
                elif expected_problem_ids != support:
                    entry["status"] = "failed"
                    failures.append(entry)
                    continue
                entry.update(evidence)
                entry["status"] = "complete"
            _write_json(manifest_path, manifest)
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _completed(
    entry: Mapping[str, Any], expected_rows: int, expected_problem_ids: set[str] | None
) -> Tuple[Dict[str, Any], set[str]] | None:
    prediction_path = Path(str(entry["prediction_path"]))
    summary_path = Path(str(entry["summary_path"]))
    if not prediction_path.is_file() or not summary_path.is_file():
        return None
    if nonempty_line_count(prediction_path) != expected_rows:
        return None
    rows = _read_jsonl(prediction_path)
    support = {str(row["problem_id"]) for row in rows}
    if len(support) != expected_rows or (
        expected_problem_ids is not None and support != expected_problem_ids
    ):
        return None
    summary = _read_json(summary_path)
    if int(summary.get("n", -1)) != expected_rows:
        return None
    return (
        {
            "prediction_sha256": file_sha256(prediction_path),
            "summary_sha256": file_sha256(summary_path),
            "correct": int(summary["correct"]),
            "accuracy": float(summary["accuracy"]),
            "problem_ids_sha256": _problem_ids_hash(support),
        },
        support,
    )


def _problem_ids_hash(problem_ids: set[str]) -> str:
    import hashlib

    return hashlib.sha256(("\n".join(sorted(problem_ids)) + "\n").encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
