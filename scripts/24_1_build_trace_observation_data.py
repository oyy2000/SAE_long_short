#!/usr/bin/env python3
"""Re-select quantile-band traces and build all Phase-0 SFT schedules."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for source_root in (SRC_ROOT,):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.student_prompts import build_student_math_prompt
from trace_length_observation.trace_observation import (
    LENGTH_RANKS,
    answer_character_span,
    answer_span_format,
    build_run_matrix,
    completion_token_count,
    materialize_budget_schedules,
    protocol_hash,
    select_by_problem,
    summarize_selected_rows,
    validate_trace_observation_config,
)
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/trace_length_observation_gate0_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--limit-problems", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--io-attempts", type=int, default=5)
    parser.add_argument("--io-wait-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    config = _read_json_retry(config_path, args.io_attempts, args.io_wait_seconds)
    validate_trace_observation_config(config)
    config_hash = protocol_hash(config)
    output_root = _resolve(str(config["outputs"]["result_root"]))
    output_dir = _resolve(args.output_dir) if args.output_dir else output_root / args.stage / "data"
    marker_path = output_dir / "DATA_COMPLETE"
    if output_dir.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-0 data evidence: {output_dir}")
    if args.stage == "formal":
        _require_frozen_protocol(config_path, config_hash)
    parent_evidence, raw_rows = _load_parent_rows(
        config, attempts=args.io_attempts, wait_seconds=args.io_wait_seconds
    )
    tokenizer = _load_tokenizer(config)
    reverified_rows = _retokenize_rows(_reverify_rows(raw_rows, config), tokenizer)
    selected, dropped = select_by_problem(
        reverified_rows,
        minimum_unique_correct=int(config["candidate_pool"]["minimum_unique_correct"]),
        tail_fraction=float(config["candidate_pool"]["tail_fraction"]),
    )
    if args.limit_problems is not None:
        if args.limit_problems <= 0:
            raise ValueError("--limit-problems must be positive.")
        keep = sorted(selected)[: args.limit_problems]
        selected = {problem_id: selected[problem_id] for problem_id in keep}
    if not selected:
        raise ValueError("No problems remain after Phase-0 candidate filtering.")
    answer_span_formats: Dict[str, int] = {}
    for choices in selected.values():
        for row in choices.values():
            answer_character_span(str(row["solution"]))
            span_format = answer_span_format(str(row["solution"]))
            answer_span_formats[span_format] = answer_span_formats.get(span_format, 0) + 1
    if args.stage == "formal" and len(selected) + len(dropped) != int(
        config["dataset"]["expected_parent_problem_count"]
    ):
        raise ValueError("Formal selection did not account for every parent problem.")

    schedules, budget_summary = materialize_budget_schedules(
        selected,
        tokenizer=tokenizer,
        seed=int(config["training"]["budget_schedule_seed"]),
        max_token_gap=int(config["training"]["maximum_token_budget_gap"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = output_dir / "selected_traces.jsonl"
    selected_rows = [
        selected[problem_id][rank]
        for problem_id in sorted(selected)
        for rank in LENGTH_RANKS
    ]
    write_jsonl(selected_path, selected_rows)
    schedule_evidence: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (regime, rank), rows in sorted(schedules.items()):
        path = output_dir / "sft" / regime / f"{rank}.jsonl"
        serialized = [_training_row(row, regime, rank) for row in rows]
        write_jsonl(path, serialized)
        schedule_key = f"{regime}__{rank}"
        summary = dict(budget_summary["schedules"][schedule_key])
        schedule_evidence[(regime, rank)] = {
            "train_path": str(path),
            "train_sha256": file_sha256(path),
            "record_count": len(serialized),
            "unique_problem_count": summary["unique_problem_count"],
            "budget_token_basis": summary["budget_token_basis"],
            "target_budget_tokens": summary["target_budget_tokens"],
            "actual_budget_tokens": summary["actual_budget_tokens"],
            "actual_completion_tokens": summary["actual_completion_tokens"],
            "actual_model_input_tokens": summary["actual_model_input_tokens"],
            "token_gap": summary["token_gap"],
            "duplicate_exposures": summary["duplicate_exposures"],
        }
    runs = build_run_matrix(config, schedule_evidence)
    launcher_shards = int(config["training"]["launcher_shards"])
    for index, run in enumerate(runs):
        run["launcher_shards"] = launcher_shards
        run["launcher_shard_index"] = index % launcher_shards
    expected_runs = int(config["training"]["expected_run_count"])
    if len(runs) != expected_runs:
        raise ValueError(f"Run count mismatch: expected={expected_runs} actual={len(runs)}")

    source_hashes = {
        "selection": file_sha256(SRC_ROOT / "trace_length_observation/trace_observation.py"),
        "student_prompts": file_sha256(
            SRC_ROOT / "length_budget_distill/student_prompts.py"
        ),
        "entrypoint": file_sha256(Path(__file__).resolve()),
    }
    selection_audit_path = output_dir / "selection_audit.json"
    selection_audit = {
        "status": "passed",
        "stage": args.stage,
        "config_hash": config_hash,
        "selection_method": config["candidate_pool"]["selection_method"],
        "minimum_unique_correct": config["candidate_pool"]["minimum_unique_correct"],
        "tail_fraction": config["candidate_pool"]["tail_fraction"],
        "source_problem_count": len({str(row["problem_id"]) for row in reverified_rows}),
        "selected_problem_count": len(selected),
        "dropped_problem_count": len(dropped),
        "dropped_problem_ids": dropped,
        "source_raw_record_count": len(reverified_rows),
        "selected_record_count": len(selected_rows),
        "answer_maskable_record_count": sum(answer_span_formats.values()),
        "answer_span_format_counts": answer_span_formats,
        "selection_summary": summarize_selected_rows(selected),
        "selected_traces_path": str(selected_path),
        "selected_traces_sha256": file_sha256(selected_path),
        "parent_evidence": parent_evidence,
        "source_hashes": source_hashes,
    }
    _write_json_exclusive(selection_audit_path, selection_audit)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = {
        "status": "complete",
        "stage": args.stage,
        "experiment_name": config["experiment_name"],
        "protocol_variant": config["protocol_variant"],
        "evidence_level": "pipeline_smoke_only" if args.stage == "smoke" else config["evidence_level"],
        "formal_claim_allowed": False,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "selection_audit_path": str(selection_audit_path),
        "selection_audit_sha256": file_sha256(selection_audit_path),
        "selected_problem_count": len(selected),
        "dropped_problem_count": len(dropped),
        "budget_summary": budget_summary,
        "source_hashes": source_hashes,
        "run_count": len(runs),
        "runs": runs,
    }
    _write_json_exclusive(manifest_path, manifest)
    marker_path.write_text(
        "status=complete\n"
        f"stage={args.stage}\n"
        f"config_hash={config_hash}\n"
        f"dataset_manifest_sha256={file_sha256(manifest_path)}\n"
        f"selection_audit_sha256={file_sha256(selection_audit_path)}\n"
        f"selected_problem_count={len(selected)}\n"
        f"run_count={len(runs)}\n",
        encoding="utf-8",
    )
    logging.info(
        "phase0_data_complete stage=%s problems=%d runs=%d output=%s",
        args.stage,
        len(selected),
        len(runs),
        output_dir,
    )


def _load_parent_rows(
    config: Mapping[str, Any], *, attempts: int, wait_seconds: float
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    parent = dict(config["parent_generation"])
    parent_config_path = _resolve(str(parent["config_path"]))
    parent_project_root = parent_config_path.parent.parent

    def resolve_parent_artifact(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else parent_project_root / path

    registered = (
        ("config_path", "config_file_sha256"),
        ("completion_marker_path", "completion_marker_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("generation_audit_path", "generation_audit_sha256"),
    )
    evidence: Dict[str, Any] = {}
    for path_key, hash_key in registered:
        path = _resolve(str(parent[path_key]))
        actual = _file_sha256_retry(path, attempts, wait_seconds)
        if actual != str(parent[hash_key]):
            raise ValueError(f"Parent hash mismatch for {path_key}: {path}")
        evidence[path_key] = {"path": str(path), "sha256": actual}
    parent_config = _read_json_retry(
        _resolve(str(parent["config_path"])), attempts, wait_seconds
    )
    if canonical_sha256(parent_config) != str(parent["canonical_config_sha256"]):
        raise ValueError("Parent canonical config hash mismatch.")
    audit_path = _resolve(str(parent["generation_audit_path"]))
    audit = _read_json_retry(audit_path, attempts, wait_seconds)
    if audit.get("status") != "passed" or int(audit.get("num_candidates", 0)) != 16:
        raise ValueError("Parent generation audit is not the registered complete 16-rollout pool.")
    all_rows: List[Dict[str, Any]] = []
    shard_evidence: List[Dict[str, Any]] = []
    for item in audit.get("input_manifests", []):
        manifest_path = resolve_parent_artifact(str(item["path"]))
        manifest_hash = _file_sha256_retry(manifest_path, attempts, wait_seconds)
        if manifest_hash != str(item["sha256"]):
            raise ValueError(f"Parent shard-manifest hash mismatch: {manifest_path}")
        shard_manifest = _read_json_retry(manifest_path, attempts, wait_seconds)
        raw = dict(shard_manifest["raw"])
        raw_path = resolve_parent_artifact(str(raw["path"]))
        raw_hash = _file_sha256_retry(raw_path, attempts, wait_seconds)
        if raw_hash != str(raw["sha256"]):
            raise ValueError(f"Parent raw-shard hash mismatch: {raw_path}")
        rows = _read_jsonl_retry(raw_path, attempts, wait_seconds)
        if len(rows) != int(raw["record_count"]):
            raise ValueError(f"Parent raw-shard row-count mismatch: {raw_path}")
        all_rows.extend(rows)
        shard_evidence.append(
            {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "raw_path": str(raw_path),
                "raw_sha256": raw_hash,
                "record_count": len(rows),
            }
        )
    if len(all_rows) != int(audit["expected_raw_record_count"]):
        raise ValueError("Parent raw pool cardinality differs from its generation audit.")
    if len({str(row["trace_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Parent raw pool contains duplicate trace IDs.")
    evidence["raw_shards"] = shard_evidence
    return evidence, all_rows


def _reverify_rows(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    expected_teacher = str(config["teacher"]["model_name"])
    reverified: List[Dict[str, Any]] = []
    mismatches = 0
    counts: Dict[str, int] = {}
    for source in rows:
        row = dict(source)
        if str(row.get("teacher_model")) != expected_teacher:
            raise ValueError(f"Unexpected teacher model in trace {row.get('trace_id')}")
        predicted = extract_final_answer(str(row.get("solution", "")))
        is_correct = verify_answer(predicted, str(row["answer"]))
        if is_correct != bool(row.get("is_correct")):
            mismatches += 1
        row["predicted_answer"] = predicted
        row["is_correct"] = is_correct
        problem_id = str(row["problem_id"])
        counts[problem_id] = counts.get(problem_id, 0) + 1
        reverified.append(row)
    if mismatches:
        raise ValueError(f"Parent correctness flags disagree with current verifier: {mismatches}")
    wrong_counts = {key: value for key, value in counts.items() if value != 16}
    if wrong_counts:
        raise ValueError(f"Parent problems do not all contain 16 rollouts: {list(wrong_counts.items())[:5]}")
    return reverified


def _retokenize_rows(rows: Iterable[Mapping[str, Any]], tokenizer: Any) -> List[Dict[str, Any]]:
    """Bind all ranking and budget counts to the registered student tokenizer."""

    retokenized: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["parent_solution_token_count"] = int(row["solution_token_count"])
        row["solution_token_count"] = completion_token_count(tokenizer, str(row["solution"]))
        row["student_prompt"] = build_student_math_prompt(str(row["question"]))
        metadata = dict(row.get("metadata", {}))
        metadata["phase0_tokenization"] = {
            "basis": "registered_student_tokenizer",
            "parent_solution_token_count": row["parent_solution_token_count"],
            "student_solution_token_count": row["solution_token_count"],
        }
        row["metadata"] = metadata
        retokenized.append(row)
    return retokenized


def _training_row(row: Mapping[str, Any], regime: str, rank: str) -> Dict[str, Any]:
    return {
        "prompt": str(row["student_prompt"]),
        "teacher_prompt": str(row["prompt"]),
        "completion": str(row["solution"]),
        "problem_id": str(row["problem_id"]),
        "trace_id": str(row["trace_id"]),
        "length_rank": rank,
        "budget_regime": regime,
        "budget_token_count": int(row["budget_token_count"]),
        "model_input_token_count": int(row["model_input_token_count"]),
        "occurrence_index": int(row["occurrence_index"]),
        "parent_solution_token_count": int(row["parent_solution_token_count"]),
        "student_solution_token_count": int(row["solution_token_count"]),
        "selection": dict(row["metadata"]["phase0_length_selection"]),
    }


def _load_tokenizer(config: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers before building Phase-0 data.") from exc
    student = dict(config["student"])
    return AutoTokenizer.from_pretrained(
        str(student["tokenizer_name"]),
        revision=student.get("revision"),
        cache_dir=student.get("cache_dir"),
        trust_remote_code=bool(student.get("trust_remote_code", False)),
    )


def _require_frozen_protocol(config_path: Path, config_hash: str) -> None:
    marker_path = config_path.parent / "PROTOCOL_FROZEN"
    if config_path.name != "frozen_protocol.json" or not marker_path.is_file():
        raise ValueError("Formal data build requires a frozen protocol and PROTOCOL_FROZEN marker.")
    marker = _read_marker(marker_path)
    if marker.get("status") != "frozen" or marker.get("config_hash") != config_hash:
        raise ValueError("Frozen protocol marker does not match the requested config.")
    if marker.get("frozen_protocol_sha256") != file_sha256(config_path):
        raise ValueError("Frozen protocol file hash differs from its marker.")


def _read_jsonl_retry(path: Path, attempts: int, wait_seconds: float) -> List[Dict[str, Any]]:
    for attempt in range(1, attempts + 1):
        try:
            return [dict(row) for row in read_jsonl(path)]
        except OSError:
            if attempt == attempts:
                raise
            logging.warning("read retry %d/%d path=%s", attempt, attempts, path)
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def _read_json_retry(path: Path, attempts: int, wait_seconds: float) -> Dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object: {path}")
            return payload
        except OSError:
            if attempt == attempts:
                raise
            logging.warning("json retry %d/%d path=%s", attempt, attempts, path)
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def _file_sha256_retry(path: Path, attempts: int, wait_seconds: float) -> str:
    for attempt in range(1, attempts + 1):
        try:
            return file_sha256(path)
        except OSError:
            if attempt == attempts:
                raise
            logging.warning("hash retry %d/%d path=%s", attempt, attempts, path)
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def _read_marker(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Malformed marker: {path}")
        result[key] = value
    return result


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _resolve(value: str | None) -> Path:
    if value is None:
        raise ValueError("Path value is required.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
