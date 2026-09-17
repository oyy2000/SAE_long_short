#!/usr/bin/env python3
"""Evaluate one intervention-trained student on the locked GSM8K cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    publish_files_hash_verified,
    read_json,
    validated_training_artifacts,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.student_evaluation import (
    evaluate_explicit_questions,
    load_student_for_evaluation,
    summarize_predictions,
)
from length_budget_distill.student_prompts import build_student_math_prompt
from length_budget_distill.verifiers import extract_final_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    publish_output_dir = _resolve(args.output_dir)
    if publish_output_dir.exists():
        raise FileExistsError(f"Evaluation output exists: {publish_output_dir}")
    output_stage_root = os.environ.get("SAE_EVAL_OUTPUT_STAGE_ROOT")
    output_dir = (
        Path(output_stage_root) / args.model_id
        if output_stage_root
        else publish_output_dir
    )
    if output_dir.exists():
        raise FileExistsError(f"Runtime evaluation output exists: {output_dir}")
    adapter_evidence = None
    runtime_adapter_path = None
    if args.adapter_path:
        adapter_source = _resolve(args.adapter_path)
        recovery_root_value = os.environ.get("SAE_ADAPTER_RECOVERY_ROOT")
        recovery_source = (
            Path(recovery_root_value) / args.model_id if recovery_root_value else None
        )
        staging_source = (
            recovery_source
            if recovery_source is not None and recovery_source.is_dir()
            else adapter_source
        )
        stage_root = os.environ.get("SAE_EVAL_ADAPTER_STAGE_ROOT")
        runtime_adapter_path = (
            _stage_adapter(staging_source, Path(stage_root), args.model_id)
            if stage_root
            else staging_source
        )
        adapter_evidence = validated_training_artifacts(runtime_adapter_path)
        adapter_evidence["canonical_source_root"] = str(adapter_source)
        adapter_evidence["staging_source_root"] = str(staging_source)
        adapter_evidence["used_recovery_mirror"] = staging_source != adapter_source
    questions, dataset_evidence = _load_locked_questions(config)
    student = dict(config["student"])
    recovery_model_path = os.environ.get("SAE_EVAL_MODEL_PATH")
    if recovery_model_path:
        student.update(
            {
                "model_name": recovery_model_path,
                "tokenizer_name": recovery_model_path,
                "revision": None,
                "cache_dir": None,
            }
        )
    bundle = load_student_for_evaluation(
        student,
        adapter_path=str(runtime_adapter_path) if runtime_adapter_path else None,
    )
    evaluation = config["evaluation"]
    rows = evaluate_explicit_questions(
        bundle,
        questions,
        batch_size=int(evaluation["batch_size"]),
        max_new_tokens=int(evaluation["max_new_tokens"]),
    )
    summary = summarize_predictions(rows)
    summary.update(
        {
            "status": "complete",
            "model_id": args.model_id,
            "adapter_path": str(_resolve(args.adapter_path)) if args.adapter_path else None,
            "runtime_adapter_path": str(runtime_adapter_path) if runtime_adapter_path else None,
            "runtime_model_path": recovery_model_path,
            "config_hash": canonical_sha256(config),
            "dataset": dataset_evidence,
            "formal_claim_allowed": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    write_json_exclusive(summary_path, summary)
    manifest = {
        "status": "complete",
        "model_id": args.model_id,
        "config_path": str(config_path),
        "config_hash": canonical_sha256(config),
        "adapter_evidence": adapter_evidence,
        "dataset": dataset_evidence,
        "prediction_count": len(rows),
        "predictions_sha256": file_sha256(predictions_path),
        "summary_sha256": file_sha256(summary_path),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "EVALUATION_COMPLETE").write_text(
        "status=complete\n"
        f"model_id={args.model_id}\n"
        f"config_hash={manifest['config_hash']}\n"
        f"predictions_sha256={manifest['predictions_sha256']}\n"
        f"summary_sha256={manifest['summary_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n"
        f"prediction_count={len(rows)}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    if output_dir != publish_output_dir:
        publish_files_hash_verified(
            output_dir,
            publish_output_dir,
            (
                "predictions.jsonl",
                "summary.json",
                "evaluation_manifest.json",
                "EVALUATION_COMPLETE",
            ),
            attempts=120,
            wait_seconds=5.0,
        )
    print(json.dumps(summary, indent=2), flush=True)


def _load_locked_questions(config):
    from datasets import load_dataset

    evaluation = config["evaluation"]
    loaded = load_dataset(
        evaluation["dataset_name"],
        evaluation["dataset_config"],
        split=evaluation["split"],
        cache_dir=os.environ.get("HF_DATASETS_CACHE") or config["student"]["cache_dir"],
    )
    start = int(evaluation["start_index"])
    stop = start + int(evaluation["limit"])
    selected = list(loaded.select(range(start, stop)))
    questions = []
    for source_index, row in enumerate(selected, start=start):
        answer = extract_final_answer(str(row["answer"]))
        if answer is None:
            raise ValueError(f"Could not parse GSM8K gold answer at index {source_index}")
        questions.append(
            {
                "problem_id": f"hf-{source_index:06d}",
                "source_index": source_index,
                "question": str(row["question"]),
                "answer": answer,
                "student_prompt": build_student_math_prompt(str(row["question"])),
            }
        )
    id_hash = hashlib.sha256(
        ("\n".join(row["problem_id"] for row in questions) + "\n").encode("utf-8")
    ).hexdigest()
    return questions, {
        "dataset_name": evaluation["dataset_name"],
        "dataset_config": evaluation["dataset_config"],
        "split": evaluation["split"],
        "start_index": start,
        "limit": len(questions),
        "problem_ids_sha256": id_hash,
        "dataset_fingerprint": getattr(loaded, "_fingerprint", None),
    }


def _stage_adapter(source: Path, root: Path, model_id: str) -> Path:
    destination = root / model_id
    root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validated_training_artifacts(destination)
        return destination
    temporary = root / f".{model_id}.partial-{os.getpid()}"
    temporary.mkdir(parents=False, exist_ok=False)
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_metrics.json",
        "TRAIN_COMPLETE",
    ):
        _copy_with_remote_io_retry(source / filename, temporary / filename)
    validated_training_artifacts(temporary)
    temporary.rename(destination)
    return destination


def _copy_with_remote_io_retry(source: Path, destination: Path) -> None:
    for attempt in range(1, 121):
        try:
            shutil.copyfile(source, destination)
            if file_sha256(source) != file_sha256(destination):
                raise OSError("Staged adapter hash mismatch.")
            return
        except OSError:
            if destination.exists():
                destination.unlink()
            if attempt == 120:
                raise
            time.sleep(5)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
