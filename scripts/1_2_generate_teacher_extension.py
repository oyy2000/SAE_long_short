#!/usr/bin/env python3
"""Generate one independent shard of the 1119-question Phase-1 teacher extension."""

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
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.ranked_sampling import (
    build_length_agnostic_teacher_prompt,
)
from length_budget_distill.teaching_utility import stable_hash
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_teaching_utility_v1.json")
    parser.add_argument("--question-shard", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    question_path = _resolve(args.question_shard)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    gate = require_passed_gate(PROJECT_ROOT, config["gate_dependency"])
    questions = list(read_jsonl(question_path))
    if not questions:
        raise ValueError("Question shard is empty.")
    teacher = dict(config["teacher"])
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=teacher["model_name"],
        revision=teacher["revision"],
        download_dir=teacher["cache_dir"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=64,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        build_length_agnostic_teacher_prompt(str(row["question"])) for row in questions
    ]
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    sampling = [
        SamplingParams(
            n=int(teacher["num_rollouts"]),
            temperature=float(teacher["temperature"]),
            top_p=float(teacher["top_p"]),
            max_tokens=int(teacher["max_new_tokens"]),
            stop=["<|im_end|>"],
            seed=int(stable_hash(int(teacher["base_seed"]), row["problem_id"])[:8], 16),
        )
        for row in questions
    ]
    generated = llm.generate(formatted, sampling, use_tqdm=True)
    rows = []
    for question, teacher_prompt, request in zip(questions, prompts, generated):
        if len(request.outputs) != int(teacher["num_rollouts"]):
            raise RuntimeError(f"Rollout count mismatch for {question['problem_id']}.")
        for candidate_index, output in enumerate(request.outputs):
            solution = output.text.strip()
            predicted = extract_final_answer(solution)
            rows.append(
                {
                    "trace_id": f"{question['problem_id']}:phase1:{candidate_index:02d}",
                    "problem_id": question["problem_id"],
                    "source_index": question["source_index"],
                    "candidate_index": candidate_index,
                    "question": question["question"],
                    "answer": question["answer"],
                    "prompt": teacher_prompt,
                    "solution": solution,
                    "predicted_answer": predicted,
                    "is_correct": verify_answer(predicted, str(question["answer"])),
                    "teacher_model": teacher["model_name"],
                    "teacher_revision": teacher["revision"],
                    "solution_token_count": len(output.token_ids),
                    "finish_reason": output.finish_reason,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_trajectories.jsonl"
    count = write_jsonl(raw_path, rows)
    manifest = {
        "status": "complete",
        "config_hash": canonical_sha256(config),
        "config_sha256": file_sha256(config_path),
        "gate_evidence": gate,
        "question_shard_path": str(question_path),
        "question_shard_sha256": file_sha256(question_path),
        "question_count": len(questions),
        "candidate_count": count,
        "expected_candidate_count": len(questions) * int(teacher["num_rollouts"]),
        "correct_candidate_count": sum(bool(row["is_correct"]) for row in rows),
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "prompt_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/ranked_sampling.py"
        ),
        "runtime": runtime_metadata(),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "generation_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "GENERATION_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\nmanifest_sha256={file_sha256(manifest_path)}\nraw_sha256={manifest['raw_sha256']}\n",
        encoding="utf-8",
    )


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
