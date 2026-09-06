"""Reusable greedy student evaluation on an explicit GSM8K question cohort."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Sequence

from .verifiers import extract_final_answer, verify_answer


def load_student_for_evaluation(
    student: Mapping[str, Any], *, adapter_path: str | None = None
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = {
        "revision": student.get("revision"),
        "cache_dir": student.get("cache_dir"),
        "local_files_only": True,
    }
    common = {key: value for key, value in common.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(student["model_name"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(student.get("torch_dtype", "bfloat16"))]
    model = AutoModelForCausalLM.from_pretrained(
        student["model_name"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        **common,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


def evaluate_explicit_questions(
    bundle: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("Evaluation batch size and max_new_tokens must be positive.")
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    torch = bundle["torch"]
    output = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(row["student_prompt"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        suffix = generated[:, inputs["input_ids"].shape[-1] :]
        texts = [
            value.strip()
            for value in tokenizer.batch_decode(suffix, skip_special_tokens=True)
        ]
        for row, text in zip(batch, texts):
            predicted = extract_final_answer(text)
            output.append(
                {
                    "problem_id": str(row["problem_id"]),
                    "source_index": int(row["source_index"]),
                    "question": str(row["question"]),
                    "gold_answer": str(row["answer"]),
                    "prediction_text": text,
                    "predicted_answer": predicted,
                    "is_correct": verify_answer(predicted, str(row["answer"])),
                    "output_token_count": len(
                        tokenizer.encode(text, add_special_tokens=False)
                    ),
                }
            )
    return output


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty predictions.")
    correct = sum(bool(row["is_correct"]) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_output_tokens": statistics.fmean(
            float(row["output_token_count"]) for row in rows
        ),
    }
