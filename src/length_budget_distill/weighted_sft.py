"""LoRA SFT with precomputed per-token supervision weights."""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from trace_length_observation.controlled_sft import (
    EncodedTrainingRecord,
    collate_encoded_records,
    controlled_causal_loss,
    deterministic_training_order,
)
from trace_length_observation.trace_observation import training_prompt_token_ids

from .records import read_jsonl


def encode_weighted_training_record(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
    completion_separator: str = "",
) -> EncodedTrainingRecord:
    prompt_ids = training_prompt_token_ids(tokenizer, str(row["prompt"]))
    separator_ids = list(
        tokenizer.encode(completion_separator, add_special_tokens=False)
    )
    completion_ids = list(
        tokenizer.encode(str(row["completion"]), add_special_tokens=False)
    )
    weights = [
        float(value)
        for value in row.get("completion_loss_weights", [1.0] * len(completion_ids))
    ]
    if len(weights) != len(completion_ids):
        raise ValueError(
            f"Precomputed mask/token mismatch for {row.get('trace_id')}: {len(weights)} != {len(completion_ids)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Supervision weights must be finite and non-negative.")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    append_eos = eos_id is not None and (
        not completion_ids or completion_ids[-1] != int(eos_id)
    )
    input_ids = (
        prompt_ids
        + separator_ids
        + completion_ids
        + ([int(eos_id)] if append_eos else [])
    )
    loss_weights = (
        [0.0] * (len(prompt_ids) + len(separator_ids))
        + weights
        + ([1.0] if append_eos else [])
    )
    if len(input_ids) > max_length:
        raise ValueError(f"Weighted example exceeds max_length: {row.get('trace_id')}")
    if not any(value > 0.0 for value in loss_weights):
        raise ValueError(f"No supervised targets: {row.get('trace_id')}")
    return EncodedTrainingRecord(
        problem_id=str(row["problem_id"]),
        trace_id=str(row["trace_id"]),
        occurrence_index=int(row.get("occurrence_index", 0)),
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        loss_weights=loss_weights,
        completion_token_count=len(completion_ids),
    )


def run_weighted_lora_sft(
    run_config: Mapping[str, Any], *, runtime_output_dir: str | Path
) -> Dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Weighted SFT requires CUDA.")
    training = dict(run_config["training"])
    student = dict(run_config["student"])
    seed = int(run_config["condition"]["seed"])
    if int(training.get("num_train_epochs", 1)) != 1:
        raise ValueError(
            "The registered weighted-SFT runner currently requires one epoch."
        )
    _set_seeds(seed, torch)
    common = {
        "revision": student.get("revision"),
        "cache_dir": student.get("cache_dir"),
        "local_files_only": True,
    }
    common = {key: value for key, value in common.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(student["model_name"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(student.get("torch_dtype", "bfloat16"))]
    model = AutoModelForCausalLM.from_pretrained(
        student["model_name"], torch_dtype=dtype, low_cpu_mem_usage=True, **common
    )
    model.config.use_cache = False
    lora = student["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if callable(getattr(model, "enable_input_require_grads", None)):
            model.enable_input_require_grads()
    model.to(torch.device("cuda:0"))
    model.train()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    source_rows = list(read_jsonl(Path(run_config["data"]["train_path"])))
    rows = (
        source_rows
        if bool(training.get("preserve_input_order", False))
        else deterministic_training_order(source_rows, seed)
    )
    encoded = [
        encode_weighted_training_record(
            tokenizer,
            row,
            max_length=int(training["max_length"]),
            completion_separator=str(training.get("completion_separator", "")),
        )
        for row in rows
    ]
    batch_size = int(training["per_device_train_batch_size"])
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    batches = [
        encoded[start : start + batch_size]
        for start in range(0, len(encoded), batch_size)
    ]
    optimizer_steps = math.ceil(len(batches) / accumulation)
    scheduler_total_steps = int(
        training.get("scheduler_total_optimizer_steps", optimizer_steps)
    )
    if scheduler_total_steps < optimizer_steps:
        raise ValueError(
            "scheduler_total_optimizer_steps cannot be smaller than executed steps."
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    warmup_steps = math.ceil(
        float(training.get("warmup_ratio", 0.03)) * scheduler_total_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, scheduler_total_steps
    )
    losses: List[float] = []
    effective_weight_updates = 0.0
    positive_target_updates = 0
    completion_token_updates = 0
    model_input_token_updates = 0
    padded_model_token_updates = 0
    started = time.monotonic()
    step_metrics = []
    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    for group_start in range(0, len(batches), accumulation):
        group = batches[group_start : group_start + accumulation]
        for records in group:
            batch = collate_encoded_records(
                records, pad_token_id=int(tokenizer.pad_token_id), torch_module=torch
            )
            input_ids = batch["input_ids"].to("cuda:0")
            attention = batch["attention_mask"].to("cuda:0")
            weights = batch["loss_weights"].to("cuda:0")
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=dtype != torch.float32
            ):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention, use_cache=False
                )
                loss = controlled_causal_loss(
                    outputs.logits,
                    input_ids,
                    weights,
                    normalization="token_mean",
                    torch_module=torch,
                )
            (loss / len(group)).backward()
            losses.append(float(loss.detach().cpu()))
            shifted = weights[:, 1:]
            effective_weight_updates += float(shifted.sum().item())
            positive_target_updates += int((shifted > 0).sum().item())
            completion_token_updates += sum(
                record.completion_token_count for record in records
            )
            model_input_token_updates += sum(
                len(record.input_ids) for record in records
            )
            padded_model_token_updates += int(input_ids.numel())
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training.get("max_grad_norm", 1.0))
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        step_metrics.append(
            {
                "optimizer_step": completed_steps,
                "mean_microbatch_loss": sum(losses[-len(group) :]) / len(group),
                "completion_token_updates": completion_token_updates,
                "model_input_token_updates": model_input_token_updates,
                "positive_target_token_updates": positive_target_updates,
                "effective_supervision_weight_updates": effective_weight_updates,
            }
        )
        logging.info(
            "weighted_sft run=%s step=%d/%d mean_loss=%.6f",
            run_config["run_name"],
            completed_steps,
            optimizer_steps,
            sum(losses[-len(group) :]) / len(group),
        )
    output_dir = Path(runtime_output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    return {
        "status": "complete",
        "run_name": str(run_config["run_name"]),
        "condition": str(run_config["condition"]["name"]),
        "seed": seed,
        "record_count": len(encoded),
        "unique_problem_count": len({row.problem_id for row in encoded}),
        "optimizer_steps": completed_steps,
        "scheduler_total_optimizer_steps": scheduler_total_steps,
        "preserved_input_order": bool(training.get("preserve_input_order", False)),
        "completion_token_updates": completion_token_updates,
        "model_input_token_updates": model_input_token_updates,
        "padded_model_token_updates": padded_model_token_updates,
        "positive_target_token_updates": positive_target_updates,
        "effective_supervision_weight_updates": effective_weight_updates,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "approximate_nonpadding_training_flops": 6
        * total_parameters
        * model_input_token_updates,
        "approximate_padded_training_flops": 6
        * total_parameters
        * padded_model_token_updates,
        "mean_train_loss": sum(losses) / len(losses),
        "final_train_loss": losses[-1],
        "optimizer_step_metrics": step_metrics,
        "elapsed_seconds": time.monotonic() - started,
    }


def _set_seeds(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False
