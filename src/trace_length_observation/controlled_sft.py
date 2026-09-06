"""Explicit-loss LoRA SFT for controlled trace-observation experiments."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from length_budget_distill.records import read_jsonl
from .trace_observation import (
    LOSS_MASKS,
    LOSS_NORMALIZATIONS,
    answer_token_mask,
    training_prompt_token_ids,
)


@dataclass
class EncodedTrainingRecord:
    problem_id: str
    trace_id: str
    occurrence_index: int
    input_ids: List[int]
    attention_mask: List[int]
    loss_weights: List[float]
    completion_token_count: int


def encode_training_record(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    loss_mask: str,
    max_length: int,
    completion_separator: str = "\n\n",
) -> EncodedTrainingRecord:
    """Tokenize without truncating and construct an auditable target mask."""

    if loss_mask not in LOSS_MASKS:
        raise ValueError(f"Unsupported loss mask: {loss_mask}")
    prompt = str(row["prompt"])
    completion = str(row["completion"])
    prompt_ids = training_prompt_token_ids(tokenizer, prompt)
    separator_ids = list(tokenizer.encode(completion_separator, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    if not completion_ids:
        raise ValueError(f"Completion tokenized to zero tokens: {row.get('trace_id')}")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    append_eos = eos_id is not None and completion_ids[-1] != int(eos_id)
    input_ids = prompt_ids + separator_ids + completion_ids + ([int(eos_id)] if append_eos else [])
    if len(input_ids) > max_length:
        raise ValueError(
            f"Training example exceeds max_length without safe truncation: "
            f"trace={row.get('trace_id')} tokens={len(input_ids)} max={max_length}"
        )
    if loss_mask == "full_completion":
        completion_weights = [1.0] * len(completion_ids)
    else:
        completion_weights = [float(value) for value in answer_token_mask(tokenizer, completion)]
        if len(completion_weights) != len(completion_ids):
            raise ValueError("Answer token mask length differs from completion token count.")
    # EOS is a target in both conditions but is not part of the registered
    # completion-token budget, which follows the generation length convention.
    if append_eos:
        completion_weights.append(1.0)
    loss_weights = [0.0] * (len(prompt_ids) + len(separator_ids)) + completion_weights
    if not any(weight > 0.0 for weight in loss_weights):
        raise ValueError(f"No supervised tokens for trace {row.get('trace_id')}")
    return EncodedTrainingRecord(
        problem_id=str(row["problem_id"]),
        trace_id=str(row["trace_id"]),
        occurrence_index=int(row.get("occurrence_index", 0)),
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        loss_weights=loss_weights,
        completion_token_count=len(completion_ids),
    )


def collate_encoded_records(
    records: Sequence[EncodedTrainingRecord],
    *,
    pad_token_id: int,
    torch_module: Any,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("Cannot collate an empty record batch.")
    width = max(len(record.input_ids) for record in records)
    input_ids = []
    attention_mask = []
    loss_weights = []
    for record in records:
        padding = width - len(record.input_ids)
        input_ids.append(record.input_ids + [pad_token_id] * padding)
        attention_mask.append(record.attention_mask + [0] * padding)
        loss_weights.append(record.loss_weights + [0.0] * padding)
    return {
        "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
        "attention_mask": torch_module.tensor(attention_mask, dtype=torch_module.long),
        "loss_weights": torch_module.tensor(loss_weights, dtype=torch_module.float32),
    }


def controlled_causal_loss(
    logits: Any,
    input_ids: Any,
    loss_weights: Any,
    *,
    normalization: str,
    torch_module: Any,
) -> Any:
    """Compute token-mean or per-sequence-mean next-token cross entropy."""

    if normalization not in LOSS_NORMALIZATIONS:
        raise ValueError(f"Unsupported loss normalization: {normalization}")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous().float()
    flat_losses = torch_module.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_targets.view(-1),
        reduction="none",
    )
    token_losses = flat_losses.view_as(shift_targets) * shift_weights
    per_sequence_counts = shift_weights.sum(dim=1)
    if bool((per_sequence_counts <= 0).any().item()):
        raise ValueError("Every sequence must contain at least one supervised target token.")
    if normalization == "token_mean":
        return token_losses.sum() / per_sequence_counts.sum()
    return (token_losses.sum(dim=1) / per_sequence_counts).mean()


def deterministic_training_order(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> List[Dict[str, Any]]:
    """Use problem identity, not length-rank trace identity, for paired ordering."""

    return sorted(
        (dict(row) for row in rows),
        key=lambda row: hashlib.sha256(
            (
                f"{int(seed)}:{row.get('problem_id')}:"
                f"{int(row.get('occurrence_index', 0))}"
            ).encode("utf-8")
        ).hexdigest(),
    )


def run_controlled_lora_sft(
    run_config: Mapping[str, Any],
    *,
    runtime_output_dir: str | Path,
) -> Dict[str, Any]:
    """Train one Phase-0 cell and return detailed optimization accounting."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("Controlled SFT requires torch, transformers, and peft.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Controlled Phase-0 SFT requires CUDA.")
    training = dict(run_config["training"])
    student = dict(run_config["student"])
    loss_normalization = str(run_config["condition"]["loss_normalization"])
    loss_mask = str(run_config["condition"]["loss_mask"])
    if loss_normalization not in LOSS_NORMALIZATIONS or loss_mask not in LOSS_MASKS:
        raise ValueError("Run condition contains unsupported loss controls.")
    seed = int(run_config["condition"]["seed"])
    _set_seeds(seed, torch)

    model_name = str(student["model_name"])
    revision = student.get("revision")
    cache_dir = student.get("cache_dir")
    common_kwargs: Dict[str, Any] = {
        "revision": revision,
        "cache_dir": cache_dir,
        "trust_remote_code": bool(student.get("trust_remote_code", False)),
    }
    common_kwargs = {key: value for key, value in common_kwargs.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(
        str(student.get("tokenizer_name", model_name)), **common_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _resolve_dtype(torch, str(student.get("torch_dtype", "bfloat16")))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **common_kwargs,
    )
    model.config.use_cache = False
    lora = dict(student["lora"])
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
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_grads):
            enable_input_grads()
    device = torch.device("cuda:0")
    model.to(device)
    model.train()

    train_path = Path(str(run_config["data"]["train_path"]))
    rows = deterministic_training_order(list(read_jsonl(train_path)), seed)
    max_length = int(training["max_length"])
    encoded = [
        encode_training_record(
            tokenizer,
            row,
            loss_mask=loss_mask,
            max_length=max_length,
            completion_separator=str(training.get("completion_separator", "\n\n")),
        )
        for row in rows
    ]
    batch_size = int(training["per_device_train_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    batches = [encoded[start : start + batch_size] for start in range(0, len(encoded), batch_size)]
    optimizer_steps = int(math.ceil(len(batches) / accumulation))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    warmup_steps = int(math.ceil(float(training.get("warmup_ratio", 0.0)) * optimizer_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    pad_token_id = int(tokenizer.pad_token_id)
    use_autocast = dtype in {torch.bfloat16, torch.float16}
    losses: List[float] = []
    grad_norms: List[float] = []
    effective_loss_token_updates = 0
    completion_token_updates = 0
    model_input_token_updates = 0
    padded_model_token_updates = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    for group_start in range(0, len(batches), accumulation):
        group = batches[group_start : group_start + accumulation]
        for records in group:
            batch = collate_encoded_records(
                records, pad_token_id=pad_token_id, torch_module=torch
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_weights = batch["loss_weights"].to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = controlled_causal_loss(
                    outputs.logits,
                    input_ids,
                    loss_weights,
                    normalization=loss_normalization,
                    torch_module=torch,
                )
            (loss / len(group)).backward()
            losses.append(float(loss.detach().cpu()))
            effective_loss_token_updates += int(loss_weights[:, 1:].sum().item())
            completion_token_updates += sum(record.completion_token_count for record in records)
            model_input_token_updates += sum(len(record.input_ids) for record in records)
            padded_model_token_updates += int(input_ids.numel())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norms.append(float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        logging.info(
            "controlled_sft run=%s step=%d/%d mean_loss=%.6f effective_tokens=%d completion_tokens=%d",
            run_config["run_name"],
            completed_steps,
            optimizer_steps,
            statistics_mean(losses[-len(group) :]),
            effective_loss_token_updates,
            completion_token_updates,
        )
    if completed_steps != optimizer_steps:
        raise AssertionError("Observed optimizer-step count differs from the planned count.")

    output_dir = Path(runtime_output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    # The conventional dense-transformer training proxy is 6*N*T FLOPs. It is
    # intentionally reported as an approximation beside observed tokens and
    # wall-clock time; LoRA does not remove the dense forward/backward passes.
    approximate_nonpadding_flops = 6 * total_parameter_count * model_input_token_updates
    approximate_padded_flops = 6 * total_parameter_count * padded_model_token_updates
    return {
        "status": "complete",
        "run_name": str(run_config["run_name"]),
        "seed": seed,
        "loss_normalization": loss_normalization,
        "loss_mask": loss_mask,
        "record_count": len(encoded),
        "unique_problem_count": len({record.problem_id for record in encoded}),
        "batch_count": len(batches),
        "optimizer_steps": optimizer_steps,
        "completion_token_updates": completion_token_updates,
        "model_input_token_updates": model_input_token_updates,
        "padded_model_token_updates": padded_model_token_updates,
        "effective_loss_token_updates": effective_loss_token_updates,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "approximate_nonpadding_training_flops": approximate_nonpadding_flops,
        "approximate_padded_training_flops": approximate_padded_flops,
        "flops_proxy_definition": "6 * total_parameter_count * token_updates",
        "mean_train_loss": statistics_mean(losses),
        "final_train_loss": losses[-1],
        "max_observed_grad_norm": max(grad_norms),
        "elapsed_seconds": time.monotonic() - started,
        "torch_initial_seed": int(torch.initial_seed()),
    }


def statistics_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return float(sum(float(value) for value in values) / len(values))


def _set_seeds(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False


def _resolve_dtype(torch_module: Any, name: str) -> Any:
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]
