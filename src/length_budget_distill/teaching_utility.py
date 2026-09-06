"""Counterfactual Teaching Value primitives for LoRA students.

The functions here are deliberately independent of experiment I/O. Entry
scripts own manifests and markers, while this module owns token alignment,
gradient algebra, reversible virtual updates, calibration, and fixed effects.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .verifiers import extract_final_answer


@dataclass(frozen=True)
class EncodedCompletion:
    record_id: str
    input_ids: List[int]
    prompt_token_count: int
    target_ids: List[int]
    answer_target_positions: List[int]


def load_lora_student(
    student_config: Mapping[str, Any],
    *,
    adapter_path: str | None = None,
) -> Tuple[Any, Any, List[Tuple[str, Any]], Dict[str, Any]]:
    """Load a base-equivalent zero LoRA or a registered adapter anchor."""

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CTV requires CUDA.")
    seed = int(student_config["lora"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    common = {
        "revision": student_config["revision"],
        "cache_dir": student_config["cache_dir"],
        "local_files_only": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(student_config["model_name"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(student_config.get("torch_dtype", "bfloat16"))]
    base = AutoModelForCausalLM.from_pretrained(
        student_config["model_name"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **common,
    ).to("cuda")
    base.config.use_cache = False
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    else:
        lora = dict(student_config["lora"])
        model = get_peft_model(
            base,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=lora["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    trainable = trainable_parameters(model)
    for _, parameter in trainable:
        parameter.data = parameter.data.float()
    evidence = {
        "model_name": student_config["model_name"],
        "revision": student_config["revision"],
        "anchor_adapter_path": adapter_path,
        "lora_seed": seed,
        "trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in trainable
        ),
        "trainable_tensor_count": len(trainable),
        "initial_lora_state_sha256": trainable_state_sha256(trainable),
    }
    return model, tokenizer, trainable, evidence


def stable_hash(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed), *(str(value) for value in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_question_split(
    problem_ids: Sequence[str],
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
    seed: int,
) -> Dict[str, List[str]]:
    expected = train_count + dev_count + test_count
    unique = sorted(set(str(value) for value in problem_ids))
    if len(unique) != expected or len(unique) != len(problem_ids):
        raise ValueError(
            f"Expected exactly {expected} unique problem IDs, observed {len(unique)}."
        )
    ordered = sorted(unique, key=lambda value: stable_hash(seed, value))
    return {
        "train": ordered[:train_count],
        "dev": ordered[train_count : train_count + dev_count],
        "test": ordered[train_count + dev_count :],
    }


def tokenize_prompt_completion(
    tokenizer: Any,
    *,
    record_id: str,
    prompt: str,
    completion: str,
    max_length: int,
) -> EncodedCompletion:
    if not prompt.strip() or not completion.strip():
        raise ValueError(f"Missing prompt/completion: {record_id}")
    prompt_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = list(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Prompt tokens are not a prefix of the chat sequence: {record_id}"
        )
    if len(full_ids) > max_length:
        raise ValueError(
            f"Sequence exceeds max_length: {record_id} {len(full_ids)} > {max_length}"
        )
    target_ids = [int(value) for value in full_ids[len(prompt_ids) :]]
    if not target_ids:
        raise ValueError(f"Completion has no targets: {record_id}")
    answer_positions: List[int] = []
    answer = extract_final_answer(completion)
    if answer:
        answer_ids = list(tokenizer.encode(answer, add_special_tokens=False))
        start = find_last_subsequence(target_ids, answer_ids)
        if start is not None:
            answer_positions = list(range(start, start + len(answer_ids)))
    return EncodedCompletion(
        record_id=record_id,
        input_ids=[int(value) for value in full_ids],
        prompt_token_count=len(prompt_ids),
        target_ids=target_ids,
        answer_target_positions=answer_positions,
    )


def completion_token_loss_vector(
    model: Any, encoded: EncodedCompletion
) -> Tuple[Any, Any, Any]:
    """Return differentiable target losses, aligned logits, and target IDs."""

    import torch

    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device="cuda")
    outputs = model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
    )
    start = encoded.prompt_token_count - 1
    logits = outputs.logits[0, start : len(encoded.input_ids) - 1, :].float()
    if logits.shape[0] != len(encoded.target_ids):
        raise RuntimeError(f"Target/logit alignment failed for {encoded.record_id}")
    targets = torch.tensor(encoded.target_ids, dtype=torch.long, device=logits.device)
    target_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
    token_losses = torch.logsumexp(logits, dim=-1) - target_logits
    return token_losses, logits, targets


def completion_loss_tensors(model: Any, encoded: EncodedCompletion) -> Dict[str, Any]:
    import torch

    token_losses, logits, targets = completion_token_loss_vector(model, encoded)
    target_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
    top_rank_logits = torch.topk(logits, k=min(100, logits.shape[-1]), dim=-1).values
    token_ranks = 1 + (top_rank_logits > target_logits[:, None]).sum(dim=-1)
    token_ranks = token_ranks.clamp(max=100)
    top50_logits, top50_indices = torch.topk(
        logits, k=min(50, logits.shape[-1]), dim=-1
    )
    top50_probabilities = torch.softmax(top50_logits, dim=-1)
    true_probability = (top50_probabilities * (top50_indices == targets[:, None])).sum(
        dim=-1
    )
    brier_top50 = (
        1.0 - 2.0 * true_probability + (top50_probabilities**2).sum(dim=-1)
    ).mean()
    answer_loss_mean = None
    if encoded.answer_target_positions:
        positions = torch.tensor(
            encoded.answer_target_positions, dtype=torch.long, device=logits.device
        )
        answer_loss_mean = token_losses.index_select(0, positions).mean()
    return {
        "token_losses": token_losses,
        "loss_sum": token_losses.sum(),
        "loss_mean": token_losses.mean(),
        "answer_loss_mean": answer_loss_mean,
        "token_count": len(encoded.target_ids),
        "mean_token_rank": token_ranks.float().mean(),
        "mean_clipped_token_rank_100": token_ranks.clamp(max=100).float().mean(),
        "brier_top50": brier_top50,
    }


def trainable_parameters(model: Any) -> List[Tuple[str, Any]]:
    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not values:
        raise ValueError("Model has no trainable parameters.")
    return values


def gradient_for_completion(
    model: Any,
    trainable: Sequence[Tuple[str, Any]],
    encoded: EncodedCompletion,
    *,
    reduction: str,
) -> Dict[str, Any]:
    if reduction not in {"token_sum", "token_mean"}:
        raise ValueError("reduction must be token_sum or token_mean.")
    model.zero_grad(set_to_none=True)
    losses = completion_loss_tensors(model, encoded)
    objective = losses["loss_sum"] if reduction == "token_sum" else losses["loss_mean"]
    objective.backward()
    gradients = clone_gradients(trainable)
    output = {
        "gradients": gradients,
        "reduction": reduction,
        "loss_sum": float(losses["loss_sum"].detach().cpu()),
        "loss_mean": float(losses["loss_mean"].detach().cpu()),
        "token_count": int(losses["token_count"]),
        "gradient_norm": gradient_norm(gradients),
        "answer_loss_mean": (
            float(losses["answer_loss_mean"].detach().cpu())
            if losses["answer_loss_mean"] is not None
            else None
        ),
        "mean_token_rank": float(losses["mean_token_rank"].detach().cpu()),
        "rsr_rank_clip_100": float(losses["mean_clipped_token_rank_100"].detach().cpu())
        / max(float(losses["loss_mean"].detach().cpu()), 1e-12),
        "brier_top50": float(losses["brier_top50"].detach().cpu()),
    }
    model.zero_grad(set_to_none=True)
    return output


def gradient_for_completion_positions(
    model: Any,
    trainable: Sequence[Tuple[str, Any]],
    encoded: EncodedCompletion,
    *,
    target_positions: Sequence[int],
) -> Dict[str, Any]:
    """Differentiate a token-mean objective on positions in the full trace context."""

    import torch

    positions = sorted(set(int(value) for value in target_positions))
    if not positions or positions[0] < 0 or positions[-1] >= len(encoded.target_ids):
        raise ValueError("Target positions are empty or outside the completion.")
    model.zero_grad(set_to_none=True)
    token_losses, _, _ = completion_token_loss_vector(model, encoded)
    index = torch.tensor(positions, dtype=torch.long, device=token_losses.device)
    selected = token_losses.index_select(0, index)
    objective = selected.mean()
    objective.backward()
    gradients = clone_gradients(trainable)
    model.zero_grad(set_to_none=True)
    return {
        "gradients": gradients,
        "loss_mean": float(objective.detach().cpu()),
        "token_count": len(positions),
        "gradient_norm": gradient_norm(gradients),
    }


def mean_probe_gradient(
    model: Any,
    trainable: Sequence[Tuple[str, Any]],
    probe: Sequence[EncodedCompletion],
    *,
    answer_only: bool = False,
) -> Tuple[List[Any], float]:
    if not probe:
        raise ValueError("Probe must be non-empty.")
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    for start in range(0, len(probe), 4):
        values = batched_probe_losses(
            model, probe[start : start + 4], answer_only=answer_only
        )
        objective = values.sum() / len(probe)
        objective.backward()
        total_loss += float(values.detach().sum().cpu())
    gradients = clone_gradients(trainable)
    model.zero_grad(set_to_none=True)
    return gradients, total_loss / len(probe)


def mean_probe_loss(
    model: Any,
    probe: Sequence[EncodedCompletion],
    *,
    answer_only: bool = False,
) -> float:
    import torch

    values: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(probe), 4):
            batch = batched_probe_losses(
                model, probe[start : start + 4], answer_only=answer_only
            )
            values.extend(float(value) for value in batch.detach().cpu().tolist())
    return sum(values) / len(values)


def mean_probe_rationale_and_answer_loss(
    model: Any, probe: Sequence[EncodedCompletion]
) -> Tuple[float, float]:
    """Measure rationale and answer-only losses from the same forward passes."""

    import torch

    rationale_values: List[float] = []
    answer_values: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(probe), 4):
            rationale, answer = batched_probe_loss_components(
                model, probe[start : start + 4]
            )
            rationale_values.extend(
                float(value) for value in rationale.detach().cpu().tolist()
            )
            answer_values.extend(
                float(value) for value in answer.detach().cpu().tolist()
            )
    return (
        sum(rationale_values) / len(rationale_values),
        sum(answer_values) / len(answer_values),
    )


def batched_probe_losses(
    model: Any,
    probe: Sequence[EncodedCompletion],
    *,
    answer_only: bool,
) -> Any:
    """Return exact per-sequence losses from a right-padded probe batch."""

    import torch

    if not probe:
        raise ValueError("Probe batch is empty.")
    token_losses_by_record = _batched_probe_token_losses(model, probe)
    values = []
    for encoded, token_losses in zip(probe, token_losses_by_record):
        if answer_only:
            if not encoded.answer_target_positions:
                raise ValueError(f"Answer target was not aligned: {encoded.record_id}")
            positions = torch.tensor(
                encoded.answer_target_positions,
                dtype=torch.long,
                device=token_losses.device,
            )
            token_losses = token_losses.index_select(0, positions)
        values.append(token_losses.mean())
    return torch.stack(values)


def batched_probe_loss_components(
    model: Any, probe: Sequence[EncodedCompletion]
) -> Tuple[Any, Any]:
    """Return per-sequence rationale and answer-only losses from one forward."""

    import torch

    token_losses_by_record = _batched_probe_token_losses(model, probe)
    rationale_values = []
    answer_values = []
    for encoded, token_losses in zip(probe, token_losses_by_record):
        if not encoded.answer_target_positions:
            raise ValueError(f"Answer target was not aligned: {encoded.record_id}")
        positions = torch.tensor(
            encoded.answer_target_positions,
            dtype=torch.long,
            device=token_losses.device,
        )
        rationale_values.append(token_losses.mean())
        answer_values.append(token_losses.index_select(0, positions).mean())
    return torch.stack(rationale_values), torch.stack(answer_values)


def _batched_probe_token_losses(
    model: Any, probe: Sequence[EncodedCompletion]
) -> List[Any]:
    import torch

    width = max(len(encoded.input_ids) for encoded in probe)
    input_ids = torch.zeros((len(probe), width), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, encoded in enumerate(probe):
        length = len(encoded.input_ids)
        input_ids[index, :length] = torch.tensor(
            encoded.input_ids, dtype=torch.long, device="cuda"
        )
        attention[index, :length] = 1
    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    values = []
    for index, encoded in enumerate(probe):
        start = encoded.prompt_token_count - 1
        logits = outputs.logits[index, start : len(encoded.input_ids) - 1, :].float()
        targets = torch.tensor(
            encoded.target_ids, dtype=torch.long, device=logits.device
        )
        values.append(
            torch.logsumexp(logits, dim=-1)
            - logits.gather(-1, targets[:, None]).squeeze(-1)
        )
    return values


def gradient_dot(left: Sequence[Any], right: Sequence[Any]) -> float:
    import torch

    if len(left) != len(right):
        raise ValueError("Gradient structures differ in length.")
    value = torch.zeros((), dtype=torch.float64, device="cuda")
    for left_tensor, right_tensor in zip(left, right):
        if left_tensor.shape != right_tensor.shape:
            raise ValueError("Gradient structures differ in shape.")
        value += torch.sum(left_tensor.double() * right_tensor.double())
    return float(value.detach().cpu())


def gradient_norm(gradients: Sequence[Any]) -> float:
    return math.sqrt(max(0.0, gradient_dot(gradients, gradients)))


def first_order_ctv(
    probe_gradient: Sequence[Any], trace_gradient: Sequence[Any], eta: float
) -> float:
    return float(eta) * gradient_dot(probe_gradient, trace_gradient)


def reversible_micro_update(
    model: Any,
    trainable: Sequence[Tuple[str, Any]],
    trace_gradient: Sequence[Any],
    *,
    eta: float,
    probe: Sequence[EncodedCompletion],
    base_probe_loss: float,
    answer_only: bool = False,
) -> Dict[str, Any]:
    import torch

    if len(trainable) != len(trace_gradient):
        raise ValueError("Update structure mismatch.")
    before_hash = trainable_state_sha256(trainable)
    originals = [parameter.detach().clone() for _, parameter in trainable]
    try:
        with torch.no_grad():
            for (_, parameter), gradient in zip(trainable, trace_gradient):
                parameter.add_(gradient.to(parameter.dtype), alpha=-float(eta))
        after_loss = mean_probe_loss(model, probe, answer_only=answer_only)
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(trainable, originals):
                parameter.copy_(original)
    after_hash = trainable_state_sha256(trainable)
    if before_hash != after_hash:
        raise RuntimeError("LoRA parameters were not restored exactly.")
    return {
        "probe_loss_before": float(base_probe_loss),
        "probe_loss_after": float(after_loss),
        "ctv_exact": float(base_probe_loss) - float(after_loss),
        "parameters_restored_exactly": True,
        "state_sha256": after_hash,
    }


def reversible_micro_update_multi(
    model: Any,
    trainable: Sequence[Tuple[str, Any]],
    trace_gradient: Sequence[Any],
    *,
    eta: float,
    probes: Mapping[str, Sequence[EncodedCompletion]],
    base_probe_losses: Mapping[str, float],
    answer_only_by_probe: Mapping[str, bool] | None = None,
) -> Dict[str, Any]:
    """Apply one virtual update, evaluate several probe sets, and restore once."""

    import torch

    if set(probes) != set(base_probe_losses):
        raise ValueError("Probe sets and baseline losses have different keys.")
    answer_only = dict(answer_only_by_probe or {})
    if set(answer_only) - set(probes):
        raise ValueError("Answer-only flags contain unknown probe sets.")
    before_hash = trainable_state_sha256(trainable)
    originals = [parameter.detach().clone() for _, parameter in trainable]
    after_losses: Dict[str, float] = {}
    try:
        with torch.no_grad():
            for (_, parameter), gradient in zip(trainable, trace_gradient):
                parameter.add_(gradient.to(parameter.dtype), alpha=-float(eta))
        groups: Dict[int, List[str]] = {}
        for name, encoded in probes.items():
            groups.setdefault(id(encoded), []).append(name)
        for names in groups.values():
            encoded = probes[names[0]]
            flags = {bool(answer_only.get(name, False)) for name in names}
            if flags == {False, True}:
                rationale_loss, answer_loss = mean_probe_rationale_and_answer_loss(
                    model, encoded
                )
                for name in names:
                    after_losses[name] = (
                        answer_loss
                        if bool(answer_only.get(name, False))
                        else rationale_loss
                    )
            else:
                measured = mean_probe_loss(
                    model, encoded, answer_only=next(iter(flags))
                )
                for name in names:
                    after_losses[name] = measured
    finally:
        with torch.no_grad():
            for (_, parameter), original in zip(trainable, originals):
                parameter.copy_(original)
    after_hash = trainable_state_sha256(trainable)
    if before_hash != after_hash:
        raise RuntimeError("LoRA parameters were not restored exactly.")
    return {
        "probe_loss_before": {
            name: float(value) for name, value in base_probe_losses.items()
        },
        "probe_loss_after": after_losses,
        "ctv_exact": {
            name: float(base_probe_losses[name]) - after_losses[name] for name in probes
        },
        "parameters_restored_exactly": True,
        "state_sha256": after_hash,
    }


def choose_calibrated_eta(
    summaries: Sequence[Mapping[str, Any]],
    *,
    minimum_sign_agreement: float,
    minimum_adjacent_spearman: float,
    maximum_median_linearization_error: float,
) -> Mapping[str, Any]:
    passing = []
    for row in summaries:
        adjacent = row.get("adjacent_spearman")
        if (
            float(row["sign_agreement"]) >= minimum_sign_agreement
            and adjacent is not None
            and float(adjacent) >= minimum_adjacent_spearman
            and float(row["median_linearization_error"])
            <= maximum_median_linearization_error
        ):
            passing.append(row)
    if not passing:
        raise ValueError(
            "No relative update passed the registered calibration thresholds."
        )
    return max(passing, key=lambda row: float(row["relative_update"]))


def normalized_linearization_error(
    actual: float, predicted: float, epsilon: float = 1e-12
) -> float:
    return abs(float(actual) - float(predicted)) / max(
        abs(float(actual)), abs(float(predicted)), epsilon
    )


def within_question_fixed_effect_regression(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    predictors: Sequence[str],
    question_field: str = "problem_id",
) -> Dict[str, Any]:
    """OLS after demeaning outcome and predictors within question."""

    import numpy as np

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[question_field]), []).append(row)
    x_values: List[List[float]] = []
    y_values: List[float] = []
    cluster_labels: List[str] = []
    for question, group in grouped.items():
        if len(group) < 2:
            continue
        y_mean = sum(float(row[outcome]) for row in group) / len(group)
        x_means = {
            name: sum(float(row[name]) for row in group) / len(group)
            for name in predictors
        }
        for row in group:
            y_values.append(float(row[outcome]) - y_mean)
            x_values.append([float(row[name]) - x_means[name] for name in predictors])
            cluster_labels.append(question)
    if len(y_values) <= len(predictors):
        raise ValueError(
            "Insufficient within-question observations for fixed-effects OLS."
        )
    matrix = np.asarray(x_values, dtype=np.float64)
    target = np.asarray(y_values, dtype=np.float64)
    beta, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = target - matrix @ beta
    dof = len(target) - int(rank)
    sigma2 = float(residual @ residual / dof)
    bread = np.linalg.pinv(matrix.T @ matrix)
    naive_covariance = sigma2 * bread
    unique_clusters = sorted(set(cluster_labels))
    meat = np.zeros((len(predictors), len(predictors)), dtype=np.float64)
    for cluster in unique_clusters:
        indices = np.asarray(
            [index for index, label in enumerate(cluster_labels) if label == cluster]
        )
        score = matrix[indices].T @ residual[indices]
        meat += np.outer(score, score)
    cluster_count = len(unique_clusters)
    if cluster_count < 2:
        raise ValueError("Cluster-robust inference requires at least two questions.")
    correction = (cluster_count / (cluster_count - 1)) * (
        (len(target) - 1) / max(1, len(target) - int(rank))
    )
    clustered_covariance = correction * bread @ meat @ bread
    naive_standard_errors = np.sqrt(np.maximum(0.0, np.diag(naive_covariance)))
    clustered_standard_errors = np.sqrt(np.maximum(0.0, np.diag(clustered_covariance)))
    return {
        "coefficients": {name: float(value) for name, value in zip(predictors, beta)},
        "standard_errors": {
            name: float(value)
            for name, value in zip(predictors, clustered_standard_errors)
        },
        "question_clustered_standard_errors": {
            name: float(value)
            for name, value in zip(predictors, clustered_standard_errors)
        },
        "naive_standard_errors": {
            name: float(value) for name, value in zip(predictors, naive_standard_errors)
        },
        "observation_count": len(target),
        "question_count": len(grouped),
        "residual_degrees_of_freedom": dof,
        "design_rank": int(rank),
        "standard_error_type": "question_cluster_robust_cr1",
    }


def select_candidate_roles(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> Dict[str, Mapping[str, Any]]:
    if len(rows) < 4:
        raise ValueError("At least four candidate traces are required.")
    ranked = sorted(
        rows,
        key=lambda row: (int(row["solution_token_count"]), str(row["trace_id"])),
    )
    tail = max(1, math.ceil(0.2 * len(ranked)))
    short = ranked[tail // 2]
    long = ranked[len(ranked) - tail + (tail - 1) // 2]
    reserved = {str(short["trace_id"]), str(long["trace_id"])}
    median_length = median(int(row["solution_token_count"]) for row in ranked)
    middle = min(
        (row for row in ranked if str(row["trace_id"]) not in reserved),
        key=lambda row: (
            abs(int(row["solution_token_count"]) - median_length),
            str(row["trace_id"]),
        ),
    )
    reserved.add(str(middle["trace_id"]))
    random_row = min(ranked, key=lambda row: stable_hash(seed, row["trace_id"]))
    if str(random_row["trace_id"]) in reserved:
        random_row = min(
            (row for row in ranked if str(row["trace_id"]) not in reserved),
            key=lambda row: stable_hash(seed, row["trace_id"]),
        )
    return {
        "quantile_short": short,
        "median": middle,
        "quantile_long": long,
        "deterministic_random": random_row,
    }


def operation_count(text: str) -> int:
    return len(
        re.findall(
            r"(?:\+|-|\*|/|=|\bplus\b|\bminus\b|\btimes\b|\bdivid(?:e|ed)\b)",
            text.lower(),
        )
    )


def scas_forward_metrics(
    model: Any,
    tokenizer: Any,
    encoded: EncodedCompletion,
    *,
    layer_suffix: str,
    weight: float = 0.5,
) -> Dict[str, float]:
    """Compute the official final-up-projection SCAS score in one forward pass."""

    import torch
    import torch.nn.functional as functional

    from .trace_baselines import scas_blocks

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(layer_suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one SCAS target module ending in {layer_suffix!r}, found {len(matches)}."
        )
    stored: Dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        stored["activation"] = output[0] if isinstance(output, tuple) else output

    handle = matches[0][1].register_forward_hook(hook)
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device="cuda")
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
    finally:
        handle.remove()
    hidden = stored["activation"].squeeze(0).float()
    normalized = functional.normalize(hidden, p=2, dim=-1)
    special_ids = set(
        int(value) for value in (getattr(tokenizer, "all_special_ids", None) or [])
    )
    question_positions = [
        index
        for index in range(encoded.prompt_token_count)
        if encoded.input_ids[index] not in special_ids
    ]
    answer_positions = [
        index
        for index in range(encoded.prompt_token_count, len(encoded.input_ids))
        if encoded.input_ids[index] not in special_ids
    ]
    if not question_positions or not answer_positions:
        raise ValueError(f"SCAS token blocks are empty: {encoded.record_id}")
    question_index = torch.tensor(question_positions, dtype=torch.long, device="cuda")
    answer_index = torch.tensor(answer_positions, dtype=torch.long, device="cuda")
    question_states = normalized.index_select(0, question_index)
    answer_states = normalized.index_select(0, answer_index)
    answer_answer_similarity = float((answer_states @ answer_states.T).mean().cpu())
    answer_question_similarity = float((answer_states @ question_states.T).mean().cpu())
    logits = outputs.logits[0, :-1, :].float()
    labels = input_ids[0, 1:]
    token_nll = functional.cross_entropy(logits, labels, reduction="none")
    question_nll_positions = [index - 1 for index in question_positions if index > 0]
    answer_nll_positions = [index - 1 for index in answer_positions if index > 0]
    question_mean_nll = float(token_nll[question_nll_positions].mean().cpu())
    answer_mean_nll = float(token_nll[answer_nll_positions].mean().cpu())
    return {
        "scas_question_mean_nll": question_mean_nll,
        "scas_answer_mean_nll": answer_mean_nll,
        "scas_answer_answer_similarity": answer_answer_similarity,
        "scas_answer_question_similarity": answer_question_similarity,
        **scas_blocks(
            answer_mean_nll=answer_mean_nll,
            question_mean_nll=question_mean_nll,
            answer_answer_similarity=answer_answer_similarity,
            answer_question_similarity=answer_question_similarity,
            weight=weight,
        ),
    }


def trainable_state_sha256(trainable: Sequence[Tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in trainable:
        tensor = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clone_gradients(trainable: Sequence[Tuple[str, Any]]) -> List[Any]:
    return [
        parameter.detach().new_zeros(parameter.shape)
        if parameter.grad is None
        else parameter.grad.detach().clone()
        for _, parameter in trainable
    ]


def find_last_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern), -1, -1):
        if list(values[start : start + len(pattern)]) == list(pattern):
            return start
    return None
