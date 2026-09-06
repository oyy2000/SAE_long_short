"""Reusable scoring utilities for SAE training-distribution ablations."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .sae_feature_analysis import paired_feature_statistics, relative_position_bin


METRIC_NAMES = (
    "token_mean_activation",
    "token_activation_frequency",
    "first_64_token_mean_activation",
)


def load_analysis_rows(corpus_path: Path) -> list[dict[str, Any]]:
    import json

    rows = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                bool(row["is_correct"])
                and row["analysis_length_label"] in {"short", "long"}
                and row["question_split"] in {"dev", "test"}
            ):
                rows.append(row)
    rows.sort(key=lambda row: int(row["corpus_index"]))
    return rows


def load_topk_sae(checkpoint_path: Path, *, input_dim: int, feature_count: int, k: int, device: Any):
    from safetensors.torch import load_file

    from .topk_sae import TopKSAE

    checkpoint = load_file(str(checkpoint_path), device="cpu")
    model = TopKSAE(input_dim, feature_count, k)
    model.load_state_dict(
        {
            name: checkpoint[name]
            for name in (
                "decoder_weight",
                "encoder_weight",
                "encoder_bias",
                "decoder_bias",
            )
        }
    )
    return (
        model.to(device).eval(),
        checkpoint["activation_mean"].to(device=device),
        float(checkpoint["activation_scale"].item()),
    )


def aggregate_trace_features(
    *,
    torch: Any,
    load_file: Any,
    model: Any,
    activation_mean: Any,
    activation_scale: float,
    chunk_paths: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    feature_count: int,
    first_n_tokens: int,
    encode_batch_size: int,
    selected_features: Sequence[int] | None = None,
    relative_bins: int = 5,
    progress_prefix: str = "",
) -> dict[str, Any]:
    """Aggregate dense per-trace feature metrics and reconstruction diagnostics."""

    corpus_indices = [int(row["corpus_index"]) for row in rows]
    lookup = torch.full((max(corpus_indices) + 1,), -1, dtype=torch.int64)
    lookup[torch.tensor(corpus_indices, dtype=torch.int64)] = torch.arange(
        len(rows), dtype=torch.int64
    )
    sums = torch.zeros((len(rows), feature_count), dtype=torch.float32)
    counts = torch.zeros_like(sums)
    first_sums = torch.zeros_like(sums)
    observed = torch.zeros(len(rows), dtype=torch.int64)
    trace_squared_error = torch.zeros(len(rows), dtype=torch.float64)
    trace_element_count = torch.zeros(len(rows), dtype=torch.int64)
    selected_lookup = None
    relative_sums = None
    token_mass: Counter[tuple[int, str, int]] = Counter()
    activation_mass: Counter[tuple[int, str]] = Counter()
    if selected_features is not None:
        selected_lookup = torch.full((feature_count,), -1, dtype=torch.int64)
        selected_lookup[torch.tensor(list(selected_features), dtype=torch.int64)] = torch.arange(
            len(selected_features), dtype=torch.int64
        )
        relative_sums = np.zeros(
            (len(rows), len(selected_features), relative_bins), dtype=np.float32
        )
    row_lengths = np.asarray(
        [int(row["solution_token_count"]) for row in rows], dtype=np.int64
    )
    global_accumulators = {
        "full": _reconstruction_accumulator(model.input_dim),
        "first_64": _reconstruction_accumulator(model.input_dim),
    }
    selected_token_count = 0
    for chunk_number, chunk_path in enumerate(chunk_paths, start=1):
        tensors = load_file(str(chunk_path), device="cpu")
        trace_indices = tensors["trace_indices"].long()
        in_range = trace_indices < len(lookup)
        local = torch.full_like(trace_indices, -1)
        local[in_range] = lookup[trace_indices[in_range]]
        keep = local >= 0
        if not bool(keep.any()):
            continue
        activations = tensors["activations"][keep]
        local = local[keep]
        positions = tensors["positions"][keep].long()
        token_ids = tensors["token_ids"][keep].long()
        observed.index_add_(0, local, torch.ones_like(local, dtype=torch.int64))
        for start in range(0, len(local), encode_batch_size):
            stop = min(len(local), start + encode_batch_size)
            cpu_local = local[start:stop]
            cpu_positions = positions[start:stop]
            cpu_token_ids = token_ids[start:stop]
            inputs = activations[start:stop].to(
                device=activation_mean.device, non_blocking=True
            )
            inputs = (inputs.float() - activation_mean) * activation_scale
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                reconstruction, values, indices, _ = model(inputs)
            errors = (reconstruction.float() - inputs).cpu()
            cpu_inputs = inputs.float().cpu()
            values = values.float().cpu()
            indices = indices.long().cpu()
            _update_feature_aggregates(
                sums,
                counts,
                first_sums,
                cpu_local,
                cpu_positions,
                indices,
                values,
                first_n_tokens,
            )
            _update_reconstruction(
                global_accumulators,
                trace_squared_error,
                trace_element_count,
                cpu_inputs,
                errors,
                cpu_local,
                cpu_positions,
                first_n_tokens,
            )
            if selected_lookup is not None and relative_sums is not None:
                _collect_selected(
                    rows,
                    selected_lookup,
                    relative_sums,
                    token_mass,
                    activation_mass,
                    cpu_local,
                    cpu_positions,
                    cpu_token_ids,
                    indices,
                    values,
                    row_lengths,
                    relative_bins,
                )
            selected_token_count += stop - start
        if chunk_number % 25 == 0:
            print(
                f"{progress_prefix} chunks={chunk_number}/{len(chunk_paths)} "
                f"tokens={selected_token_count}",
                flush=True,
            )
    expected = np.asarray(row_lengths)
    observed_numpy = observed.numpy()
    if not np.array_equal(expected, observed_numpy):
        mismatch = np.flatnonzero(expected != observed_numpy)[:5]
        raise ValueError(
            f"Activation lengths mismatch: "
            f"{[(rows[i]['trace_id'], int(expected[i]), int(observed_numpy[i])) for i in mismatch]}"
        )
    lengths = observed.clamp_min(1).float().unsqueeze(1)
    first_denominator = torch.minimum(
        observed, torch.full_like(observed, first_n_tokens)
    ).clamp_min(1).float().unsqueeze(1)
    metrics = {
        "token_mean_activation": (sums / lengths).numpy(),
        "token_activation_frequency": (counts / lengths).numpy(),
        "first_64_token_mean_activation": (first_sums / first_denominator).numpy(),
    }
    reconstruction = {
        name: _finalize_reconstruction(accumulator)
        for name, accumulator in global_accumulators.items()
    }
    per_trace_mse = np.divide(
        trace_squared_error.numpy(),
        np.maximum(trace_element_count.numpy(), 1),
    )
    reconstruction["sequence_normalized_mse"] = float(per_trace_mse.mean())
    reconstruction["sequence_normalized_mse_short"] = float(
        per_trace_mse[
            np.asarray([row["analysis_length_label"] == "short" for row in rows])
        ].mean()
    )
    reconstruction["sequence_normalized_mse_long"] = float(
        per_trace_mse[
            np.asarray([row["analysis_length_label"] == "long" for row in rows])
        ].mean()
    )
    if relative_sums is not None:
        denominators = np.zeros((len(rows), relative_bins), dtype=np.int64)
        for row_index, length in enumerate(observed_numpy.tolist()):
            for position in range(int(length)):
                denominators[
                    row_index,
                    relative_position_bin(position, int(length), relative_bins),
                ] += 1
        relative_means = relative_sums / np.maximum(denominators[:, None, :], 1)
    else:
        relative_means = None
    return {
        "metrics": metrics,
        "reconstruction": reconstruction,
        "token_mass": token_mass,
        "activation_mass": activation_mass,
        "relative_means": relative_means,
    }


def statistics_for_metrics(
    metrics: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    frequency = metrics["token_activation_frequency"]
    return {
        name: paired_feature_statistics(values, rows, frequency_values=frequency)
        for name, values in metrics.items()
    }


def mutual_decoder_matches(
    *,
    left_name: str,
    right_name: str,
    left_candidates: Sequence[Mapping[str, Any]],
    right_candidates: Sequence[Mapping[str, Any]],
    left_decoders: np.ndarray,
    right_decoders: np.ndarray,
    left_test_activations: np.ndarray,
    right_test_activations: np.ndarray,
    minimum_decoder_cosine: float,
    minimum_activation_correlation: float,
    require_same_direction: bool,
) -> list[dict[str, Any]]:
    """Match candidate dictionaries by mutual decoder nearest neighbors."""

    left = np.asarray(left_decoders, dtype=np.float64)
    right = np.asarray(right_decoders, dtype=np.float64)
    if left.shape[0] != len(left_candidates) or right.shape[0] != len(right_candidates):
        raise ValueError("Decoder rows must align with candidate rows.")
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    cosine = left @ right.T
    left_choice = cosine.argmax(axis=1)
    right_choice = cosine.argmax(axis=0)
    rows = []
    for left_slot, right_slot in enumerate(left_choice.tolist()):
        left_candidate = left_candidates[left_slot]
        right_candidate = right_candidates[right_slot]
        correlation = _safe_correlation(
            left_test_activations[:, left_slot],
            right_test_activations[:, right_slot],
        )
        mutual = bool(right_choice[right_slot] == left_slot)
        same_direction = bool(
            left_candidate["direction"] == right_candidate["direction"]
        )
        decoder_cosine = float(cosine[left_slot, right_slot])
        stable = bool(
            mutual
            and decoder_cosine >= minimum_decoder_cosine
            and correlation >= minimum_activation_correlation
            and (same_direction or not require_same_direction)
        )
        rows.append(
            {
                "left_condition": left_name,
                "right_condition": right_name,
                "left_slot": left_slot,
                "right_slot": right_slot,
                "left_feature_id": int(left_candidate["feature_id"]),
                "right_feature_id": int(right_candidate["feature_id"]),
                "left_direction": str(left_candidate["direction"]),
                "right_direction": str(right_candidate["direction"]),
                "decoder_cosine": decoder_cosine,
                "test_trace_activation_correlation": correlation,
                "mutual_nearest_neighbor": mutual,
                "same_direction": same_direction,
                "stable_match": stable,
            }
        )
    return rows


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Activation vectors must align.")
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _update_feature_aggregates(
    sums: Any,
    counts: Any,
    first_sums: Any,
    local: Any,
    positions: Any,
    indices: Any,
    values: Any,
    first_n_tokens: int,
) -> None:
    boundaries = [0]
    boundaries.extend(
        (local[1:] != local[:-1]).nonzero(as_tuple=False).flatten().add(1).tolist()
    )
    boundaries.append(len(local))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        row_index = int(local[left])
        feature_ids = indices[left:right].reshape(-1)
        feature_values = values[left:right].reshape(-1)
        positive = feature_values > 0
        if bool(positive.any()):
            active_ids = feature_ids[positive]
            active_values = feature_values[positive]
            sums[row_index].scatter_add_(0, active_ids, active_values)
            counts[row_index].scatter_add_(
                0, active_ids, active_values.new_ones(active_values.shape)
            )
        early = positions[left:right] < first_n_tokens
        if bool(early.any()):
            early_ids = indices[left:right][early].reshape(-1)
            early_values = values[left:right][early].reshape(-1)
            positive = early_values > 0
            if bool(positive.any()):
                first_sums[row_index].scatter_add_(
                    0, early_ids[positive], early_values[positive]
                )


def _collect_selected(
    rows: Sequence[Mapping[str, Any]],
    selected_lookup: Any,
    relative_sums: np.ndarray,
    token_mass: Counter,
    activation_mass: Counter,
    local: Any,
    positions: Any,
    token_ids: Any,
    indices: Any,
    values: Any,
    row_lengths: np.ndarray,
    relative_bins: int,
) -> None:
    slots = selected_lookup[indices]
    token_offsets, topk_offsets = ((slots >= 0) & (values > 0)).nonzero(
        as_tuple=True
    )
    for token_offset, topk_offset in zip(
        token_offsets.tolist(), topk_offsets.tolist()
    ):
        row_index = int(local[token_offset])
        slot = int(slots[token_offset, topk_offset])
        feature_id = int(indices[token_offset, topk_offset])
        position = int(positions[token_offset])
        value = float(values[token_offset, topk_offset])
        label = str(rows[row_index]["analysis_length_label"])
        relative_sums[
            row_index,
            slot,
            relative_position_bin(
                position, int(row_lengths[row_index]), relative_bins
            ),
        ] += value
        token_mass[(feature_id, label, int(token_ids[token_offset]))] += value
        activation_mass[(feature_id, label)] += value


def _reconstruction_accumulator(input_dim: int) -> dict[str, Any]:
    import torch

    return {
        "squared_error": 0.0,
        "input_squared": 0.0,
        "input_sum": torch.zeros(input_dim, dtype=torch.float64),
        "token_count": 0,
    }


def _update_reconstruction(
    accumulators: dict[str, dict[str, Any]],
    trace_squared_error: Any,
    trace_element_count: Any,
    inputs: Any,
    errors: Any,
    local: Any,
    positions: Any,
    first_n_tokens: int,
) -> None:
    squared_per_token = errors.double().square().sum(dim=1)
    element_count = inputs.shape[1]
    trace_squared_error.index_add_(0, local, squared_per_token)
    trace_element_count.index_add_(
        0,
        local,
        trace_element_count.new_full((len(local),), element_count),
    )
    for name, mask in (
        ("full", np.ones(len(local), dtype=bool)),
        ("first_64", (positions < first_n_tokens).numpy()),
    ):
        if not np.any(mask):
            continue
        selected = np.flatnonzero(mask)
        accumulator = accumulators[name]
        accumulator["squared_error"] += float(
            squared_per_token[selected].sum().item()
        )
        accumulator["input_squared"] += float(
            inputs[selected].double().square().sum().item()
        )
        accumulator["input_sum"] += inputs[selected].double().sum(dim=0)
        accumulator["token_count"] += len(selected)


def _finalize_reconstruction(accumulator: Mapping[str, Any]) -> dict[str, float | int]:
    token_count = int(accumulator["token_count"])
    input_dim = len(accumulator["input_sum"])
    squared_error = float(accumulator["squared_error"])
    centered_energy = float(accumulator["input_squared"]) - float(
        accumulator["input_sum"].square().sum().item()
    ) / max(1, token_count)
    return {
        "token_count": token_count,
        "mse": squared_error / max(1, token_count * input_dim),
        "explained_variance": 1.0 - squared_error / max(centered_energy, 1e-12),
    }
