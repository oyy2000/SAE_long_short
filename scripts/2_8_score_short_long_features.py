#!/usr/bin/env python3
"""Score one SAE for question-paired short-versus-long feature differences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
    runtime_metadata,
)
from length_budget_distill.sae_feature_analysis import (
    holm_adjust,
    paired_feature_statistics,
    relative_position_bin,
    select_discovery_features,
)


METRIC_NAMES = (
    "token_mean_activation",
    "token_activation_frequency",
    "maximum_activation",
    "first_64_token_mean_activation",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase2_short_long_feature_analysis_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    parser.add_argument("--layer-index", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.time()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    config = read_json(config_path)
    parent_config, chunks, checkpoint_path, evidence = _validate_inputs(
        config, args.layer_index, args.k
    )
    import torch
    from safetensors.torch import load_file

    from length_budget_distill.topk_sae import TopKSAE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense SAE encoding.")
    device = torch.device("cuda")
    checkpoint = load_file(str(checkpoint_path), device="cpu")
    hidden_size = int(parent_config["teacher"]["hidden_size"])
    feature_count = int(parent_config["sae"]["feature_count"])
    model = TopKSAE(hidden_size, feature_count, args.k)
    model.load_state_dict(
        {
            key: checkpoint[key]
            for key in (
                "decoder_weight",
                "encoder_weight",
                "encoder_bias",
                "decoder_bias",
            )
        }
    )
    model = model.to(device).eval()
    activation_mean = checkpoint["activation_mean"].to(device=device)
    activation_scale = float(checkpoint["activation_scale"].item())
    rows = _load_analysis_rows(_resolve(config["parent_sae"]["corpus_path"]))
    comparison = config["comparison"]
    token_config = config["token_analysis"]
    discovery_rows = [
        row for row in rows if row["question_split"] == comparison["discovery_split"]
    ]
    confirmation_rows = [
        row
        for row in rows
        if row["question_split"] == comparison["confirmation_split"]
    ]
    print(
        json.dumps(
            {
                "event": "analysis_start",
                "layer_index": args.layer_index,
                "k": args.k,
                "discovery_traces": len(discovery_rows),
                "confirmation_traces": len(confirmation_rows),
                "feature_count": feature_count,
            }
        ),
        flush=True,
    )
    discovery_metrics, discovery_observed, _, _, _ = _aggregate_split(
        torch=torch,
        load_file=load_file,
        model=model,
        activation_mean=activation_mean,
        activation_scale=activation_scale,
        chunks=chunks,
        rows=discovery_rows,
        feature_count=feature_count,
        split_name=str(comparison["discovery_split"]),
        first_n_tokens=int(token_config["first_n_tokens"]),
        encode_batch_size=int(config["runtime"]["encode_batch_size"]),
        selected_features=None,
        relative_bins=int(token_config["relative_position_bins"]),
    )
    _validate_observed_lengths(discovery_rows, discovery_observed)
    discovery_stats = _statistics_for_metrics(discovery_metrics, discovery_rows)
    discovery_rule = config["feature_discovery"]
    selected = select_discovery_features(
        discovery_stats[comparison["primary_metric"]],
        discovery_stats[comparison["position_control_metric"]],
        per_direction=int(discovery_rule["candidates_per_direction"]),
        minimum_prevalence=float(discovery_rule["minimum_trace_prevalence"]),
        maximum_bh_q=float(discovery_rule["maximum_dev_bh_q_value"]),
        minimum_abs_paired_d=float(
            discovery_rule["minimum_dev_absolute_paired_d"]
        ),
    )
    selected_feature_ids = [int(row["feature_id"]) for row in selected]
    discovery_selected_metrics = {
        name: values[:, selected_feature_ids].copy()
        for name, values in discovery_metrics.items()
    }
    del discovery_metrics
    confirmation_metrics, confirmation_observed, events, token_frequency, relative = (
        _aggregate_split(
            torch=torch,
            load_file=load_file,
            model=model,
            activation_mean=activation_mean,
            activation_scale=activation_scale,
            chunks=chunks,
            rows=confirmation_rows,
            feature_count=feature_count,
            split_name=str(comparison["confirmation_split"]),
            first_n_tokens=int(token_config["first_n_tokens"]),
            encode_batch_size=int(config["runtime"]["encode_batch_size"]),
            selected_features=selected_feature_ids,
            relative_bins=int(token_config["relative_position_bins"]),
        )
    )
    _validate_observed_lengths(confirmation_rows, confirmation_observed)
    confirmation_stats = _statistics_for_metrics(
        confirmation_metrics, confirmation_rows
    )
    confirmation_selected_metrics = {
        name: values[:, selected_feature_ids].copy()
        for name, values in confirmation_metrics.items()
    }
    primary_test_p = [
        float(confirmation_stats[comparison["primary_metric"]]["p_value"][feature])
        for feature in selected_feature_ids
    ]
    adjusted = holm_adjust(primary_test_p)
    for index, candidate in enumerate(selected):
        feature_id = int(candidate["feature_id"])
        expected_sign = 1 if candidate["direction"] == "short" else -1
        candidate["confirmation_holm_p_value"] = float(adjusted[index])
        candidate["metrics"] = {}
        for metric_name in METRIC_NAMES:
            candidate["metrics"][metric_name] = {
                "dev": _feature_stat_row(discovery_stats[metric_name], feature_id),
                "test": _feature_stat_row(confirmation_stats[metric_name], feature_id),
            }
        test_primary_d = float(
            confirmation_stats[comparison["primary_metric"]]["paired_d"][feature_id]
        )
        test_early_d = float(
            confirmation_stats[comparison["position_control_metric"]]["paired_d"][
                feature_id
            ]
        )
        candidate["direction_replicated"] = bool(expected_sign * test_primary_d > 0)
        candidate["first_64_direction_replicated"] = bool(
            expected_sign * test_early_d > 0
        )
        candidate["confirmed"] = bool(
            candidate["direction_replicated"]
            and candidate["first_64_direction_replicated"]
            and candidate["confirmation_holm_p_value"]
            <= float(discovery_rule["confirmation_holm_alpha"])
            and abs(test_primary_d)
            >= float(discovery_rule["minimum_test_absolute_paired_d"])
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    statistics_path = output_dir / "feature_statistics.csv"
    _write_feature_statistics(
        statistics_path, feature_count, discovery_stats, confirmation_stats
    )
    candidates_path = output_dir / "discovered_features.json"
    write_json_exclusive(
        candidates_path,
        {
            "layer_index": args.layer_index,
            "k": args.k,
            "selection_split": comparison["discovery_split"],
            "confirmation_split": comparison["confirmation_split"],
            "positive_effect_means": "short-associated",
            "negative_effect_means": "long-associated",
            "candidates": selected,
        },
    )
    trace_metrics_path = output_dir / "selected_trace_metrics.csv"
    _write_selected_trace_metrics(
        trace_metrics_path,
        discovery_rows,
        confirmation_rows,
        selected,
        discovery_selected_metrics,
        confirmation_selected_metrics,
    )
    event_path = output_dir / "selected_feature_token_events.jsonl"
    _write_jsonl_exclusive(event_path, events)
    token_frequency_path = output_dir / "test_token_frequency.csv"
    _write_token_frequency(token_frequency_path, token_frequency)
    relative_path = output_dir / "selected_feature_relative_position.csv"
    _write_relative_metrics(relative_path, confirmation_rows, selected, relative)
    confirmed_count = sum(bool(row["confirmed"]) for row in selected)
    summary = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "layer_index": args.layer_index,
        "k": args.k,
        "feature_count": feature_count,
        "discovery_trace_count": len(discovery_rows),
        "confirmation_trace_count": len(confirmation_rows),
        "discovery_question_count": len(
            {str(row["problem_id"]) for row in discovery_rows}
        ),
        "confirmation_question_count": len(
            {str(row["problem_id"]) for row in confirmation_rows}
        ),
        "candidate_count": len(selected),
        "discovery_gate_pass_count": sum(
            bool(row["passes_discovery_gate"]) for row in selected
        ),
        "heldout_confirmed_count": confirmed_count,
        "short_confirmed_count": sum(
            bool(row["confirmed"]) and row["direction"] == "short"
            for row in selected
        ),
        "long_confirmed_count": sum(
            bool(row["confirmed"]) and row["direction"] == "long"
            for row in selected
        ),
        "elapsed_seconds": time.time() - started,
        "input_evidence": evidence,
        "artifacts": {
            path.name: {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                statistics_path,
                candidates_path,
                trace_metrics_path,
                event_path,
                token_frequency_path,
                relative_path,
            )
        },
        "runtime": runtime_metadata(("python", "torch", "numpy", "scipy", "safetensors")),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "library_code_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/sae_feature_analysis.py"
        ),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = output_dir / "scoring_summary.json"
    write_json_exclusive(summary_path, summary)
    (output_dir / "FEATURE_SCORING_COMPLETE").write_text(
        f"status=complete\nconfig_hash={summary['config_hash']}\n"
        f"summary_sha256={file_sha256(summary_path)}\n"
        f"layer_index={args.layer_index}\nk={args.k}\n"
        f"heldout_confirmed_count={confirmed_count}\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _aggregate_split(
    *,
    torch,
    load_file,
    model,
    activation_mean,
    activation_scale: float,
    chunks: list[Path],
    rows: list[dict[str, Any]],
    feature_count: int,
    split_name: str,
    first_n_tokens: int,
    encode_batch_size: int,
    selected_features: list[int] | None,
    relative_bins: int,
):
    corpus_indices = [int(row["corpus_index"]) for row in rows]
    max_index = max(corpus_indices)
    local_lookup = torch.full((max_index + 1,), -1, dtype=torch.int64)
    local_lookup[torch.tensor(corpus_indices, dtype=torch.int64)] = torch.arange(
        len(rows), dtype=torch.int64
    )
    sums = torch.zeros((len(rows), feature_count), dtype=torch.float32)
    counts = torch.zeros_like(sums)
    maxima = torch.zeros_like(sums)
    first_sums = torch.zeros_like(sums)
    observed = torch.zeros(len(rows), dtype=torch.int64)
    selected_lookup = None
    events: list[dict[str, Any]] = []
    token_frequency: Counter[tuple[str, int]] = Counter()
    relative_sums = None
    if selected_features is not None:
        selected_lookup = torch.full((feature_count,), -1, dtype=torch.int64)
        selected_lookup[torch.tensor(selected_features)] = torch.arange(
            len(selected_features), dtype=torch.int64
        )
        relative_sums = np.zeros(
            (len(rows), len(selected_features), relative_bins), dtype=np.float32
        )
    row_lengths = np.asarray(
        [int(row["solution_token_count"]) for row in rows], dtype=np.int64
    )
    processed_tokens = 0
    for chunk_number, chunk_path in enumerate(chunks, start=1):
        tensors = load_file(str(chunk_path), device="cpu")
        trace_indices = tensors["trace_indices"].long()
        in_range = trace_indices < len(local_lookup)
        local = torch.full_like(trace_indices, -1)
        local[in_range] = local_lookup[trace_indices[in_range]]
        keep = local >= 0
        if not bool(keep.any()):
            continue
        activations = tensors["activations"][keep]
        local = local[keep]
        positions = tensors["positions"][keep].long()
        token_ids = tensors["token_ids"][keep].long()
        observed.index_add_(0, local, torch.ones_like(local, dtype=torch.int64))
        if selected_features is not None:
            labels = [str(rows[int(index)]["analysis_length_label"]) for index in local]
            token_frequency.update(
                (label, int(token_id))
                for label, token_id in zip(labels, token_ids.tolist())
            )
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
                values, indices, _ = model.encode(inputs)
            values = values.float().cpu()
            indices = indices.long().cpu()
            _update_dense_aggregates(
                sums,
                counts,
                maxima,
                first_sums,
                cpu_local,
                cpu_positions,
                indices,
                values,
                first_n_tokens,
            )
            if selected_lookup is not None and relative_sums is not None:
                _collect_selected_events(
                    rows,
                    selected_lookup,
                    relative_sums,
                    events,
                    cpu_local,
                    cpu_positions,
                    cpu_token_ids,
                    indices,
                    values,
                    row_lengths,
                    relative_bins,
                    split_name,
                )
            processed_tokens += stop - start
        if chunk_number % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "encoding_progress",
                        "split": split_name,
                        "chunk": chunk_number,
                        "chunks": len(chunks),
                        "selected_tokens": processed_tokens,
                    }
                ),
                flush=True,
            )
    lengths = observed.clamp_min(1).float().unsqueeze(1)
    first_denominator = torch.minimum(
        observed, torch.full_like(observed, first_n_tokens)
    ).clamp_min(1).float().unsqueeze(1)
    metrics = {
        "token_mean_activation": (sums / lengths).numpy(),
        "token_activation_frequency": (counts / lengths).numpy(),
        "maximum_activation": maxima.numpy(),
        "first_64_token_mean_activation": (first_sums / first_denominator).numpy(),
    }
    if relative_sums is not None:
        denominators = np.zeros((len(rows), relative_bins), dtype=np.int64)
        for row_index, length in enumerate(observed.tolist()):
            for position in range(int(length)):
                denominators[
                    row_index,
                    relative_position_bin(position, int(length), relative_bins),
                ] += 1
        relative_means = relative_sums / np.maximum(denominators[:, None, :], 1)
    else:
        relative_means = None
    return metrics, observed.numpy(), events, token_frequency, relative_means


def _update_dense_aggregates(
    sums,
    counts,
    maxima,
    first_sums,
    local,
    positions,
    indices,
    values,
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
            maxima[row_index].scatter_reduce_(
                0, active_ids, active_values, reduce="amax", include_self=True
            )
        early_tokens = positions[left:right] < first_n_tokens
        if bool(early_tokens.any()):
            early_ids = indices[left:right][early_tokens].reshape(-1)
            early_values = values[left:right][early_tokens].reshape(-1)
            early_positive = early_values > 0
            if bool(early_positive.any()):
                first_sums[row_index].scatter_add_(
                    0, early_ids[early_positive], early_values[early_positive]
                )


def _collect_selected_events(
    rows,
    selected_lookup,
    relative_sums,
    events,
    local,
    positions,
    token_ids,
    indices,
    values,
    row_lengths,
    relative_bins,
    split_name,
) -> None:
    slots = selected_lookup[indices]
    mask = (slots >= 0) & (values > 0)
    token_offsets, topk_offsets = mask.nonzero(as_tuple=True)
    for token_offset, topk_offset in zip(
        token_offsets.tolist(), topk_offsets.tolist()
    ):
        row_index = int(local[token_offset])
        position = int(positions[token_offset])
        slot = int(slots[token_offset, topk_offset])
        feature_id = int(indices[token_offset, topk_offset])
        value = float(values[token_offset, topk_offset])
        length = int(row_lengths[row_index])
        bin_index = relative_position_bin(position, length, relative_bins)
        relative_sums[row_index, slot, bin_index] += value
        row = rows[row_index]
        events.append(
            {
                "split": split_name,
                "corpus_index": int(row["corpus_index"]),
                "trace_id": str(row["trace_id"]),
                "problem_id": str(row["problem_id"]),
                "analysis_length_label": str(row["analysis_length_label"]),
                "feature_id": feature_id,
                "position": position,
                "relative_position_bin": bin_index,
                "token_id": int(token_ids[token_offset]),
                "activation": value,
            }
        )


def _statistics_for_metrics(metrics, rows):
    frequency = metrics["token_activation_frequency"]
    return {
        name: paired_feature_statistics(values, rows, frequency_values=frequency)
        for name, values in metrics.items()
    }


def _feature_stat_row(statistics, feature_id: int) -> dict[str, float]:
    return {
        key: float(statistics[key][feature_id])
        for key in (
            "effect",
            "ci_low",
            "ci_high",
            "paired_d",
            "t_statistic",
            "p_value",
            "bh_q_value",
            "prevalence",
        )
    }


def _write_feature_statistics(path, feature_count, discovery, confirmation) -> None:
    fieldnames = ["feature_id"]
    statistic_names = (
        "effect",
        "ci_low",
        "ci_high",
        "paired_d",
        "p_value",
        "bh_q_value",
        "prevalence",
    )
    for split_name in ("dev", "test"):
        for metric_name in METRIC_NAMES:
            fieldnames.extend(
                f"{split_name}_{metric_name}_{statistic}"
                for statistic in statistic_names
            )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_id in range(feature_count):
            row: dict[str, Any] = {"feature_id": feature_id}
            for split_name, collection in (
                ("dev", discovery),
                ("test", confirmation),
            ):
                for metric_name in METRIC_NAMES:
                    statistics = collection[metric_name]
                    for statistic in statistic_names:
                        row[f"{split_name}_{metric_name}_{statistic}"] = float(
                            statistics[statistic][feature_id]
                        )
            writer.writerow(row)


def _write_selected_trace_metrics(
    path,
    discovery_rows,
    confirmation_rows,
    selected,
    discovery_metrics,
    confirmation_metrics,
) -> None:
    feature_ids = [int(row["feature_id"]) for row in selected]
    fieldnames = [
        "question_split",
        "problem_id",
        "trace_id",
        "corpus_index",
        "analysis_length_label",
        "solution_token_count",
    ]
    fieldnames.extend(
        f"feature_{feature_id}_{metric_name}"
        for feature_id in feature_ids
        for metric_name in METRIC_NAMES
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rows, metrics in (
            (discovery_rows, discovery_metrics),
            (confirmation_rows, confirmation_metrics),
        ):
            for row_index, source in enumerate(rows):
                output = {name: source[name] for name in fieldnames[:6]}
                for slot, feature_id in enumerate(feature_ids):
                    for metric_name in METRIC_NAMES:
                        output[f"feature_{feature_id}_{metric_name}"] = float(
                            metrics[metric_name][row_index, slot]
                        )
                writer.writerow(output)


def _write_relative_metrics(path, rows, selected, relative) -> None:
    if relative is None:
        raise ValueError("Relative-position metrics were not computed.")
    fieldnames = [
        "problem_id",
        "trace_id",
        "corpus_index",
        "analysis_length_label",
        "feature_id",
        "direction",
        "relative_position_bin",
        "mean_activation",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows):
            for slot, candidate in enumerate(selected):
                for bin_index in range(relative.shape[2]):
                    writer.writerow(
                        {
                            "problem_id": row["problem_id"],
                            "trace_id": row["trace_id"],
                            "corpus_index": row["corpus_index"],
                            "analysis_length_label": row["analysis_length_label"],
                            "feature_id": candidate["feature_id"],
                            "direction": candidate["direction"],
                            "relative_position_bin": bin_index,
                            "mean_activation": float(
                                relative[row_index, slot, bin_index]
                            ),
                        }
                    )


def _write_token_frequency(path, counter) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["analysis_length_label", "token_id", "count"]
        )
        writer.writeheader()
        for (label, token_id), count in sorted(counter.items()):
            writer.writerow(
                {
                    "analysis_length_label": label,
                    "token_id": token_id,
                    "count": count,
                }
            )


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_observed_lengths(rows, observed) -> None:
    mismatches = []
    for row, count in zip(rows, observed.tolist()):
        expected = int(row["solution_token_count"])
        if int(count) != expected:
            mismatches.append((row["trace_id"], expected, int(count)))
    if mismatches:
        raise ValueError(f"Activation token-count mismatches: {mismatches[:5]}")


def _load_analysis_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
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


def _validate_inputs(config, layer_index: int, k: int):
    parent = config["parent_sae"]
    parent_config_path = _resolve(parent["config_path"])
    parent_marker_path = _resolve(parent["completion_marker_path"])
    marker = read_key_value_marker(parent_marker_path)
    if marker.get("status") != "passed" or marker.get("validated_sae_count") != "6":
        raise RuntimeError("Parent SAE pilot is not complete.")
    if file_sha256(parent_marker_path) != parent["completion_marker_sha256"]:
        raise ValueError("Parent completion marker changed after protocol freeze.")
    if file_sha256(parent_config_path) != parent["config_sha256"]:
        raise ValueError("Parent SAE config changed after protocol freeze.")
    parent_config = read_json(parent_config_path)
    if canonical_sha256(parent_config) != parent["config_hash"]:
        raise ValueError("Parent SAE config hash mismatch.")
    if layer_index not in [
        int(value)
        for value in parent_config["activation_extraction"]["layer_indices_zero_based"]
    ] or k not in [int(value) for value in parent_config["sae"]["k_values"]]:
        raise ValueError("Requested SAE is outside the registered parent matrix.")
    training_dir = _resolve(parent["training_root"]) / f"layer_{layer_index:02d}_k_{k:03d}"
    training_marker_path = training_dir / "SAE_TRAINING_COMPLETE"
    metrics_path = training_dir / "training_metrics.json"
    training_marker = read_key_value_marker(training_marker_path)
    if training_marker.get("status") != "complete":
        raise RuntimeError("SAE training marker is incomplete.")
    if training_marker.get("training_metrics_sha256") != file_sha256(metrics_path):
        raise ValueError("SAE training metrics hash mismatch.")
    metrics = read_json(metrics_path)
    checkpoint_path = Path(metrics["model_path"])
    if training_marker.get("model_sha256") != file_sha256(checkpoint_path):
        raise ValueError("SAE checkpoint hash mismatch.")
    activation_root = _resolve(parent["activation_root"])
    chunks = []
    activation_evidence = []
    for shard_dir in sorted(activation_root.glob("shard_*_of_*")):
        manifest_path = shard_dir / "activation_manifest.json"
        shard_marker_path = shard_dir / "ACTIVATIONS_COMPLETE"
        shard_marker = read_key_value_marker(shard_marker_path)
        if shard_marker.get("status") != "complete" or shard_marker.get(
            "manifest_sha256"
        ) != file_sha256(manifest_path):
            raise ValueError(f"Activation shard marker mismatch: {shard_dir}")
        manifest = read_json(manifest_path)
        layer = next(
            row
            for row in manifest["layers"]
            if int(row["layer_index"]) == layer_index
        )
        for chunk in layer["chunks"]:
            path = Path(chunk["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            chunks.append(path)
        activation_evidence.append(
            {
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "marker_path": str(shard_marker_path),
                "marker_sha256": file_sha256(shard_marker_path),
                "layer_token_count": int(layer["token_count"]),
            }
        )
    return parent_config, chunks, checkpoint_path, {
        "parent_marker_path": str(parent_marker_path),
        "parent_marker_sha256": file_sha256(parent_marker_path),
        "training_marker_path": str(training_marker_path),
        "training_marker_sha256": file_sha256(training_marker_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "activation_shards": activation_evidence,
    }


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
