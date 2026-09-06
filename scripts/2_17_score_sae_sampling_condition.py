#!/usr/bin/env python3
"""Score one SAE sampling condition on common full-sequence held-out traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from length_budget_distill.sae_ablation import (
    METRIC_NAMES,
    aggregate_trace_features,
    load_analysis_rows,
    load_topk_sae,
    statistics_for_metrics,
)
from length_budget_distill.sae_feature_analysis import holm_adjust, select_discovery_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "results/phase2_sae_sampling_ablation_v1/exploratory/"
            "protocol/frozen_protocol.json"
        ),
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    started = time.time()
    config_path = _resolve(args.config)
    config = read_json(config_path)
    condition = _condition(config, args.condition)
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else _resolve(config["sampling_ablation"]["outputs"]["result_root"])
        / "condition_scores"
        / args.condition
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    evidence = _validate_protocol(config_path, config)
    artifacts = _condition_artifacts(config, condition)
    evidence.update(_validate_condition_artifacts(config, condition, artifacts))

    import torch
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for common-set SAE scoring.")
    device = torch.device("cuda")
    primary = config["sampling_ablation"]["primary_sae"]
    model, activation_mean, activation_scale = load_topk_sae(
        artifacts["checkpoint_path"],
        input_dim=int(config["teacher"]["hidden_size"]),
        feature_count=int(primary["feature_count"]),
        k=int(primary["k"]),
        device=device,
    )
    rows = load_analysis_rows(
        _resolve(config["sampling_ablation"]["parent_sae"]["corpus_path"])
    )
    evaluation = config["sampling_ablation"]["evaluation"]
    dev_rows = [
        row for row in rows if row["question_split"] == evaluation["discovery_split"]
    ]
    test_rows = [
        row
        for row in rows
        if row["question_split"] == evaluation["confirmation_split"]
    ]
    print(
        json.dumps(
            {
                "event": "condition_scoring_start",
                "condition": args.condition,
                "dev_traces": len(dev_rows),
                "test_traces": len(test_rows),
                "chunks": len(artifacts["chunk_paths"]),
            }
        ),
        flush=True,
    )
    common = {
        "torch": torch,
        "load_file": load_file,
        "model": model,
        "activation_mean": activation_mean,
        "activation_scale": activation_scale,
        "chunk_paths": artifacts["chunk_paths"],
        "feature_count": int(primary["feature_count"]),
        "first_n_tokens": 64,
        "encode_batch_size": int(evaluation["common_eval_encode_batch_size"]),
        "relative_bins": int(evaluation["relative_position_bins"]),
    }
    dev = aggregate_trace_features(
        **common,
        rows=dev_rows,
        selected_features=None,
        progress_prefix=f"{args.condition}/dev",
    )
    dev_stats = statistics_for_metrics(dev["metrics"], dev_rows)
    candidates = select_discovery_features(
        dev_stats[evaluation["primary_metric"]],
        dev_stats[evaluation["early_metric"]],
        per_direction=int(evaluation["candidate_features_per_direction"]),
        minimum_prevalence=float(evaluation["minimum_trace_prevalence"]),
        maximum_bh_q=float(evaluation["maximum_dev_bh_q_value"]),
        minimum_abs_paired_d=float(evaluation["minimum_dev_absolute_paired_d"]),
    )
    feature_ids = [int(row["feature_id"]) for row in candidates]
    dev_selected = {
        name: values[:, feature_ids].copy() for name, values in dev["metrics"].items()
    }
    dev_reconstruction = dev["reconstruction"]
    del dev

    test = aggregate_trace_features(
        **common,
        rows=test_rows,
        selected_features=feature_ids,
        progress_prefix=f"{args.condition}/test",
    )
    test_stats = statistics_for_metrics(test["metrics"], test_rows)
    test_selected = {
        name: values[:, feature_ids].copy() for name, values in test["metrics"].items()
    }
    primary_p = [
        float(test_stats[evaluation["primary_metric"]]["p_value"][feature_id])
        for feature_id in feature_ids
    ]
    holm = holm_adjust(primary_p)
    for slot, candidate in enumerate(candidates):
        feature_id = int(candidate["feature_id"])
        expected_sign = 1 if candidate["direction"] == "short" else -1
        candidate["confirmation_holm_p_value"] = float(holm[slot])
        candidate["metrics"] = {
            metric: {
                "dev": _stat_row(dev_stats[metric], feature_id),
                "test": _stat_row(test_stats[metric], feature_id),
            }
            for metric in METRIC_NAMES
        }
        primary_d = float(
            test_stats[evaluation["primary_metric"]]["paired_d"][feature_id]
        )
        early_d = float(
            test_stats[evaluation["early_metric"]]["paired_d"][feature_id]
        )
        candidate["direction_replicated"] = bool(expected_sign * primary_d > 0)
        candidate["first_64_direction_replicated"] = bool(
            expected_sign * early_d > 0
        )
        candidate["confirmed"] = bool(
            candidate["direction_replicated"]
            and candidate["first_64_direction_replicated"]
            and float(holm[slot]) <= float(evaluation["confirmation_holm_alpha"])
            and abs(primary_d)
            >= float(evaluation["minimum_test_absolute_paired_d"])
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    statistics_path = output_dir / "feature_statistics.csv"
    _write_feature_statistics(
        statistics_path,
        int(primary["feature_count"]),
        dev_stats,
        test_stats,
    )
    candidates_path = output_dir / "discovered_features.json"
    write_json_exclusive(
        candidates_path,
        {
            "condition": args.condition,
            "positive_effect_means": "short-associated",
            "negative_effect_means": "long-associated",
            "candidates": candidates,
        },
    )
    trace_metrics_path = output_dir / "selected_trace_metrics.csv"
    _write_trace_metrics(
        trace_metrics_path,
        dev_rows,
        test_rows,
        feature_ids,
        dev_selected,
        test_selected,
    )
    reconstruction_path = output_dir / "common_set_reconstruction.json"
    write_json_exclusive(
        reconstruction_path,
        {"dev": dev_reconstruction, "test": test["reconstruction"]},
    )
    token_path = output_dir / "test_token_associations.csv"
    _write_token_associations(
        token_path,
        feature_ids,
        test["token_mass"],
        test["activation_mass"],
        top_tokens=20,
    )
    relative_path = output_dir / "test_relative_position_profiles.csv"
    _write_relative_profiles(
        relative_path,
        test_rows,
        feature_ids,
        test["relative_means"],
    )
    contribution_path = output_dir / "training_sample_contributions.csv"
    contribution_summary_path = output_dir / "training_sample_contribution_summary.json"
    contribution_summary = _write_training_contributions(
        contribution_path,
        contribution_summary_path,
        _resolve(config["sampling_ablation"]["parent_sae"]["corpus_path"]),
        artifacts["train_sample_path"],
        args.condition,
        load_file,
    )

    confirmed = [row for row in candidates if row["confirmed"]]
    summary = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "condition": args.condition,
        "condition_definition": condition,
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "candidate_count": len(candidates),
        "heldout_confirmed_count": len(confirmed),
        "short_confirmed_count": sum(row["direction"] == "short" for row in confirmed),
        "long_confirmed_count": sum(row["direction"] == "long" for row in confirmed),
        "dev_trace_count": len(dev_rows),
        "test_trace_count": len(test_rows),
        "dev_question_count": len({str(row["problem_id"]) for row in dev_rows}),
        "test_question_count": len({str(row["problem_id"]) for row in test_rows}),
        "common_set_reconstruction": {
            "dev": dev_reconstruction,
            "test": test["reconstruction"],
        },
        "training_sample_contribution": contribution_summary,
        "elapsed_seconds": time.time() - started,
        "input_evidence": evidence,
        "artifacts": {},
        "runtime": runtime_metadata(
            ("python", "torch", "numpy", "scipy", "safetensors")
        ),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "library_code_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/sae_ablation.py"
        ),
        "formal_claim_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    for path in (
        statistics_path,
        candidates_path,
        trace_metrics_path,
        reconstruction_path,
        token_path,
        relative_path,
        contribution_path,
        contribution_summary_path,
    ):
        summary["artifacts"][path.name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    summary_path = output_dir / "scoring_summary.json"
    write_json_exclusive(summary_path, summary)
    (output_dir / "CONDITION_SCORING_COMPLETE").write_text(
        f"status=complete\ncondition={args.condition}\n"
        f"config_hash={summary['config_hash']}\n"
        f"summary_sha256={file_sha256(summary_path)}\n"
        f"heldout_confirmed_count={len(confirmed)}\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _condition(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [
        row for row in config["sampling_ablation"]["conditions"] if row["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicated condition: {name}")
    return matches[0]


def _validate_protocol(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    marker_path = config_path.parent / "PROTOCOL_FROZEN"
    manifest_path = config_path.parent / "protocol_manifest.json"
    marker = read_key_value_marker(marker_path)
    if marker.get("status") != "frozen":
        raise RuntimeError("Sampling-ablation protocol is not frozen.")
    if marker.get("config_hash") != canonical_sha256(config):
        raise ValueError("Frozen protocol hash mismatch.")
    if marker.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Protocol manifest hash mismatch.")
    return {
        "protocol_marker_path": str(marker_path),
        "protocol_marker_sha256": file_sha256(marker_path),
        "protocol_manifest_path": str(manifest_path),
        "protocol_manifest_sha256": file_sha256(manifest_path),
    }


def _condition_artifacts(
    config: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    ablation = config["sampling_ablation"]
    layer = int(ablation["primary_sae"]["layer_index"])
    if bool(condition["train_new_sae"]):
        sample_root = _resolve(ablation["outputs"]["result_root"]) / "token_samples" / condition["name"]
        training_dir = _resolve(ablation["outputs"]["result_root"]) / "sae_training" / condition["name"]
        checkpoint_dir = _resolve(ablation["outputs"]["checkpoint_root"]) / condition["name"]
    else:
        sample_root = _resolve(ablation["parent_sae"]["baseline_sample_root"])
        training_dir = _resolve(ablation["parent_sae"]["baseline_training_dir"])
        checkpoint_dir = _resolve(ablation["parent_sae"]["baseline_checkpoint_dir"])
    sample_manifest_path = sample_root / "sample_manifest.json"
    sample_manifest = read_json(sample_manifest_path)
    layer_row = next(
        row for row in sample_manifest["layers"] if int(row["layer_index"]) == layer
    )
    train_sample = next(row for row in layer_row["samples"] if row["split"] == "train")
    chunk_paths = [Path(row["path"]) for row in layer_row["source_chunks"]]
    return {
        "sample_root": sample_root,
        "sample_manifest_path": sample_manifest_path,
        "sample_manifest": sample_manifest,
        "layer_manifest": layer_row,
        "train_sample_path": Path(train_sample["path"]),
        "training_dir": training_dir,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_path": checkpoint_dir / "sae_model.safetensors",
        "training_metrics_path": training_dir / "training_metrics.json",
        "chunk_paths": chunk_paths,
    }


def _validate_condition_artifacts(
    config: Mapping[str, Any], condition: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    sample_marker_path = artifacts["sample_root"] / "TOKEN_SAMPLES_COMPLETE"
    sample_marker = read_key_value_marker(sample_marker_path)
    if sample_marker.get("status") != "complete":
        raise RuntimeError("Token sample marker is incomplete.")
    if sample_marker.get("manifest_sha256") != file_sha256(
        artifacts["sample_manifest_path"]
    ):
        raise ValueError("Token sample manifest hash mismatch.")
    for sample in artifacts["layer_manifest"]["samples"]:
        if file_sha256(sample["path"]) != sample["sha256"]:
            raise ValueError(f"Token sample hash mismatch: {sample['path']}")
    baseline_manifest_path = (
        _resolve(config["sampling_ablation"]["parent_sae"]["baseline_sample_root"])
        / "sample_manifest.json"
    )
    baseline_manifest = read_json(baseline_manifest_path)
    activation_manifests = []
    for manifest_evidence in baseline_manifest["activation_manifests"]:
        manifest_path = Path(manifest_evidence["path"])
        if file_sha256(manifest_path) != manifest_evidence["sha256"]:
            raise ValueError(f"Activation manifest hash mismatch: {manifest_path}")
        marker_path = manifest_path.parent / "ACTIVATIONS_COMPLETE"
        marker = read_key_value_marker(marker_path)
        if marker.get("status") != "complete" or marker.get(
            "manifest_sha256"
        ) != manifest_evidence["sha256"]:
            raise ValueError(f"Activation marker mismatch: {marker_path}")
        activation_manifests.append(
            {
                "path": str(manifest_path),
                "sha256": manifest_evidence["sha256"],
                "marker_path": str(marker_path),
                "marker_sha256": file_sha256(marker_path),
            }
        )
    for chunk in artifacts["layer_manifest"]["source_chunks"]:
        if not Path(chunk["path"]).is_file():
            raise FileNotFoundError(chunk["path"])
    training_marker_path = artifacts["training_dir"] / "SAE_TRAINING_COMPLETE"
    training_marker = read_key_value_marker(training_marker_path)
    training_metrics = read_json(artifacts["training_metrics_path"])
    if training_marker.get("status") != "complete":
        raise RuntimeError("SAE training marker is incomplete.")
    if training_marker.get("training_metrics_sha256") != file_sha256(
        artifacts["training_metrics_path"]
    ):
        raise ValueError("SAE training metrics hash mismatch.")
    if training_marker.get("model_sha256") != file_sha256(
        artifacts["checkpoint_path"]
    ):
        raise ValueError("SAE checkpoint hash mismatch.")
    expected_hash = (
        canonical_sha256(config)
        if bool(condition["train_new_sae"])
        else config["sampling_ablation"]["parent_sae"]["protocol_hash"]
    )
    if training_metrics["config_hash"] != expected_hash:
        raise ValueError("SAE was trained under an unexpected protocol.")
    return {
        "sample_manifest_path": str(artifacts["sample_manifest_path"]),
        "sample_manifest_sha256": file_sha256(artifacts["sample_manifest_path"]),
        "training_marker_path": str(training_marker_path),
        "training_marker_sha256": file_sha256(training_marker_path),
        "training_metrics_path": str(artifacts["training_metrics_path"]),
        "training_metrics_sha256": file_sha256(artifacts["training_metrics_path"]),
        "checkpoint_path": str(artifacts["checkpoint_path"]),
        "checkpoint_sha256": file_sha256(artifacts["checkpoint_path"]),
        "activation_chunk_count": len(artifacts["chunk_paths"]),
        "activation_manifests": activation_manifests,
    }


def _stat_row(statistics: Mapping[str, Any], feature_id: int) -> dict[str, float]:
    return {
        key: float(statistics[key][feature_id])
        for key in (
            "effect",
            "ci_low",
            "ci_high",
            "paired_d",
            "p_value",
            "bh_q_value",
            "prevalence",
        )
    }


def _write_feature_statistics(
    path: Path,
    feature_count: int,
    dev: Mapping[str, Mapping[str, Any]],
    test: Mapping[str, Mapping[str, Any]],
) -> None:
    statistic_names = (
        "effect",
        "ci_low",
        "ci_high",
        "paired_d",
        "p_value",
        "bh_q_value",
        "prevalence",
    )
    fields = ["feature_id"] + [
        f"{split}_{metric}_{statistic}"
        for split in ("dev", "test")
        for metric in METRIC_NAMES
        for statistic in statistic_names
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature_id in range(feature_count):
            row: dict[str, Any] = {"feature_id": feature_id}
            for split, collection in (("dev", dev), ("test", test)):
                for metric in METRIC_NAMES:
                    for statistic in statistic_names:
                        row[f"{split}_{metric}_{statistic}"] = float(
                            collection[metric][statistic][feature_id]
                        )
            writer.writerow(row)


def _write_trace_metrics(
    path: Path,
    dev_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature_ids: Sequence[int],
    dev_metrics: Mapping[str, np.ndarray],
    test_metrics: Mapping[str, np.ndarray],
) -> None:
    fields = [
        "question_split",
        "problem_id",
        "trace_id",
        "corpus_index",
        "analysis_length_label",
        "solution_token_count",
    ] + [
        f"feature_{feature_id}_{metric}"
        for feature_id in feature_ids
        for metric in METRIC_NAMES
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rows, metrics in ((dev_rows, dev_metrics), (test_rows, test_metrics)):
            for row_index, source in enumerate(rows):
                row = {key: source[key] for key in fields[:6]}
                for slot, feature_id in enumerate(feature_ids):
                    for metric in METRIC_NAMES:
                        row[f"feature_{feature_id}_{metric}"] = float(
                            metrics[metric][row_index, slot]
                        )
                writer.writerow(row)


def _write_token_associations(
    path: Path,
    feature_ids: Sequence[int],
    token_mass: Counter,
    activation_mass: Counter,
    *,
    top_tokens: int,
) -> None:
    fields = [
        "feature_id",
        "analysis_length_label",
        "rank",
        "token_id",
        "activation_mass",
        "within_label_mass_share",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for feature_id in feature_ids:
            for label in ("short", "long"):
                ranked = sorted(
                    (
                        (token_id, mass)
                        for (candidate, observed_label, token_id), mass in token_mass.items()
                        if candidate == feature_id and observed_label == label
                    ),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:top_tokens]
                denominator = float(activation_mass[(feature_id, label)])
                for rank, (token_id, mass) in enumerate(ranked, start=1):
                    writer.writerow(
                        {
                            "feature_id": feature_id,
                            "analysis_length_label": label,
                            "rank": rank,
                            "token_id": token_id,
                            "activation_mass": float(mass),
                            "within_label_mass_share": float(mass) / max(denominator, 1e-12),
                        }
                    )


def _write_relative_profiles(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    feature_ids: Sequence[int],
    relative_means: np.ndarray,
) -> None:
    fields = ["feature_id", "analysis_length_label", "relative_position_bin", "mean_activation"]
    labels = np.asarray([row["analysis_length_label"] for row in rows])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for slot, feature_id in enumerate(feature_ids):
            for label in ("short", "long"):
                means = relative_means[labels == label, slot].mean(axis=0)
                for bin_index, value in enumerate(means):
                    writer.writerow(
                        {
                            "feature_id": feature_id,
                            "analysis_length_label": label,
                            "relative_position_bin": bin_index,
                            "mean_activation": float(value),
                        }
                    )


def _write_training_contributions(
    path: Path,
    summary_path: Path,
    corpus_path: Path,
    sample_path: Path,
    condition: str,
    load_file: Any,
) -> dict[str, Any]:
    all_rows = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["question_split"] == "train":
                all_rows.append(row)
    sampled = load_file(str(sample_path), device="cpu")["trace_indices"].long().numpy()
    counts = Counter(int(value) for value in sampled.tolist())
    fields = [
        "condition",
        "corpus_index",
        "trace_id",
        "problem_id",
        "analysis_length_label",
        "solution_token_count",
        "sampled_token_count",
        "sampled_fraction",
    ]
    records = []
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in sorted(all_rows, key=lambda row: int(row["corpus_index"])):
            corpus_index = int(source["corpus_index"])
            length = int(source["solution_token_count"])
            count = int(counts[corpus_index])
            record = {
                "condition": condition,
                "corpus_index": corpus_index,
                "trace_id": source["trace_id"],
                "problem_id": source["problem_id"],
                "analysis_length_label": source["analysis_length_label"],
                "solution_token_count": length,
                "sampled_token_count": count,
                "sampled_fraction": count / max(length, 1),
            }
            records.append(record)
            writer.writerow(record)
    from scipy.stats import pearsonr, spearmanr

    lengths = np.asarray([row["solution_token_count"] for row in records], dtype=float)
    sample_counts = np.asarray([row["sampled_token_count"] for row in records], dtype=float)
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in records:
        grouped[str(row["analysis_length_label"])].append(int(row["sampled_token_count"]))
    summary = {
        "condition": condition,
        "train_trace_count": len(records),
        "sampled_token_count": int(sample_counts.sum()),
        "sampled_count_min": int(sample_counts.min()),
        "sampled_count_max": int(sample_counts.max()),
        "sampled_count_mean": float(sample_counts.mean()),
        "length_sample_count_pearson_r": float(pearsonr(lengths, sample_counts).statistic),
        "length_sample_count_spearman_r": float(spearmanr(lengths, sample_counts).statistic),
        "mean_sampled_count_by_label": {
            label: float(np.mean(values)) for label, values in sorted(grouped.items())
        },
    }
    write_json_exclusive(summary_path, summary)
    return summary


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
