#!/usr/bin/env python3
"""Build exact-budget trace-balanced activation samples for one SAE condition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import read_json, validated_artifact_marker, write_json_exclusive
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl
from length_budget_distill.sae_sampling import (
    activation_normalization,
    deterministic_positions_for_quotas,
    maximally_equal_trace_quotas,
)


SPLITS = ("train", "dev", "test")


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
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    config = read_json(config_path)
    ablation = config["sampling_ablation"]
    condition = next(
        (row for row in ablation["conditions"] if row["name"] == args.condition),
        None,
    )
    if condition is None or not bool(condition["train_new_sae"]):
        raise ValueError("Condition must be a registered newly trained SAE condition.")
    if condition["sampling_unit"] != "trace_then_position":
        raise ValueError("This builder only implements trace-balanced conditions.")
    parent = ablation["parent_sae"]
    parent_protocol_path = _resolve(parent["protocol_path"])
    if (
        file_sha256(parent_protocol_path) != parent["protocol_sha256"]
        or canonical_sha256(read_json(parent_protocol_path)) != parent["protocol_hash"]
    ):
        raise ValueError("Parent protocol hash mismatch.")
    corpus_path = _resolve(parent["corpus_path"])
    corpus = list(read_jsonl(corpus_path))
    if [int(row["corpus_index"]) for row in corpus] != list(range(len(corpus))):
        raise ValueError("Corpus indices are not dense and ordered.")
    layer_index = int(ablation["primary_sae"]["layer_index"])
    hidden_size = int(config["teacher"]["hidden_size"])
    position_limit = condition["position_limit"]
    seed = int(config["token_sampling"]["seed"])
    budgets = config["token_sampling"]["tokens_per_layer"]
    selection_matrix = np.zeros(
        (len(corpus), max(int(row["solution_token_count"]) for row in corpus)),
        dtype=bool,
    )
    quotas_by_split = {}
    selection_summaries = {}
    for split_code, split in enumerate(SPLITS):
        capacities = {
            int(row["corpus_index"]): min(
                int(row["solution_token_count"]),
                int(position_limit)
                if position_limit is not None
                else int(row["solution_token_count"]),
            )
            for row in corpus
            if row["question_split"] == split
        }
        quotas = maximally_equal_trace_quotas(
            capacities,
            target_tokens=int(budgets[split]),
            seed=seed + split_code,
        )
        selected = deterministic_positions_for_quotas(
            capacities,
            quotas,
            seed=seed,
            layer_index=layer_index,
            split_code=split_code,
        )
        for trace_index, positions in selected.items():
            selection_matrix[trace_index, positions] = True
        quotas_by_split[split] = quotas
        quota_values = np.asarray(list(quotas.values()), dtype=np.int64)
        capacities_values = np.asarray(list(capacities.values()), dtype=np.int64)
        selection_summaries[split] = {
            "trace_count": len(quotas),
            "target_tokens": int(budgets[split]),
            "selected_tokens": int(quota_values.sum()),
            "quota_min": int(quota_values.min()),
            "quota_max": int(quota_values.max()),
            "quota_mean": float(quota_values.mean()),
            "available_position_min": int(capacities_values.min()),
            "available_position_max": int(capacities_values.max()),
            "position_limit": position_limit,
        }
    manifests = _validated_activation_manifests(
        _resolve(parent["activation_root"]),
        expected_parent_hash=parent["protocol_hash"],
    )
    import torch
    from safetensors.torch import load_file, save_file

    buffers = {
        split: {
            "activations": torch.empty(
                (int(budgets[split]), hidden_size), dtype=torch.bfloat16
            ),
            "trace_indices": torch.empty(int(budgets[split]), dtype=torch.int32),
            "positions": torch.empty(int(budgets[split]), dtype=torch.int32),
            "token_ids": torch.empty(int(budgets[split]), dtype=torch.int32),
        }
        for split in SPLITS
    }
    offsets = Counter()
    split_codes = np.asarray(
        [{"train": 0, "dev": 1, "test": 2}[str(row["question_split"])] for row in corpus],
        dtype=np.int8,
    )
    source_chunks = []
    for manifest_path, manifest in manifests:
        layer = next(
            row for row in manifest["layers"] if int(row["layer_index"]) == layer_index
        )
        for chunk in layer["chunks"]:
            path = Path(chunk["path"])
            tensors = load_file(path, device="cpu")
            traces = tensors["trace_indices"].long().numpy()
            positions = tensors["positions"].long().numpy()
            keep = selection_matrix[traces, positions]
            for split_code, split in enumerate(SPLITS):
                selected = np.flatnonzero(keep & (split_codes[traces] == split_code))
                if not len(selected):
                    continue
                source_index = torch.from_numpy(selected.astype(np.int64))
                start = int(offsets[split])
                stop = start + len(selected)
                if stop > int(budgets[split]):
                    raise RuntimeError(f"Selected too many {split} tokens.")
                target = slice(start, stop)
                for name in buffers[split]:
                    buffers[split][name][target] = tensors[name][source_index].to(
                        buffers[split][name].dtype
                    )
                offsets[split] = stop
            source_chunks.append({"path": str(path), "registered_sha256": chunk["sha256"]})
    for split in SPLITS:
        if offsets[split] != int(budgets[split]):
            raise RuntimeError(
                f"Observed {offsets[split]} selected {split} tokens; expected {budgets[split]}."
            )
        observed = Counter(int(value) for value in buffers[split]["trace_indices"].tolist())
        expected = quotas_by_split[split]
        if observed != Counter(expected):
            raise RuntimeError(f"Observed per-trace quotas differ for {split}.")
    output_dir.mkdir(parents=True, exist_ok=False)
    mean, scale, normalization_summary = activation_normalization(
        buffers["train"]["activations"]
    )
    normalizer_path = output_dir / f"layer_{layer_index:02d}_normalizer.safetensors"
    _save_atomic(
        save_file,
        {"mean": mean.contiguous(), "scale": scale.reshape(1)},
        normalizer_path,
        {"layer_index": str(layer_index), "condition": args.condition},
    )
    samples = []
    for split in SPLITS:
        path = output_dir / f"layer_{layer_index:02d}_{split}.safetensors"
        _save_atomic(
            save_file,
            {name: tensor.contiguous() for name, tensor in buffers[split].items()},
            path,
            {
                "layer_index": str(layer_index),
                "split": split,
                "condition": args.condition,
                "sampling_unit": condition["sampling_unit"],
            },
        )
        samples.append(
            {
                "split": split,
                "path": str(path),
                "sha256": file_sha256(path),
                "sampled_tokens": int(buffers[split]["activations"].shape[0]),
                "available_tokens": int(
                    sum(
                        min(
                            int(row["solution_token_count"]),
                            int(position_limit)
                            if position_limit is not None
                            else int(row["solution_token_count"]),
                        )
                        for row in corpus
                        if row["question_split"] == split
                    )
                ),
                "quota_summary": selection_summaries[split],
            }
        )
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "condition": args.condition,
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
        "sampling_unit": condition["sampling_unit"],
        "position_limit": position_limit,
        "layers": [
            {
                "layer_index": layer_index,
                "samples": samples,
                "normalizer_path": str(normalizer_path),
                "normalizer_sha256": file_sha256(normalizer_path),
                "normalization": normalization_summary,
                "source_chunks": source_chunks,
            }
        ],
        "source_activation_manifests": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path, _ in manifests
        ],
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "library_code_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/sae_sampling.py"
        ),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "sample_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "TOKEN_SAMPLES_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\ncondition={args.condition}\n"
        "formal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _validated_activation_manifests(root: Path, *, expected_parent_hash: str):
    manifests = []
    for shard_dir in sorted(root.glob("shard_*_of_*")):
        manifest_path = shard_dir / "activation_manifest.json"
        validated_artifact_marker(
            shard_dir / "ACTIVATIONS_COMPLETE",
            expected_status="complete",
            hash_bindings={"manifest_sha256": manifest_path},
        )
        manifest = read_json(manifest_path)
        if manifest["config_hash"] != expected_parent_hash:
            raise ValueError(f"Activation manifest parent hash mismatch: {manifest_path}")
        manifests.append((manifest_path, manifest))
    if len(manifests) != 3:
        raise ValueError("Expected exactly three activation shard manifests.")
    return manifests


def _save_atomic(save_file, tensors, path: Path, metadata) -> None:
    partial = path.parent / f".{path.name}.partial-{os.getpid()}"
    save_file(tensors, partial, metadata=metadata)
    os.replace(partial, path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
