#!/usr/bin/env python3
"""Build deterministic train/dev/test token samples from activation shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    validated_artifact_marker,
    write_json_exclusive,
)
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.records import read_jsonl
from length_budget_distill.sae_sampling import (
    PriorityReservoir,
    activation_normalization,
    token_priorities,
)


SPLITS = ("train", "dev", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json",
    )
    parser.add_argument(
        "--corpus",
        default="results/phase2_sae_pilot_v1/formal/corpus/mixed_trajectories.jsonl",
    )
    parser.add_argument(
        "--activation-root",
        default="results/phase2_sae_pilot_v1/formal/activations",
    )
    parser.add_argument(
        "--output-dir", default="results/phase2_sae_pilot_v1/formal/token_samples"
    )
    args = parser.parse_args()
    config_path = _resolve(args.config)
    corpus_path = _resolve(args.corpus)
    activation_root = _resolve(args.activation_root)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    corpus = list(read_jsonl(corpus_path))
    if [int(row["corpus_index"]) for row in corpus] != list(range(len(corpus))):
        raise ValueError("Corpus indices are not dense and ordered.")
    split_by_trace = np.asarray([str(row["question_split"]) for row in corpus])
    extraction = config["activation_extraction"]
    shard_count = int(extraction["trajectory_shards"])
    manifests = []
    for shard_index in range(shard_count):
        shard_dir = activation_root / f"shard_{shard_index:02d}_of_{shard_count:02d}"
        manifest_path = shard_dir / "activation_manifest.json"
        validated_artifact_marker(
            shard_dir / "ACTIVATIONS_COMPLETE",
            expected_status="complete",
            hash_bindings={"manifest_sha256": manifest_path},
        )
        manifest = read_json(manifest_path)
        if (
            int(manifest["shard_index"]) != shard_index
            or int(manifest["shard_count"]) != shard_count
            or manifest["config_hash"] != canonical_sha256(config)
        ):
            raise ValueError("Activation shard manifest topology mismatch.")
        manifests.append((manifest_path, manifest))
    from safetensors.torch import load_file, save_file

    output_dir.mkdir(parents=True, exist_ok=False)
    sample_config = config["token_sampling"]
    hidden_size = int(config["teacher"]["hidden_size"])
    output_layers = []
    for layer_index in extraction["layer_indices_zero_based"]:
        layer_index = int(layer_index)
        reservoirs = {
            split: PriorityReservoir(
                int(sample_config["tokens_per_layer"][split]), hidden_size
            )
            for split in SPLITS
        }
        chunk_evidence = []
        for manifest_path, manifest in manifests:
            layer = next(
                row
                for row in manifest["layers"]
                if int(row["layer_index"]) == layer_index
            )
            for chunk in layer["chunks"]:
                path = Path(chunk["path"])
                if file_sha256(path) != chunk["sha256"]:
                    raise ValueError(f"Activation chunk hash mismatch: {path}")
                tensors = load_file(path, device="cpu")
                traces = tensors["trace_indices"]
                positions = tensors["positions"]
                token_ids = tensors["token_ids"]
                trace_numpy = traces.numpy().astype(np.int64, copy=False)
                token_splits = split_by_trace[trace_numpy]
                for split_code, split in enumerate(SPLITS):
                    selected = np.flatnonzero(token_splits == split)
                    if not len(selected):
                        continue
                    index = np.asarray(selected, dtype=np.int64)
                    priorities = token_priorities(
                        trace_numpy[index],
                        positions.numpy()[index],
                        token_ids.numpy()[index],
                        seed=int(sample_config["seed"]),
                        layer_index=layer_index,
                        split_code=split_code,
                    )
                    torch_index = __import__("torch").from_numpy(index)
                    reservoirs[split].add(
                        tensors["activations"][torch_index],
                        traces[torch_index],
                        positions[torch_index],
                        token_ids[torch_index],
                        priorities,
                    )
                chunk_evidence.append({"path": str(path), "sha256": chunk["sha256"]})
        split_outputs = []
        sampled = {split: reservoirs[split].tensors() for split in SPLITS}
        mean, scale, normalization_summary = activation_normalization(
            sampled["train"]["activations"]
        )
        normalizer_path = output_dir / f"layer_{layer_index:02d}_normalizer.safetensors"
        _save_atomic(
            save_file,
            {"mean": mean.contiguous(), "scale": scale.reshape(1)},
            normalizer_path,
            {"layer_index": str(layer_index)},
        )
        for split in SPLITS:
            path = output_dir / f"layer_{layer_index:02d}_{split}.safetensors"
            _save_atomic(
                save_file,
                sampled[split],
                path,
                {
                    "layer_index": str(layer_index),
                    "split": split,
                    "sampling_method": sample_config["method"],
                },
            )
            split_outputs.append(
                {
                    "split": split,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "sampled_tokens": int(sampled[split]["activations"].shape[0]),
                    "available_tokens": reservoirs[split].seen,
                }
            )
        output_layers.append(
            {
                "layer_index": layer_index,
                "samples": split_outputs,
                "normalizer_path": str(normalizer_path),
                "normalizer_sha256": file_sha256(normalizer_path),
                "normalization": normalization_summary,
                "source_chunks": chunk_evidence,
            }
        )
        print(f"sampled layer {layer_index}", flush=True)
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
        "activation_manifests": [
            {"path": str(path), "sha256": file_sha256(path)} for path, _ in manifests
        ],
        "layers": output_layers,
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "sample_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "TOKEN_SAMPLES_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "layers": len(output_layers)}, indent=2))


def _save_atomic(save_file, tensors, path: Path, metadata) -> None:
    partial = path.parent / f".{path.name}.partial-{os.getpid()}"
    save_file(tensors, partial, metadata=metadata)
    os.replace(partial, path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
