#!/usr/bin/env python3
"""Extract completion-token teacher residual streams for one trajectory shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from length_budget_distill.experiment_io import (
    read_json,
    validated_artifact_marker,
    write_json_exclusive,
)
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    runtime_metadata,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.sae_activations import (
    ActivationChunkWriter,
    encode_replayed_trace,
    padded_batch,
)
from length_budget_distill.sae_data import trace_shard


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
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index.")
    config_path = _resolve(args.config)
    corpus_path = _resolve(args.corpus)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = read_json(config_path)
    corpus_marker = corpus_path.parent / "CORPUS_COMPLETE"
    validated_artifact_marker(
        corpus_marker,
        expected_status="complete",
        hash_bindings={"corpus_sha256": corpus_path},
    )
    extraction = config["activation_extraction"]
    if args.shard_count != int(extraction["trajectory_shards"]):
        raise ValueError("Runtime shard count differs from frozen protocol.")
    rows = [
        row
        for row in read_jsonl(corpus_path)
        if trace_shard(str(row["trace_id"]), args.shard_count) == args.shard_index
    ]
    if not rows:
        raise ValueError("No trajectories assigned to extraction shard.")
    teacher = config["teacher"]
    snapshot = Path(teacher["snapshot_path"])
    if file_sha256(snapshot / "tokenizer.json") != teacher["tokenizer_json_sha256"]:
        raise ValueError("Teacher tokenizer hash changed.")
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for residual activation extraction.")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    model.to("cuda")
    hidden_size = int(model.config.hidden_size)
    if hidden_size != int(teacher["hidden_size"]):
        raise ValueError("Teacher hidden size changed.")
    blocks = getattr(getattr(model, "model", model), "layers")
    layers = [int(value) for value in extraction["layer_indices_zero_based"]]
    if len(blocks) != int(teacher["num_hidden_layers"]):
        raise ValueError("Teacher layer count changed.")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layers:

        def hook(_module, _inputs, output, *, index=layer_index):
            captured[index] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        handles.append(blocks[layer_index].register_forward_hook(hook))
    output_dir.mkdir(parents=True, exist_ok=False)
    writers = {
        layer: ActivationChunkWriter(output_dir, layer, int(extraction["chunk_tokens"]))
        for layer in layers
    }
    trace_records = []
    batch_size = int(extraction["batch_size"])
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            encoded = [
                encode_replayed_trace(
                    tokenizer,
                    row,
                    max_sequence_tokens=int(extraction["max_sequence_tokens"]),
                )
                for row in batch_rows
            ]
            input_ids, attention_mask = padded_batch(encoded, tokenizer.pad_token_id)
            captured.clear()
            with torch.inference_mode():
                model(
                    input_ids=input_ids.to("cuda", non_blocking=True),
                    attention_mask=attention_mask.to("cuda", non_blocking=True),
                    use_cache=False,
                    return_dict=True,
                )
            if set(captured) != set(layers):
                raise RuntimeError("Not all registered residual hooks fired.")
            for batch_index, (row, encoding) in enumerate(zip(batch_rows, encoded)):
                left = int(encoding["completion_start"])
                right = left + len(encoding["solution_ids"])
                token_ids = input_ids[batch_index, left:right]
                for layer in layers:
                    writers[layer].append(
                        captured[layer][batch_index, left:right, :],
                        token_ids,
                        trace_index=int(row["corpus_index"]),
                    )
                trace_records.append(
                    {
                        "corpus_index": int(row["corpus_index"]),
                        "trace_id": str(row["trace_id"]),
                        "problem_id": str(row["problem_id"]),
                        "question_split": str(row["question_split"]),
                        "analysis_length_label": str(row["analysis_length_label"]),
                        "is_correct": bool(row["is_correct"]),
                        "reported_solution_tokens": int(row["solution_token_count"]),
                        "replayed_solution_tokens": len(encoding["solution_ids"]),
                        "original_prompt_tokens": int(
                            encoding["original_prompt_tokens"]
                        ),
                        "retained_prompt_tokens": int(
                            encoding["retained_prompt_tokens"]
                        ),
                        "truncated_prompt_tokens": int(
                            encoding["truncated_prompt_tokens"]
                        ),
                    }
                )
            if start % (50 * batch_size) == 0:
                print(
                    f"shard={args.shard_index} traces={min(start + batch_size, len(rows))}/{len(rows)}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    layer_manifests = [writers[layer].finish() for layer in layers]
    token_counts = {int(row["token_count"]) for row in layer_manifests}
    if len(token_counts) != 1:
        raise ValueError("Extracted layer token counts disagree.")
    trace_path = output_dir / "trace_index.jsonl"
    write_jsonl(trace_path, trace_records)
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "config_hash": canonical_sha256(config),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "trajectory_count": len(rows),
        "token_count_per_layer": next(iter(token_counts)),
        "trace_index_path": str(trace_path),
        "trace_index_sha256": file_sha256(trace_path),
        "layers": layer_manifests,
        "runtime": runtime_metadata(("python", "torch", "transformers", "safetensors")),
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "formal_claim_allowed": False,
    }
    manifest_path = output_dir / "activation_manifest.json"
    write_json_exclusive(manifest_path, manifest)
    (output_dir / "ACTIVATIONS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={manifest['config_hash']}\n"
        f"trace_index_sha256={manifest['trace_index_sha256']}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "layers"}, indent=2))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
