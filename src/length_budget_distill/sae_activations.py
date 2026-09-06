"""Teacher-trace replay and chunked residual-activation storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from .factorial import file_sha256


def encode_replayed_trace(
    tokenizer: Any, row: Mapping[str, Any], *, max_sequence_tokens: int
) -> dict[str, Any]:
    """Replay the parent chat prompt and retain the complete assistant solution."""

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(row["prompt"])}],
        tokenize=True,
        add_generation_prompt=True,
    )
    solution_ids = tokenizer.encode(str(row["solution"]), add_special_tokens=False)
    if not solution_ids:
        raise ValueError(f"Trace has no solution tokens: {row['trace_id']}")
    if len(solution_ids) >= max_sequence_tokens:
        raise ValueError(
            f"Solution itself exceeds max_sequence_tokens: {row['trace_id']}"
        )
    retained_prompt = list(prompt_ids)
    truncated_prompt_tokens = 0
    if len(retained_prompt) + len(solution_ids) > max_sequence_tokens:
        keep = max_sequence_tokens - len(solution_ids)
        truncated_prompt_tokens = len(retained_prompt) - keep
        retained_prompt = retained_prompt[-keep:]
    return {
        "input_ids": retained_prompt + list(solution_ids),
        "completion_start": len(retained_prompt),
        "solution_ids": list(solution_ids),
        "original_prompt_tokens": len(prompt_ids),
        "retained_prompt_tokens": len(retained_prompt),
        "truncated_prompt_tokens": truncated_prompt_tokens,
    }


class ActivationChunkWriter:
    """Append token rows and atomically publish bounded safetensors chunks."""

    def __init__(self, output_dir: Path, layer_index: int, chunk_tokens: int) -> None:
        self.output_dir = output_dir / f"layer_{layer_index:02d}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.layer_index = int(layer_index)
        self.chunk_tokens = int(chunk_tokens)
        self.buffers: dict[str, list[torch.Tensor]] = {
            "activations": [],
            "token_ids": [],
            "trace_indices": [],
            "positions": [],
        }
        self.buffered_tokens = 0
        self.chunk_index = 0
        self.total_tokens = 0
        self.chunks: list[dict[str, Any]] = []

    def append(
        self,
        activations: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        trace_index: int,
    ) -> None:
        values = activations.detach().to(device="cpu", dtype=torch.bfloat16)
        ids = token_ids.detach().to(device="cpu", dtype=torch.int32)
        token_count = int(values.shape[0])
        if values.ndim != 2 or ids.shape != (token_count,):
            raise ValueError("Malformed activation append payload.")
        self.buffers["activations"].append(values.contiguous())
        self.buffers["token_ids"].append(ids.contiguous())
        self.buffers["trace_indices"].append(
            torch.full((token_count,), int(trace_index), dtype=torch.int32)
        )
        self.buffers["positions"].append(torch.arange(token_count, dtype=torch.int32))
        self.buffered_tokens += token_count
        if self.buffered_tokens >= self.chunk_tokens:
            self.flush()

    def flush(self) -> None:
        if self.buffered_tokens == 0:
            return
        tensors = {
            name: torch.cat(parts, dim=0).contiguous()
            for name, parts in self.buffers.items()
        }
        filename = f"chunk_{self.chunk_index:05d}.safetensors"
        path = self.output_dir / filename
        partial = self.output_dir / f".{filename}.partial-{os.getpid()}"
        save_file(
            tensors,
            partial,
            metadata={
                "layer_index": str(self.layer_index),
                "token_count": str(self.buffered_tokens),
                "stream": "post_block_residual",
            },
        )
        os.replace(partial, path)
        observed = int(tensors["activations"].shape[0])
        self.chunks.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "token_count": observed,
                "hidden_size": int(tensors["activations"].shape[1]),
            }
        )
        self.total_tokens += observed
        self.chunk_index += 1
        self.buffered_tokens = 0
        for parts in self.buffers.values():
            parts.clear()

    def finish(self) -> dict[str, Any]:
        self.flush()
        return {
            "layer_index": self.layer_index,
            "token_count": self.total_tokens,
            "chunk_count": len(self.chunks),
            "chunks": self.chunks,
        }


def padded_batch(
    encoded: Sequence[Mapping[str, Any]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in encoded)
    input_ids = torch.full(
        (len(encoded), max_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(encoded), max_length), dtype=torch.long)
    for index, row in enumerate(encoded):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : len(values)] = values
        attention_mask[index, : len(values)] = 1
    return input_ids, attention_mask
