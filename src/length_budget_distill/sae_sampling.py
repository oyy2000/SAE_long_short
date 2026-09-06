"""Deterministic priority sampling of activation tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


def maximally_equal_trace_quotas(
    capacities: Mapping[int, int],
    *,
    target_tokens: int,
    seed: int,
) -> dict[int, int]:
    """Allocate an exact token budget with maximally equal per-trace quotas.

    Traces with insufficient capacity saturate first. Any indivisible remainder is
    assigned by a deterministic SplitMix64 ordering of trace identity.
    """

    if target_tokens <= 0 or not capacities:
        raise ValueError("A positive target and nonempty capacities are required.")
    normalized = {int(key): int(value) for key, value in capacities.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("Trace capacities cannot be negative.")
    if sum(normalized.values()) < target_tokens:
        raise ValueError("Available trace tokens do not cover the requested budget.")
    low, high = 0, max(normalized.values())
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(value, middle) for value in normalized.values()) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    quotas = {key: min(value, low) for key, value in normalized.items()}
    remainder = target_tokens - sum(quotas.values())
    eligible = np.asarray(
        [key for key, capacity in normalized.items() if capacity > quotas[key]],
        dtype=np.int64,
    )
    if remainder > len(eligible):
        raise RuntimeError("Water-filled quota remainder exceeds eligible traces.")
    if remainder:
        priorities = token_priorities(
            eligible,
            np.zeros(len(eligible), dtype=np.int64),
            np.zeros(len(eligible), dtype=np.int64),
            seed=seed,
            layer_index=0,
            split_code=0,
        )
        for index in np.argsort(priorities, kind="stable")[:remainder]:
            quotas[int(eligible[index])] += 1
    if sum(quotas.values()) != target_tokens:
        raise RuntimeError("Trace quota allocation missed the exact target.")
    return quotas


def deterministic_positions_for_quotas(
    capacities: Mapping[int, int],
    quotas: Mapping[int, int],
    *,
    seed: int,
    layer_index: int,
    split_code: int,
) -> dict[int, np.ndarray]:
    """Choose positions without replacement for registered per-trace quotas."""

    selected: dict[int, np.ndarray] = {}
    for trace_index in sorted(capacities):
        capacity = int(capacities[trace_index])
        quota = int(quotas.get(trace_index, 0))
        if not 0 <= quota <= capacity:
            raise ValueError(f"Invalid quota for trace {trace_index}: {quota}/{capacity}")
        positions = np.arange(capacity, dtype=np.int64)
        priorities = token_priorities(
            np.full(capacity, int(trace_index), dtype=np.int64),
            positions,
            np.zeros(capacity, dtype=np.int64),
            seed=seed,
            layer_index=layer_index,
            split_code=split_code,
        )
        chosen = np.argsort(priorities, kind="stable")[:quota]
        selected[int(trace_index)] = np.sort(positions[chosen])
    return selected


def token_priorities(
    trace_indices: np.ndarray,
    positions: np.ndarray,
    token_ids: np.ndarray,
    *,
    seed: int,
    layer_index: int,
    split_code: int,
) -> np.ndarray:
    """Vectorized SplitMix64 priorities keyed by token identity."""

    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        values = (
            np.asarray(trace_indices, dtype=np.uint64) * np.uint64(0x9E3779B185EBCA87)
        ) & mask
        values ^= (
            np.asarray(positions, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        ) & mask
        values ^= (
            np.asarray(token_ids, dtype=np.uint64) * np.uint64(0x165667B19E3779F9)
        ) & mask
        values ^= np.uint64(seed) * np.uint64(0x85EBCA77C2B2AE63)
        values ^= np.uint64(layer_index + 1) * np.uint64(0x27D4EB2F165667C5)
        values ^= np.uint64(split_code + 1) * np.uint64(0x94D049BB133111EB)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values &= mask
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values &= mask
        values ^= values >> np.uint64(31)
    return values


@dataclass
class PriorityReservoir:
    capacity: int
    hidden_size: int

    def __post_init__(self) -> None:
        self.activations = torch.empty(
            (self.capacity, self.hidden_size), dtype=torch.bfloat16
        )
        self.trace_indices = torch.empty(self.capacity, dtype=torch.int32)
        self.positions = torch.empty(self.capacity, dtype=torch.int32)
        self.token_ids = torch.empty(self.capacity, dtype=torch.int32)
        self.priorities = np.empty(self.capacity, dtype=np.uint64)
        self.size = 0
        self.seen = 0

    def add(
        self,
        activations: torch.Tensor,
        trace_indices: torch.Tensor,
        positions: torch.Tensor,
        token_ids: torch.Tensor,
        priorities: np.ndarray,
    ) -> None:
        count = int(activations.shape[0])
        self.seen += count
        if count == 0:
            return
        if not (
            trace_indices.shape == positions.shape == token_ids.shape == (count,)
            and len(priorities) == count
        ):
            raise ValueError("Priority-reservoir fields have inconsistent shapes.")
        offset = 0
        if self.size < self.capacity:
            fill = min(count, self.capacity - self.size)
            target = slice(self.size, self.size + fill)
            self.activations[target] = activations[:fill].to(torch.bfloat16)
            self.trace_indices[target] = trace_indices[:fill].to(torch.int32)
            self.positions[target] = positions[:fill].to(torch.int32)
            self.token_ids[target] = token_ids[:fill].to(torch.int32)
            self.priorities[target] = priorities[:fill]
            self.size += fill
            offset = fill
        if offset == count:
            return
        incoming_priorities = priorities[offset:]
        current = self.priorities[: self.size]
        combined = np.concatenate((current, incoming_priorities))
        keep = np.argpartition(combined, self.capacity - 1)[: self.capacity]
        old_keep = keep[keep < self.capacity]
        new_keep = keep[keep >= self.capacity] - self.capacity
        retained = np.zeros(self.capacity, dtype=bool)
        retained[old_keep] = True
        drop = np.flatnonzero(~retained)
        if len(drop) != len(new_keep):
            raise RuntimeError("Priority reservoir replacement cardinality mismatch.")
        if len(drop):
            source = torch.from_numpy(new_keep.astype(np.int64)) + offset
            target = torch.from_numpy(drop.astype(np.int64))
            self.activations[target] = activations[source].to(torch.bfloat16)
            self.trace_indices[target] = trace_indices[source].to(torch.int32)
            self.positions[target] = positions[source].to(torch.int32)
            self.token_ids[target] = token_ids[source].to(torch.int32)
            self.priorities[drop] = incoming_priorities[new_keep]

    def tensors(self) -> dict[str, torch.Tensor]:
        if self.size != self.capacity:
            raise ValueError(
                f"Only observed {self.size} tokens for capacity {self.capacity}."
            )
        order = np.argsort(self.priorities[: self.size], kind="stable")
        index = torch.from_numpy(order.astype(np.int64))
        priorities = self.priorities[: self.size][order].copy()
        return {
            "activations": self.activations[index].contiguous(),
            "trace_indices": self.trace_indices[index].contiguous(),
            "positions": self.positions[index].contiguous(),
            "token_ids": self.token_ids[index].contiguous(),
            "priorities": torch.from_numpy(priorities.view(np.int64)).contiguous(),
        }


def activation_normalization(
    activations: torch.Tensor, *, batch_size: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute a train-sample mean and scalar norm calibration."""

    if activations.ndim != 2 or activations.shape[0] == 0:
        raise ValueError("Expected a nonempty activation matrix.")
    total = torch.zeros(activations.shape[1], dtype=torch.float64)
    for start in range(0, len(activations), batch_size):
        total += activations[start : start + batch_size].to(torch.float64).sum(dim=0)
    mean = (total / len(activations)).to(torch.float32)
    squared_norm = 0.0
    for start in range(0, len(activations), batch_size):
        centered = activations[start : start + batch_size].to(torch.float32) - mean
        squared_norm += float(centered.square().sum().item())
    expected_squared_norm = squared_norm / len(activations)
    scale = torch.tensor(
        (activations.shape[1] / max(expected_squared_norm, 1e-12)) ** 0.5,
        dtype=torch.float32,
    )
    return (
        mean,
        scale,
        {
            "expected_centered_squared_norm": expected_squared_norm,
            "scale": float(scale.item()),
        },
    )
