"""A compact TopK sparse autoencoder and training/evaluation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TopKSAE(nn.Module):
    """Untied TopK SAE with unit-norm decoder feature directions."""

    def __init__(self, input_dim: int, feature_count: int, k: int) -> None:
        super().__init__()
        if not 0 < k <= feature_count:
            raise ValueError("k must lie in [1, feature_count]")
        self.input_dim = int(input_dim)
        self.feature_count = int(feature_count)
        self.k = int(k)
        decoder = torch.randn(feature_count, input_dim) / math.sqrt(input_dim)
        decoder = F.normalize(decoder, dim=1)
        self.decoder_weight = nn.Parameter(decoder)
        self.encoder_weight = nn.Parameter(decoder.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(feature_count))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def pre_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        centered = inputs - self.decoder_bias
        return F.linear(centered, self.encoder_weight, self.encoder_bias)

    def encode(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.pre_activations(inputs)
        values, indices = torch.topk(pre, self.k, dim=-1, sorted=False)
        values = F.relu(values)
        return values, indices, pre

    def decode_sparse(
        self, values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        directions = self.decoder_weight[indices]
        return torch.einsum("bk,bkd->bd", values, directions) + self.decoder_bias

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values, indices, pre = self.encode(inputs)
        reconstruction = self.decode_sparse(values, indices)
        return reconstruction, values, indices, pre

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder_weight.copy_(F.normalize(self.decoder_weight, dim=1))

    @torch.no_grad()
    def remove_decoder_parallel_gradients_(self) -> None:
        if self.decoder_weight.grad is None:
            return
        parallel = (self.decoder_weight.grad * self.decoder_weight).sum(
            dim=1, keepdim=True
        )
        self.decoder_weight.grad.sub_(parallel * self.decoder_weight)


@dataclass
class RunningFeatureActivity:
    steps_since_active: torch.Tensor
    activation_counts: torch.Tensor

    @classmethod
    def create(
        cls, feature_count: int, device: torch.device
    ) -> "RunningFeatureActivity":
        return cls(
            steps_since_active=torch.zeros(
                feature_count, dtype=torch.long, device=device
            ),
            activation_counts=torch.zeros(
                feature_count, dtype=torch.long, device=device
            ),
        )

    @torch.no_grad()
    def update(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        self.steps_since_active.add_(1)
        active = indices[values > 0]
        if active.numel() == 0:
            return
        unique, counts = torch.unique(active, return_counts=True)
        self.steps_since_active[unique] = 0
        self.activation_counts.index_add_(0, unique, counts.to(torch.long))


def auxiliary_dead_feature_loss(
    model: TopKSAE,
    residual: torch.Tensor,
    pre_activations: torch.Tensor,
    dead_mask: torch.Tensor,
    auxiliary_k: int,
) -> torch.Tensor:
    """Reconstruct the detached main residual using only currently dead features."""

    dead_count = int(dead_mask.sum().item())
    if dead_count == 0 or auxiliary_k <= 0:
        return residual.new_zeros(())
    k = min(int(auxiliary_k), dead_count)
    masked = pre_activations.masked_fill(~dead_mask.unsqueeze(0), -torch.inf)
    values, indices = torch.topk(masked, k, dim=-1, sorted=False)
    values = F.relu(values)
    auxiliary = torch.einsum("bk,bkd->bd", values, model.decoder_weight[indices])
    denominator = residual.detach().square().mean().clamp_min(1e-8)
    return (auxiliary - residual.detach()).square().mean() / denominator


@torch.no_grad()
def reconstruction_metrics(
    model: TopKSAE,
    batches: Any,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> dict[str, Any]:
    squared_error = 0.0
    input_squared = 0.0
    input_sum = torch.zeros(model.input_dim, dtype=torch.float64)
    token_count = 0
    l0_total = 0
    feature_counts = torch.zeros(model.feature_count, dtype=torch.long)
    for cpu_batch in batches:
        inputs = cpu_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            reconstruction, values, indices, _ = model(inputs)
        error = reconstruction.float() - inputs
        squared_error += float(error.square().sum().item())
        input_squared += float(inputs.square().sum().item())
        input_sum += inputs.double().sum(dim=0).cpu()
        token_count += int(inputs.shape[0])
        active = values > 0
        l0_total += int(active.sum().item())
        chosen = indices[active].detach().cpu()
        if chosen.numel():
            feature_counts.index_add_(
                0, chosen, torch.ones_like(chosen, dtype=torch.long)
            )
    element_count = token_count * model.input_dim
    mse = squared_error / max(1, element_count)
    centered_energy = input_squared - float(input_sum.square().sum().item()) / max(
        1, token_count
    )
    explained = 1.0 - squared_error / max(centered_energy, 1e-12)
    return {
        "token_count": token_count,
        "mse": mse,
        "explained_variance": explained,
        "mean_l0": l0_total / max(1, token_count),
        "dead_feature_fraction": float((feature_counts == 0).float().mean().item()),
        "activation_frequency": (feature_counts.float() / max(1, token_count)).tolist(),
    }
