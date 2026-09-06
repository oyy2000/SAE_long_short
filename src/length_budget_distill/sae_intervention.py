"""Common-prefix generation with residual-stream SAE feature interventions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    mode: str
    strength: float


class SAEInterventionController:
    """Forward-hook controller for constant enhancement or gated suppression."""

    def __init__(
        self,
        *,
        torch_module: Any,
        checkpoint_path: str | Path,
        layer_module: Any,
        short_feature_ids: Sequence[int],
        long_feature_ids: Sequence[int],
        random_feature_ids: Sequence[int],
        k: int,
        maximum_delta_fraction: float,
        device: Any,
        dtype: Any,
    ) -> None:
        from safetensors.torch import load_file

        torch = torch_module
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        self.torch = torch
        self.layer_module = layer_module
        self.k = int(k)
        self.maximum_delta_fraction = float(maximum_delta_fraction)
        self.scale = float(checkpoint["activation_scale"].item())
        self.activation_mean = checkpoint["activation_mean"].to(
            device=device, dtype=dtype
        )
        self.decoder_bias = checkpoint["decoder_bias"].to(device=device, dtype=dtype)
        self.encoder_weight = checkpoint["encoder_weight"].to(
            device=device, dtype=dtype
        )
        self.encoder_bias = checkpoint["encoder_bias"].to(device=device, dtype=dtype)
        decoder = checkpoint["decoder_weight"]
        self.short_direction = _normalized_decoder_sum(
            torch, decoder, short_feature_ids, device=device, dtype=dtype
        )
        self.random_direction = _normalized_decoder_sum(
            torch, decoder, random_feature_ids, device=device, dtype=dtype
        )
        self.long_feature_ids = torch.tensor(
            list(long_feature_ids), dtype=torch.long, device=device
        )
        self.long_decoder = decoder[list(long_feature_ids)].to(
            device=device, dtype=dtype
        )
        self.feature_to_long_slot = torch.full(
            (int(checkpoint["encoder_bias"].numel()),),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.feature_to_long_slot[self.long_feature_ids] = torch.arange(
            len(long_feature_ids), dtype=torch.long, device=device
        )
        self._handle = None
        self._spec = InterventionSpec("no_steering", "none", 0.0)
        self._reset_diagnostics()

    @contextlib.contextmanager
    def activate(self, spec: InterventionSpec):
        if self._handle is not None:
            raise RuntimeError("An SAE intervention hook is already active.")
        self._spec = spec
        self._reset_diagnostics()
        if spec.mode != "none":
            self._handle = self.layer_module.register_forward_hook(self._hook)
        try:
            yield self
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "mode": self._spec.mode,
            "strength": float(self._spec.strength),
            "continuation_forward_tokens": self.forward_tokens,
            "modified_tokens": self.modified_tokens,
            "gated_feature_events": self.gated_feature_events,
            "mean_delta_to_hidden_norm_fraction": self.delta_fraction_sum
            / max(self.forward_tokens, 1),
            "max_delta_to_hidden_norm_fraction": self.delta_fraction_max,
        }

    def _reset_diagnostics(self) -> None:
        self.forward_tokens = 0
        self.modified_tokens = 0
        self.gated_feature_events = 0
        self.delta_fraction_sum = 0.0
        self.delta_fraction_max = 0.0

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        # The first branch forward is a prompt+prefix prefill. Intervention begins
        # only on subsequent one-token cached forwards.
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            return output
        flat = hidden[:, 0, :]
        if self._spec.mode == "short_enhance":
            delta = (
                float(self._spec.strength)
                * self.short_direction.unsqueeze(0)
                / self.scale
            ).expand_as(flat)
            gated_events = flat.shape[0]
        elif self._spec.mode == "random_enhance":
            delta = (
                float(self._spec.strength)
                * self.random_direction.unsqueeze(0)
                / self.scale
            ).expand_as(flat)
            gated_events = flat.shape[0]
        elif self._spec.mode == "long_suppress":
            normalized = (flat - self.activation_mean) * self.scale
            centered = normalized - self.decoder_bias
            pre = self.torch.nn.functional.linear(
                centered, self.encoder_weight, self.encoder_bias
            )
            values, indices = self.torch.topk(
                pre, self.k, dim=-1, sorted=False
            )
            values = self.torch.relu(values)
            slots = self.feature_to_long_slot[indices]
            selected = (slots >= 0) & (values > 0)
            safe_slots = slots.clamp_min(0)
            directions = self.long_decoder[safe_slots]
            weighted = values * selected.to(values.dtype)
            delta_normalized = self.torch.einsum(
                "bk,bkd->bd", weighted, directions
            )
            delta = -float(self._spec.strength) * delta_normalized / self.scale
            gated_events = int(selected.sum().item())
        else:
            return output
        delta = self._clamp_delta(delta, flat)
        fractions = delta.float().norm(dim=-1) / flat.float().norm(
            dim=-1
        ).clamp_min(1e-12)
        modified = fractions > 0
        self.forward_tokens += int(flat.shape[0])
        self.modified_tokens += int(modified.sum().item())
        self.gated_feature_events += int(gated_events)
        self.delta_fraction_sum += float(fractions.sum().item())
        self.delta_fraction_max = max(
            self.delta_fraction_max, float(fractions.max().item())
        )
        changed = hidden.clone()
        changed[:, 0, :] = flat + delta.to(dtype=flat.dtype)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    def _clamp_delta(self, delta: Any, hidden: Any) -> Any:
        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        maximum = (
            self.maximum_delta_fraction
            * hidden.float().norm(dim=-1, keepdim=True)
        )
        multiplier = self.torch.minimum(
            self.torch.ones_like(delta_norm), maximum / delta_norm.clamp_min(1e-12)
        )
        return delta * multiplier.to(dtype=delta.dtype)


def generate_common_prefix_branches(
    *,
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    controller: SAEInterventionController,
    prompts: Sequence[str],
    specs: Sequence[InterventionSpec],
    prefix_seed: int,
    continuation_seed: int,
    common_prefix_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, list[dict[str, Any]]]:
    """Generate one natural prefix and matched stochastic continuations."""

    torch = torch_module
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_width = int(inputs["input_ids"].shape[1])
    _set_seed(torch, prefix_seed)
    with torch.inference_mode():
        prefix_output = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(common_prefix_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prefix_width = int(prefix_output.shape[1] - prompt_width)
    prefix_tokens = prefix_output[:, prompt_width:]
    eos_id = int(tokenizer.eos_token_id)
    completed = [bool((row == eos_id).any().item()) for row in prefix_tokens]
    active_indices = [index for index, value in enumerate(completed) if not value]
    result = {spec.name: [None] * len(prompts) for spec in specs}

    for index in range(len(prompts)):
        if completed[index]:
            ids = _through_first_eos(prefix_tokens[index].tolist(), eos_id)
            record = _decode_record(tokenizer, ids, prefix_width, completed_in_prefix=True)
            for spec in specs:
                result[spec.name][index] = {
                    **record,
                    "intervention_diagnostics": {
                        "mode": spec.mode,
                        "strength": float(spec.strength),
                        "continuation_forward_tokens": 0,
                        "modified_tokens": 0,
                        "gated_feature_events": 0,
                        "mean_delta_to_hidden_norm_fraction": 0.0,
                        "max_delta_to_hidden_norm_fraction": 0.0,
                    },
                }
    if not active_indices:
        return result

    active = torch.tensor(active_indices, device=prefix_output.device, dtype=torch.long)
    branch_inputs = prefix_output.index_select(0, active)
    branch_attention = torch.cat(
        (
            inputs["attention_mask"].index_select(0, active),
            torch.ones(
                (len(active_indices), prefix_width),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            ),
        ),
        dim=1,
    )
    continuation_budget = max(1, int(max_new_tokens) - prefix_width)
    for spec in specs:
        _set_seed(torch, continuation_seed)
        with controller.activate(spec), torch.inference_mode():
            generated = model.generate(
                input_ids=branch_inputs,
                attention_mask=branch_attention,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=continuation_budget,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        diagnostics = controller.diagnostics()
        for active_slot, original_index in enumerate(active_indices):
            ids = generated[active_slot, prompt_width:].tolist()
            ids = _through_first_eos(ids, eos_id)
            result[spec.name][original_index] = {
                **_decode_record(
                    tokenizer, ids, prefix_width, completed_in_prefix=False
                ),
                "intervention_diagnostics": dict(diagnostics),
            }
    if any(value is None for rows in result.values() for value in rows):
        raise AssertionError("Common-prefix generation left an unfilled result cell.")
    return result


def _normalized_decoder_sum(
    torch: Any,
    decoder: Any,
    feature_ids: Sequence[int],
    *,
    device: Any,
    dtype: Any,
) -> Any:
    direction = decoder[list(feature_ids)].float().sum(dim=0)
    direction = direction / direction.norm().clamp_min(1e-12)
    return direction.to(device=device, dtype=dtype)


def _through_first_eos(token_ids: list[int], eos_id: int) -> list[int]:
    if eos_id in token_ids:
        return token_ids[: token_ids.index(eos_id)]
    return token_ids


def _decode_record(
    tokenizer: Any,
    token_ids: list[int],
    prefix_width: int,
    *,
    completed_in_prefix: bool,
) -> dict[str, Any]:
    return {
        "response": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "response_token_ids": [int(value) for value in token_ids],
        "output_token_count": len(token_ids),
        "common_prefix_token_count": min(prefix_width, len(token_ids)),
        "completed_in_prefix": bool(completed_in_prefix),
    }


def _set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
