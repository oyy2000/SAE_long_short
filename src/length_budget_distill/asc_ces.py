"""ASC-CES equations 1--9 (Findings ACL 2026, paper 1828).

The public ASC extraction script implements a different mean-difference baseline.
This implementation learns only a residual vector and keeps the language model
frozen. Unspecified implementation choices are registered in the run config.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch.nn import functional as F


class ResidualAddition:
    """Add a vector at decode states, including the state predicting token one.

    Teacher forcing uses an explicit [batch, sequence] state mask. Cached
    generation modifies only the last state, including on the prompt prefill.
    Returning new tensors avoids mutating residuals used by autograd or hooks.
    """

    def __init__(self, layer, vector):
        self.layer = layer
        self.vector = vector
        self.handle = None
        self.mask = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.mask is None:
            mask = torch.zeros(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
            mask[:, -1] = 1
        else:
            if tuple(self.mask.shape) != tuple(hidden.shape[:2]):
                raise ValueError("Steering mask must match residual states")
            mask = self.mask.to(device=hidden.device, dtype=hidden.dtype)
        changed = hidden + mask.unsqueeze(-1) * self.vector.to(hidden)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    @contextmanager
    def applied(self, mask=None):
        if self.handle is not None:
            raise RuntimeError("Nested steering would add the vector twice")
        self.mask = mask
        self.handle = self.layer.register_forward_hook(self._hook)
        try:
            yield self
        finally:
            self.handle.remove()
            self.handle = None
            self.mask = None


def response_inputs(tokenizer, prompt, response, *, device, max_length):
    """Tokenize the rendered prompt and completion separately, preserving boundary.

    EOS is part of the response likelihood. Oversize traces are rejected, never
    silently truncated into a different calibration target.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    suffix = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix or len(prefix) + len(suffix) > max_length:
        raise ValueError("Empty prompt or calibration sequence exceeds registered cap")
    ids = torch.tensor([prefix + suffix], device=device)
    # Each index denotes the state predicting the next token, not its target.
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, len(prefix) - 1 : -1] = True
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, mask


def response_energy(logits, input_ids, state_mask):
    """Eq. 2: mean response-token NLL, excluding all prompt targets."""
    selected = state_mask[:, :-1].bool()
    if not selected.any():
        raise ValueError("Energy needs at least one response target")
    return F.cross_entropy(logits[:, :-1][selected].float(), input_ids[:, 1:][selected])


def contrastive_energy(short_energy, long_energy):
    """Eq. 3: softplus(E(short) - E(long)); never summed sequence NLL."""
    return F.softplus(short_energy - long_energy)


def forward_kl(base_logits, steered_logits, state_mask, *, chunk_size=64):
    """Eq. 7: full-vocabulary KL(base || steered), mean over valid states.

    Base is detached. Chunking bounds temporary float32 softmax allocations;
    there is no sampled-token or top-k approximation to the KL.
    """
    selected = state_mask.bool()
    base = base_logits[selected].detach()
    steered = steered_logits[selected]
    if base.shape != steered.shape or not base.shape[0]:
        raise ValueError("KL requires aligned nonempty token distributions")
    total = steered.new_zeros((), dtype=torch.float32)
    for offset in range(0, len(base), chunk_size):
        log_base = F.log_softmax(base[offset : offset + chunk_size].float(), dim=-1)
        log_steered = F.log_softmax(steered[offset : offset + chunk_size].float(), dim=-1)
        total = total + (log_base.exp() * (log_base - log_steered)).sum()
    return total / len(base)


def trust_region_penalty(kl, *, epsilon=0.02, coefficient=20.0):
    """Eq. 9: fixed hinge penalty, not a dynamically updated multiplier."""
    return coefficient * torch.relu(kl - epsilon)


def pair_energies(model, tokenizer, controller, row, *, max_length):
    energies = []
    for key in ("concise", "verbose"):
        inputs, mask = response_inputs(tokenizer, row["prompt"], row[key],
                                      device=model.device, max_length=max_length)
        with controller.applied(mask):
            logits = model(**inputs, use_cache=False).logits
        energies.append(response_energy(logits, inputs["input_ids"], mask))
    return energies


def text_kl(model, tokenizer, controller, text, *, max_length):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    if inputs["input_ids"].shape[1] < 2:
        raise ValueError("Generic-text KL sample needs at least two tokens")
    mask = inputs["attention_mask"].bool()
    mask[:, -1] = False
    with torch.no_grad():
        base = model(**inputs, use_cache=False).logits
    with controller.applied(mask):
        steered = model(**inputs, use_cache=False).logits
    return forward_kl(base, steered, mask)
