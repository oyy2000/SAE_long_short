"""Versioned, cache-cloned interventions; historical generation is unchanged.

The last sampled prefix token is deliberately not in the cached forward state.
Replaying that token under the hook intervenes on the FIRST continuation logit.
"""
from __future__ import annotations

import copy
from typing import Any

from .sae_intervention import SAEInterventionController, _set_seed


def clone_cache(cache: Any) -> Any:
    """Independent storage for both legacy tuple and Transformers DynamicCache."""
    if cache is None:
        raise ValueError("Generation did not return a KV cache.")
    return copy.deepcopy(cache)


def cache_length(cache: Any) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


class MeasuredSAEController(SAEInterventionController):
    """Keep the old intervention formula, add measured target-code changes."""

    def __init__(self, *, measured_feature_ids, **kwargs):
        super().__init__(**kwargs)
        self.measured_ids = self.torch.tensor(
            measured_feature_ids, device=kwargs['device'], dtype=self.torch.long)

    def _reset_diagnostics(self):
        super()._reset_diagnostics()
        self.activation_before = 0.0
        self.activation_after = 0.0
        self.measured_values = 0

    def feature_values(self, hidden, feature_ids=None):
        torch = self.torch
        centered = (hidden - self.activation_mean) * self.scale - self.decoder_bias
        pre = torch.nn.functional.linear(centered, self.encoder_weight, self.encoder_bias)
        values, indices = torch.topk(pre, self.k, dim=-1, sorted=False)
        values = values.relu()
        ids = self.measured_ids if feature_ids is None else feature_ids
        # Preserve actual TopK membership, not merely positive encoder logits.
        return ((indices.unsqueeze(-1) == ids) * values.unsqueeze(-1)).sum(dim=-2)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise RuntimeError('A continuation intervention received a prefill; cache contract broken.')
        before = self.feature_values(hidden[:, 0, :]).float()
        changed = super()._hook(module, inputs, output)
        after_hidden = changed[0] if isinstance(changed, tuple) else changed
        after = self.feature_values(after_hidden[:, 0, :]).float()
        self.activation_before += float(before.sum().item())
        self.activation_after += float(after.sum().item())
        self.measured_values += before.numel()
        return changed

    def diagnostics(self):
        return {**super().diagnostics(),
                'measured_feature_values': self.measured_values,
                'mean_target_activation_before': self.activation_before / max(1, self.measured_values),
                'mean_target_activation_after': self.activation_after / max(1, self.measured_values)}


def generate_cache_cloned_branches(*, torch_module, model, tokenizer, branches,
                                  prompts, prefix_seed, continuation_seed,
                                  common_prefix_tokens, max_new_tokens,
                                  temperature, top_p):
    """Return paired branches; `branches` contains (spec, controller) tuples.

    Diagnostics are batch aggregates, explicitly labelled as such. Completed
    prefix rows are copied unchanged; continued rows share cloned KV tensors,
    identical prompt/prefix tokens and the initial continuation RNG state.
    """
    torch = torch_module
    if not 0 < common_prefix_tokens < max_new_tokens:
        raise ValueError('Require 0 < prefix length < total generation budget.')
    names = [spec.name for spec, _ in branches]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate branch names.')
    rendered = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
        for p in prompts]
    inputs = tokenizer(rendered, return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    width = inputs['input_ids'].shape[1]
    generation = dict(do_sample=True, temperature=temperature, top_p=top_p,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                      use_cache=True)
    _set_seed(torch, prefix_seed)
    result = {}
    with torch.inference_mode():
        prefix = model.generate(**inputs, **generation, max_new_tokens=common_prefix_tokens,
                                return_dict_in_generate=True)
        sequence = prefix.sequences
        prefix_width = sequence.shape[1] - width
        if cache_length(prefix.past_key_values) != sequence.shape[1] - 1:
            raise RuntimeError('Unexpected prefix cache length; refusing an off-by-one intervention.')
        eos = int(tokenizer.eos_token_id)
        finished = [(row == eos).any().item() for row in sequence[:, width:]]
        mask = torch.cat([inputs['attention_mask'], torch.ones_like(sequence[:, width:])], dim=1)
        for spec, controller in branches:
            _set_seed(torch, continuation_seed)
            with controller.activate(spec):
                if all(finished):
                    output = sequence
                else:
                    cache = clone_cache(prefix.past_key_values)
                    output = model.generate(input_ids=sequence, attention_mask=mask,
                                            past_key_values=cache, **generation,
                                            max_new_tokens=max_new_tokens-prefix_width)
                    del cache
                diagnostics = controller.diagnostics()
            if cache_length(prefix.past_key_values) != sequence.shape[1] - 1:
                raise RuntimeError('A continuation mutated the shared prefix cache.')
            rows = []
            for index, row in enumerate(output[:, width:]):
                ids = row.tolist()
                ended = eos in ids
                if ended:
                    ids = ids[:ids.index(eos)+1]
                actual_prefix = min(prefix_width, len(ids))
                rows.append({'response': tokenizer.decode(ids, skip_special_tokens=True),
                             'response_token_ids': ids,
                             'output_token_count': len(ids),
                             'common_prefix_token_count': actual_prefix,
                             'prefix_completed_before_intervention': bool(finished[index]),
                             'hit_max_new_tokens': not ended and len(ids) == max_new_tokens,
                             'intervention_diagnostics_scope': 'batch_including_completed_prefix_padding',
                             'intervention_diagnostics': diagnostics})
            result[spec.name] = rows
    return result
