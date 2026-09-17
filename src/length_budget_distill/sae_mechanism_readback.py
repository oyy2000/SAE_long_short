"""Fresh-cohort SAE measurements and independent-prefix causal readback.

The input cohort is immutable. Discovery fits two named control directions;
development measures all registered doses. Confirmation requires a later frozen
generation/selection protocol and is deliberately unavailable in this entrypoint.
"""
from pathlib import Path
from collections import Counter
import json
import logging
import os
import re
import shutil
import subprocess
import time

import numpy as np

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save, seal, verify, admission, teacher_bundle
from .baseline_method_analysis import audit_cohort
from .answer_marker_diagnostic import region_masks
from .sae_paired_intervention import MeasuredSAEController, clone_cache, cache_length

CODE = Path(__file__).resolve().parents[2]


def unit(vector):
    norm = vector.float().norm()
    if not bool(norm.isfinite()) or float(norm) < 1e-10:
        raise ValueError('Nonfinite or empty direction')
    return vector.float() / norm


def sparse_change(before, after, selected_ids):
    """Compare actual positive sparse codes; zero-valued TopK slots are inactive."""
    def mapping(pair):
        indices, values = pair
        return {int(i): float(v) for i, v in zip(indices.tolist(), values.float().tolist()) if v > 0}
    left, right = mapping(before), mapping(after)
    active = set(left) | set(right); target = set(selected_ids)
    return {'active_before': len(left), 'active_after': len(right),
        'active_entered': len(set(right)-set(left)), 'active_exited': len(set(left)-set(right)),
        'active_jaccard': len(set(left)&set(right))/max(1, len(active)),
        'target_l1_change': sum(abs(right.get(i, 0)-left.get(i, 0)) for i in active & target),
        'nontarget_l1_change': sum(abs(right.get(i, 0)-left.get(i, 0)) for i in active-target),
        'target_sum_before': sum(left.get(i, 0) for i in target),
        'target_sum_after': sum(right.get(i, 0) for i in target)}


def response_regions(text, offsets, variant, pattern):
    regions, found = region_masks(text, offsets, pattern)
    if variant == 'no_marker':
        # This synthetic control has a known final newline followed by gold.
        boundary = text.rindex('\n')+1
        regions = np.array([2 if right > boundary else 0 for left, right in offsets], dtype=np.int8)
    return regions, found


def probe_positions(regions):
    """Positions index processed response tokens; -1 is last prompt token.

    The logit always predicts the token *after* the named processed position.
    Separate labels can intentionally share a position on very short bodies.
    """
    if not len(regions):
        raise ValueError('Empty response')
    result = {'first_response_predictor': -1, 'response_end': len(regions)-1}
    body = np.flatnonzero(regions == 0)
    marker = np.flatnonzero(regions == 1)
    if len(body): result['body_middle'] = int(body[(len(body)-1)//2])
    if len(marker):
        result['before_marker'] = int(marker[0])-1
        result['after_marker'] = int(marker[-1])
    return result


def last_state_hook(direction, rho, captured):
    """Modify one cached token, saving the same incoming h and rounded h+delta."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[:2] != (1, 1):
            raise RuntimeError('Readback requires one independently replayed prefix token')
        if captured:
            raise RuntimeError('Unexpected repeated hook')
        before = hidden[:, -1, :]
        changed = (before.float()+float(rho)*before.float().norm(dim=-1, keepdim=True)*direction).to(before.dtype)
        captured.update(before=before.detach().clone(), after=changed.detach().clone())
        replacement = changed.unsqueeze(1)
        return (replacement, *output[1:]) if isinstance(output, tuple) else replacement
    return hook


def independent_replay(model, layer, prefix_ids, branches):
    """Yield actual next logits and local states; clone pristine prefix cache.

    The last token is absent from shared KV state. Every branch recomputes it
    through all layers, so earlier intervention cannot contaminate a later dose.
    """
    import torch
    prefix = torch.tensor([prefix_ids], device=model.device, dtype=torch.long)
    if prefix.shape[1] < 2:
        raise ValueError('Need at least two prefix tokens')
    cached = model.model(input_ids=prefix[:, :-1], use_cache=True, return_dict=True).past_key_values
    if cache_length(cached) != prefix.shape[1]-1:
        raise RuntimeError('Prefix cache off by one')
    original_state = None
    for name, rho, direction in branches:
        captured = {}; handle = layer.register_forward_hook(last_state_hook(direction, rho, captured))
        try:
            branch_cache = clone_cache(cached)
            output = model.model(input_ids=prefix[:, -1:], past_key_values=branch_cache,
                                 use_cache=True, return_dict=True)
            logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()[0]
        finally:
            handle.remove()
        if cache_length(cached) != prefix.shape[1]-1:
            raise RuntimeError('Mutated shared cache')
        if original_state is None: original_state = captured['before'].clone()
        if not torch.equal(original_state, captured['before']):
            raise RuntimeError('Branches did not see the identical incoming hidden state')
        yield name, rho, logits, captured
        del output, branch_cache


def prepare(config_path):
    cfg = read_json(config_path); project = Path(cfg['project_root'])
    root = project/cfg['result_root']; parent = project/cfg['input_root']
    if root.exists(): raise FileExistsError(root)
    verify(parent/'protocol/FROZEN.json'); verify(parent/'protocol/SOURCES.json')
    verify(parent/'baseline/merged/COMPLETE.json'); verify(parent/'baseline/smoke/COMPLETE.json')
    source_cfg = read_json(parent/'protocol/frozen_config.json')
    cfg.update(teacher=source_cfg['teacher'], sae=source_cfg['sae'], input_root=str(parent),
               result_root=str(root), code_root=str(root/'code'))
    if cfg['allowed_splits'] != ['smoke', 'discovery', 'dev']:
        raise ValueError('Confirmation needs a separately frozen protocol')
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(CODE/folder, root/'code'/folder,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'), symlinks=True)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/FROZEN.json', [Path(config_path), root/'protocol/SOURCES.json',
         root/'protocol/frozen_config.json', parent/'protocol/FROZEN.json',
         parent/'baseline/merged/COMPLETE.json', parent/'baseline/smoke/COMPLETE.json'],
         formal_claim_allowed=False, stage='sae_readback_protocol_preparation')


def load(config_path):
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']): raise ValueError('Use the frozen readback snapshot')
    return cfg


def inputs(cfg, split, shard):
    if split not in cfg['allowed_splits']: raise ValueError('Unapproved split')
    if not 0 <= shard < cfg['shards']: raise ValueError('Invalid shard')
    if split == 'smoke' and shard != 0: raise ValueError('Smoke has a single shard')
    parent = Path(cfg['input_root'])
    verify(parent/'protocol/FROZEN.json')
    baseline = parent/'baseline'/('smoke' if split == 'smoke' else 'merged')
    verify(baseline/'COMPLETE.json')
    all_questions = list(read_jsonl(parent/'inputs'/('smoke_questions.jsonl' if split == 'smoke' else 'questions.jsonl')))
    selected = [r for r in all_questions if r['question_split'] == split]
    expected = cfg['expected_counts'][split]
    if len(selected) != expected: raise ValueError('Input cohort count mismatch')
    if split != 'smoke': selected = [r for i, r in enumerate(selected) if i % cfg['shards'] == shard]
    ids = {r['problem_id'] for r in selected}
    traces = audit_cohort([r for r in read_jsonl(baseline/'predictions.jsonl') if r['problem_id'] in ids], ids)
    variants = {}
    for row in read_jsonl(parent/'inputs/reference_variants.jsonl'):
        if row['problem_id'] not in ids: continue
        key = (row['problem_id'], row['variant'])
        if key in variants: raise ValueError('Duplicate reference variant')
        variants[key] = row
    return selected, traces, variants


def bundle(cfg, out):
    import torch
    from safetensors.torch import load_file
    admission(cfg)
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free,driver_version',
        '--format=csv'], check=True, text=True, capture_output=True)
    save(out/'hardware.json', {'job_id': os.environ['SLURM_JOB_ID'], 'inventory_csv': inventory.stdout,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_device_uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable'))})
    model, tok = teacher_bundle(cfg); sae = cfg['sae']; layer = model.model.layers[sae['layer_index']]
    measured = list(dict.fromkeys(sae['short_features']+sae['long_features']+sum(sae['random_feature_sets'], [])))
    controller = MeasuredSAEController(torch_module=torch, checkpoint_path=sae['checkpoint_path'], layer_module=layer,
        short_feature_ids=sae['short_features'], long_feature_ids=sae['long_features'],
        random_feature_ids=sae['random_feature_sets'][0], measured_feature_ids=measured,
        k=sae['top_k'], maximum_delta_fraction=max(cfg['doses']), device=model.device, dtype=torch.bfloat16)
    # Match NormMatchedController: round decoder to BF16, then sum in FP32.
    decoder = load_file(sae['checkpoint_path'], device='cpu')['decoder_weight'].to(model.device, torch.bfloat16)
    directions = {'sae_short_8': unit(decoder[sae['short_features']].float().sum(0))}
    for i, ids in enumerate(sae['random_feature_sets']): directions[f'random_sae_{i+1}'] = unit(decoder[ids].float().sum(0))
    del decoder
    torch.cuda.reset_peak_memory_stats()
    return model, tok, layer, controller, measured, directions


def encode(tok, question, text):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': question['prompt']}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    response = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    # Appended tokenization matches generation's prompt boundary. Decoded original
    # generations are retokenized because the input stage did not save token IDs.
    return prompt_ids, response['input_ids'], response['offset_mapping']


def response_hidden_states(model, layer, ids, prompt_length):
    """Capture response block outputs without allocating vocabulary logits."""
    import torch
    if not 0 < prompt_length < len(ids):
        raise ValueError('A nonempty prompt and response are required')
    captured = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured['hidden'] = h[0, prompt_length:, :].detach().clone()
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, return_dict=True)
    finally: handle.remove()
    return captured['hidden']


def extract_one(model, layer, controller, ids, prompt_length, regions, feature_ids, chunk_size):
    import torch
    hidden = response_hidden_states(model, layer, ids, prompt_length)
    values = torch.cat([controller.feature_values(chunk).float().cpu() for chunk in hidden.split(chunk_size)])
    if len(hidden) != len(regions): raise ValueError('Response activation/token count mismatch')
    region_stats = {}
    for r, name in enumerate(('reasoning_body', 'answer_marker', 'answer_suffix')):
        indices = np.flatnonzero(regions == r)
        selected = values[indices.tolist()]
        region_stats[name] = {'tokens': len(indices), 'feature_sum': selected.sum(0).tolist(),
            'feature_positive_count': (selected > 0).sum(0).tolist()}
    events = [{'position': int(i), 'feature_id': feature_ids[int(j)], 'activation': float(values[i, j])}
              for i, j in (values > 0).nonzero().tolist()]
    marker = np.flatnonzero(regions == 1)
    vectors = {'response_mean': hidden.float().mean(0).cpu().numpy()}
    if len(marker): vectors['marker_end'] = hidden[int(marker[-1])].float().cpu().numpy()
    return region_stats, events, vectors


def extract(cfg, split, shard):
    import torch
    root = Path(cfg['result_root']); out = root/'extraction'/split/f'shard_{shard:02d}'
    if split not in ('smoke', 'discovery'): raise ValueError('Only discovery and smoke extraction is registered')
    if split != 'smoke': verify(root/'probes/smoke/shard_00/COMPLETE.json')
    questions, traces, variants = inputs(cfg, split, shard)
    out.mkdir(parents=True, exist_ok=False)
    model, tok, layer, controller, features, directions = bundle(cfg, out)
    rows = []; vectors = {}; start = time.monotonic()
    with torch.inference_mode(), (out/'features.jsonl').open('x') as handle:
        for question in questions:
            pid = question['problem_id']
            sources = [r for (q, v), r in variants.items() if q == pid]
            sources += [{'variant': 'unmodified_generated', 'text': traces[pid]['solution']}]
            for source in sources:
                variant = source['variant']; text = source['text']
                prompt, response, offsets = encode(tok, question, text)
                if 'token_ids' in source and (source['token_ids'] != response or source['token_offsets'] != [list(x) for x in offsets]):
                    raise ValueError('Frozen reference token/span alignment changed')
                if len(prompt)+len(response) > cfg['maximum_sequence_tokens']: raise ValueError('No silent sequence truncation')
                regions, marker = response_regions(text, offsets, variant, cfg['answer_heading_pattern'])
                stats, events, hidden = extract_one(model, layer, controller, prompt+response, len(prompt), regions,
                                                    features, cfg['encoding_chunk_tokens'])
                for name, vector in hidden.items(): vectors[pid+'__'+variant+'__'+name] = vector
                row = {'problem_id': pid, 'question_split': split, 'variant': variant, 'text_sha256': canonical_sha256(text),
                    'token_ids': response, 'token_offsets': offsets, 'token_regions': regions.tolist(),
                    'has_explicit_marker': marker, 'prompt_tokens': len(prompt), 'feature_ids': features,
                    'region_statistics': stats, 'events': events,
                    'generation_retokenized': variant == 'unmodified_generated',
                    'base_alignment': source.get('base_alignment'), 'equation_mutation': source.get('equation_mutation')}
                rows.append(row); handle.write(json.dumps(row)+'\n')
            handle.flush(); logging.info('SAE extraction %s %s complete', split, pid)
    np.savez_compressed(out/'hidden_summaries.npz', **vectors)
    counts = Counter(r['variant'] for r in rows)
    for variant in cfg['required_variants']:
        audit_cohort([r for r in rows if r['variant'] == variant], [q['problem_id'] for q in questions])
    summary = {'questions': len(questions), 'variant_counts': dict(counts), 'split': split, 'shard': shard,
        'feature_ids': features, 'elapsed_seconds': time.monotonic()-start,
        'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'confirmation_inspected': False, 'formal_claim_allowed': False}
    save(out/'summary.json', summary)
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', *sorted(out.glob('*'))], stage='sae_reference_feature_extraction')


def fit_directions(cfg):
    root = Path(cfg['result_root']); parent = Path(cfg['input_root']); out = root/'directions'
    questions = [r for r in read_jsonl(parent/'inputs/questions.jsonl') if r['question_split'] == 'discovery']
    if len(questions) != cfg['expected_counts']['discovery']: raise ValueError('Discovery cohort changed')
    vectors = {}; markers = []
    for shard in range(cfg['shards']):
        source = root/'extraction/discovery'/f'shard_{shard:02d}'; verify(source/'COMPLETE.json'); markers.append(source/'COMPLETE.json')
        expected = [r['problem_id'] for i, r in enumerate(questions) if i % cfg['shards'] == shard]
        rows = list(read_jsonl(source/'features.jsonl'))
        for variant in cfg['required_variants']: audit_cohort([r for r in rows if r['variant'] == variant], expected)
        with np.load(source/'hidden_summaries.npz', allow_pickle=False) as data:
            if set(data.files) & set(vectors): raise ValueError('Overlapping discovery vectors')
            vectors.update({k: data[k] for k in data.files})
    differences = {'dense_reference_minus_generated': [], 'answer_format': []}
    for question in questions:
        pid = question['problem_id']
        differences['dense_reference_minus_generated'].append(vectors[pid+'__reference__response_mean']-vectors[pid+'__unmodified_generated__response_mean'])
        differences['answer_format'].append(vectors[pid+'__reference__marker_end']-vectors[pid+'__result_marker__marker_end'])
    directions = {}; norms = {}
    for key, values in differences.items():
        vector = np.stack(values).astype(np.float64).mean(0); norms[key] = float(np.linalg.norm(vector))
        if not np.isfinite(norms[key]) or norms[key] < 1e-10: raise ValueError('Invalid control direction')
        directions[key] = (vector/norms[key]).astype(np.float32)
    out.mkdir(parents=True, exist_ok=False); np.savez(out/'directions.npz', **directions)
    save(out/'definition.json', {'questions': [r['problem_id'] for r in questions], 'source_split': 'discovery',
        'unnormalized_norms': norms, 'question_weighting': 'equal weight for each paired question',
        'dense_reference_minus_generated': 'Mean official reference response state minus mean unmodified generated response state; can confound style, content and length.',
        'answer_format': 'Answer-minus-Result residual at the last marker token with identical reasoning body; local marker identity/format direction.',
        'confirmation_inspected': False})
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', *markers, out/'directions.npz', out/'definition.json'], stage='sae_discovery_control_direction_fit')


def probe(cfg, split, shard):
    import torch
    root = Path(cfg['result_root']); out = root/'probes'/split/f'shard_{shard:02d}'
    if split not in ('smoke', 'dev'): raise ValueError('Only smoke/dev causal probes are registered')
    bindings = [root/'protocol/FROZEN.json']
    if split == 'smoke': verify(root/'extraction/smoke/shard_00/COMPLETE.json')
    else:
        verify(root/'probes/smoke/shard_00/COMPLETE.json'); verify(root/'directions/COMPLETE.json')
        bindings.append(root/'directions/COMPLETE.json')
    questions, traces, _ = inputs(cfg, split, shard); out.mkdir(parents=True, exist_ok=False)
    model, tok, layer, controller, features, directions = bundle(cfg, out)
    if split != 'smoke':
        with np.load(root/'directions/directions.npz', allow_pickle=False) as data:
            directions.update({name: torch.tensor(data[name], device=model.device) for name in data.files})
    # Literal next-token probability, not the probability of the whole multi-token string.
    answer_ids = sorted({tok.encode(s, add_special_tokens=False)[0] for s in cfg['answer_token_forms']})
    for form in cfg['answer_token_forms']:
        if len(tok.encode(form, add_special_tokens=False)) != 1: raise ValueError('Answer probe form is not one token: '+repr(form))
    eos_ids = model.generation_config.eos_token_id
    eos_ids = sorted(set(eos_ids if isinstance(eos_ids, list) else [eos_ids]))
    save(out/'token_probability_definitions.json', {'answer_forms': cfg['answer_token_forms'], 'answer_token_ids': answer_ids,
        'eos_token_ids': eos_ids, 'scope': 'Sum over registered individual next-token IDs, before sampling filters.'})
    branches = [('zero', 0., directions['sae_short_8'])]
    branches += [(name, rho, direction) for name, direction in directions.items() for rho in cfg['doses']]
    rows = []; start = time.monotonic(); expected_keys = set()
    with torch.inference_mode(), (out/'measurements.jsonl').open('x') as handle:
        for question in questions:
            pid = question['problem_id']; text = traces[pid]['solution']
            prompt, response, offsets = encode(tok, question, text)
            regions, found = response_regions(text, offsets, 'unmodified_generated', cfg['answer_heading_pattern'])
            if len(prompt)+len(response) > cfg['maximum_sequence_tokens']: raise ValueError('No silent sequence truncation')
            positions = probe_positions(regions)
            for position_name, position in positions.items():
                prefix = prompt+response[:position+1]; base_logp = None; before_codes = None
                for name, rho, logits, captured in independent_replay(model, layer, prefix, branches):
                    logp = logits.log_softmax(-1)
                    if not bool(logp.isfinite().all()): raise ValueError('Nonfinite next logits')
                    if base_logp is None:
                        base_logp = logp.clone()
                        before_codes = tuple(t[0] for t in controller.sparse_codes(captured['before']))
                    after_codes = tuple(t[0] for t in controller.sparse_codes(captured['after']))
                    delta = captured['after'].float()-captured['before'].float()
                    fraction = float(delta.norm()/captured['before'].float().norm().clamp_min(1e-12))
                    kl = float((base_logp.exp()*(base_logp-logp)).sum())
                    if name == 'zero' and (fraction != 0. or abs(kl) > 1e-7): raise ValueError('Zero-dose sanity failed')
                    feature_after = controller.feature_values(captured['after'])[0].float().tolist()
                    feature_before = controller.feature_values(captured['before'])[0].float().tolist()
                    row = {'problem_id': pid, 'question_split': split, 'position_name': position_name,
                        'processed_response_token_index': position, 'prefix_tokens': len(prefix),
                        'prefix_token_sha256': canonical_sha256(prefix), 'response_text_sha256': canonical_sha256(text),
                        'generation_retokenized': True, 'has_explicit_marker': found,
                        'direction': name, 'requested_rho': rho, 'actual_delta_fraction': fraction,
                        'incoming_state_identical_across_branches': True, 'forward_kl': kl,
                        'answer_next_probability': float(logp[answer_ids].exp().sum()),
                        'eos_next_probability': float(logp[eos_ids].exp().sum()),
                        'top_token_id': int(logp.argmax()), 'feature_ids': features,
                        'feature_before': feature_before, 'feature_after': feature_after,
                        **sparse_change(before_codes, after_codes, cfg['sae']['short_features'])}
                    rows.append(row); key = (pid, position_name, name, rho)
                    if key in expected_keys: raise ValueError('Duplicate local readback')
                    expected_keys.add(key); handle.write(json.dumps(row)+'\n')
                if base_logp is None: raise ValueError('No branch readback')
            handle.flush(); logging.info('SAE local probes %s %s positions=%d complete', split, pid, len(positions))
    audit_cohort([{'problem_id': pid} for pid in sorted({r['problem_id'] for r in rows})], [q['problem_id'] for q in questions])
    expected_rows = sum(len(probe_positions(response_regions(traces[q['problem_id']]['solution'],
        encode(tok, q, traces[q['problem_id']]['solution'])[2], 'unmodified_generated', cfg['answer_heading_pattern'])[0]))
        for q in questions)*len(branches)
    if len(rows) != expected_rows: raise ValueError('Missing readback branches')
    save(out/'summary.json', {'questions': len(questions), 'measurements': len(rows), 'expected_measurements': expected_rows,
        'directions': list(directions), 'doses': cfg['doses'], 'split': split, 'shard': shard,
        'elapsed_seconds': time.monotonic()-start, 'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'minimum_forward_kl': min(r['forward_kl'] for r in rows), 'confirmation_inspected': False,
        'causal_scope': 'One local intervention in a fixed unmodified prefix; no rollout length or accuracy effect measured.',
        'target_definition': 'Historical eight short features; all other positive codes count as nontarget.',
        'formal_claim_allowed': False})
    seal(out/'COMPLETE.json', bindings+sorted(out.glob('*')), stage='sae_same_state_next_logit_readback')
