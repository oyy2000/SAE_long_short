"""Registered full-rollout SAE direction, feature-count and timing controls.

Uses the existing norm-matched hook and per-question uniform sampler. A separate
registered protocol is required before extending development to confirmation.
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
from .sae_norm_intervention import NormMatchedController, generate_condition
from .sae_mechanism_readback import unit, response_regions
from .baseline_method_analysis import audit_cohort
from .gsm8k_grading_v3 import grade_gsm8k_response

CODE = Path(__file__).resolve().parents[2]


class WindowDirectionController(NormMatchedController):
    """Apply a named unit direction at a registered per-row generation window."""
    def __init__(self, *, directions, tokenizer, marker_pattern, **kwargs):
        super().__init__(**kwargs)
        self.directions = directions
        self.tokenizer = tokenizer
        self.marker_pattern = marker_pattern

    def begin(self, spec, batch_size):
        if spec['direction'] not in self.directions: raise ValueError('Unknown direction')
        if spec['window'] not in ('full', 'fixed', 'before_marker', 'after_marker'):
            raise ValueError('Unknown intervention window')
        if spec.get('end') is not None and spec['end'] <= spec['start']:
            raise ValueError('Empty intervention window')
        super().begin(spec, batch_size)
        self.direction = self.directions[spec['direction']]
        self.token_history = [[] for _ in range(batch_size)]
        self.marker_steps = [None]*batch_size

    def observe_tokens(self, tokens, step):
        for i, (token, live) in enumerate(zip(tokens.tolist(), self.live.tolist())):
            if not live: continue
            self.token_history[i].append(token)
            if self.marker_steps[i] is None:
                text = self.tokenizer.decode(self.token_history[i], skip_special_tokens=True)
                if re.search(self.marker_pattern, text): self.marker_steps[i] = int(step)

    def _hook(self, module, inputs, output):
        if self.step < self.spec['start'] or (self.spec.get('end') is not None and self.step >= self.spec['end']):
            return output
        original_live = self.live
        window = self.spec['window']
        if window in ('before_marker', 'after_marker'):
            seen = self.torch.tensor([s is not None for s in self.marker_steps], device=self.live.device)
            self.live = original_live & (seen if window == 'after_marker' else ~seen)
        try:
            if not bool(self.live.any()): return output
            return super()._hook(module, inputs, output)
        finally:
            self.live = original_live

    def end(self):
        rows = super().end()
        for row, step in zip(rows, self.marker_steps):
            row.update(first_marker_completed_step=step, marker_observed=step is not None,
                       window=self.spec['window'], window_start=self.spec['start'], window_end=self.spec.get('end'))
        return rows


def direction_feature_sets(sae):
    short = sae['short_features']
    if len(short) != 8 or len(set(short)) != 8: raise ValueError('Expected eight distinct historical short features')
    sets = {'sae_short_8': short}
    sets.update({f'random_sae_{i+1}': ids for i, ids in enumerate(sae['random_feature_sets'])})
    for n in (1, 2, 4): sets[f'sae_top_{n}'] = short[:n]
    # top_1 is the first individual feature; no duplicate single_0 condition.
    for i in range(1, len(short)): sets[f'sae_single_{i}'] = [short[i]]
    for i in range(len(short)): sets[f'sae_leave_out_{i}'] = short[:i]+short[i+1:]
    return sets


def conditions(cfg, family):
    common = dict(dictionary='historical_layer17_k64', mode='short', count=8, start=0, end=None, window='full')
    def spec(name, direction, rho, **extra): return {**common, 'name':name, 'direction':direction, 'rho':rho, **extra}
    baseline = spec('unmodified', 'sae_short_8', 0.)
    if family == 'dose':
        result = [baseline]+[spec(f'{name}__rho{rho:g}', name, rho) for name in cfg['dose_directions'] for rho in cfg['doses']]
    elif family == 'ablation':
        rho = cfg['ablation_rho']; window_tokens = cfg['window_tokens']
        names = ['sae_short_8']+[name for name in direction_feature_sets(cfg['sae']) if name.startswith(('sae_top_', 'sae_single_', 'sae_leave_out_'))]
        result = [baseline]+[spec(name, name, rho) for name in names]
        result += [spec('early_window', 'sae_short_8', rho, window='fixed', end=window_tokens),
                   spec('delayed_window', 'sae_short_8', rho, window='fixed', start=window_tokens),
                   spec('before_marker_window', 'sae_short_8', rho, window='before_marker'),
                   spec('after_marker_window', 'sae_short_8', rho, window='after_marker')]
    else: raise ValueError('Unknown generation family')
    if len({s['name'] for s in result}) != len(result): raise ValueError('Duplicate condition names')
    feature_sets = direction_feature_sets(cfg['sae'])
    for item in result:
        item['injected_feature_ids'] = feature_sets.get(item['direction'])
        item['measured_short_feature_ids'] = cfg['sae']['short_features']
    return result


def validate_confirmation_transfer(cfg, development, reference_rho):
    """Carry the original reference dose and controls to the reserved split."""
    if cfg.get('evaluation_split', 'dev') != 'confirmation':
        raise ValueError('Expected a separately registered confirmation protocol')
    for key in ('generation', 'dose_directions', 'ablation_rho', 'window_tokens',
                'ablation_questions', 'ablation_subset_seed', 'smoke_conditions',
                'shards', 'norm_rounding_tolerance', 'answer_heading_pattern',
                'teacher', 'sae', 'readback_root', 'input_root'):
        if cfg[key] != development[key]:
            raise ValueError('Confirmation changed the development setting: '+key)
    if cfg['doses'] != [reference_rho] or cfg['ablation_rho'] != reference_rho:
        raise ValueError('Confirmation must use the original registered reference dose')
    if cfg['confirmation_plan']['reference_rho'] != reference_rho:
        raise ValueError('Analysis and generation reference doses differ')
    if cfg['confirmation_plan']['primary_metrics'] != ['is_correct', 'generated_tokens', 'reasoning_body']:
        raise ValueError('Unexpected confirmation primary metrics')
    expected_references = ['unmodified'] + [f'{name}__rho{reference_rho:g}'
        for name in cfg['dose_directions'] if name != 'sae_short_8']
    if cfg['confirmation_plan']['references'] != expected_references:
        raise ValueError('Confirmation must retain all five alternative directions')


def select_generation_cohort(rows, split, expected_count):
    if split not in ('dev', 'confirmation'):
        raise ValueError('Generation supports only registered dev/confirmation roles')
    if len({r['problem_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated question ID across mechanism roles')
    selected = [r for r in rows if r['question_split'] == split]
    if len(selected) != expected_count:
        raise ValueError('Mechanism generation cohort count changed')
    return selected


def prepare(config_path):
    cfg = read_json(config_path); project = Path(cfg['project_root']); root = project/cfg['result_root']
    readback = project/cfg['readback_root']; parent = project/cfg['input_root']
    if root.exists(): raise FileExistsError(root)
    verify(readback/'protocol/FROZEN.json'); verify(readback/'protocol/SOURCES.json'); verify(readback/'directions/COMPLETE.json')
    verify(parent/'protocol/FROZEN.json'); verify(parent/'baseline/merged/COMPLETE.json')
    old = read_json(readback/'protocol/frozen_config.json')
    cfg.update(teacher=old['teacher'], sae=old['sae'], readback_root=str(readback), input_root=str(parent),
               result_root=str(root), code_root=str(root/'code'))
    split = cfg.get('evaluation_split', 'dev'); extra_bindings = []
    if split == 'confirmation':
        development = project/cfg['development_generation_root']
        analysis = project/cfg['development_analysis_root']
        for marker in (development/'protocol/FROZEN.json', development/'protocol/SOURCES.json', analysis/'COMPLETE.json'):
            verify(marker); extra_bindings.append(marker)
        parent_config = read_json(parent/'protocol/frozen_config.json')
        validate_confirmation_transfer(cfg, read_json(development/'protocol/frozen_config.json'),
                                       parent_config['sae']['rho_reference'])
    questions = select_generation_cohort(list(read_jsonl(parent/'inputs/questions.jsonl')), split,
                                        cfg.get('evaluation_questions', cfg['dev_questions']))
    smoke = list(read_jsonl(parent/'inputs/smoke_questions.jsonl'))
    if len(smoke) != cfg['smoke_questions'] or set(r['problem_id'] for r in smoke) & set(r['problem_id'] for r in questions):
        raise ValueError('Smoke count or role isolation changed')
    ablation = sorted(questions, key=lambda r:canonical_sha256([cfg['ablation_subset_seed'], r['problem_id']]))[:cfg['ablation_questions']]
    for name, rows in [(split, questions), ('smoke', smoke), ('ablation', ablation)]:
        write_jsonl(root/'inputs'/f'{name}.jsonl', [{**r, 'gold_answer':r['raw_answer']} for r in rows])
    save(root/'inputs/conditions.json', {family:conditions(cfg, family) for family in ('dose', 'ablation')})
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(CODE/folder, root/'code'/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'), symlinks=True)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/FROZEN.json', [Path(config_path), root/'protocol/SOURCES.json', root/'protocol/frozen_config.json',
        readback/'protocol/FROZEN.json', readback/'directions/COMPLETE.json', parent/'protocol/FROZEN.json',
        *extra_bindings, *sorted((root/'inputs').glob('*'))], formal_claim_allowed=False,
        stage='sae_full_generation_controls_protocol', evaluation_split=split)


def load(config_path):
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen generation source')
    return cfg


def build_bundle(cfg, out):
    import torch
    admission(cfg)
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free,driver_version', '--format=csv'],
                               check=True, text=True, capture_output=True)
    save(out/'hardware.json', {'job_id':os.environ['SLURM_JOB_ID'], 'inventory_csv':inventory.stdout,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable'))})
    model, tok = teacher_bundle(cfg); sae = cfg['sae']
    control = WindowDirectionController(directions={}, tokenizer=tok, marker_pattern=cfg['answer_heading_pattern'],
        short_ids=sae['short_features'], long_ids=sae['long_features'], random_ids=sae['random_feature_sets'][0],
        torch_module=torch, checkpoint_path=sae['checkpoint_path'], layer_module=model.model.layers[sae['layer_index']],
        k=sae['top_k'], maximum_delta_fraction=max(cfg['doses']+[cfg['ablation_rho']]), device=model.device, dtype=torch.bfloat16)
    control.directions = {name:unit(control.decoder[ids].float().sum(0)) for name, ids in direction_feature_sets(sae).items()}
    readback = Path(cfg['readback_root']); verify(readback/'directions/COMPLETE.json')
    with np.load(readback/'directions/directions.npz', allow_pickle=False) as data:
        control.directions.update({name:torch.tensor(data[name], device=model.device) for name in data.files})
    torch.cuda.reset_peak_memory_stats()
    return model, tok, control


def generated_regions(tok, text, pattern):
    encoding = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    regions, found = response_regions(text, encoding['offset_mapping'], 'unmodified_generated', pattern)
    return {'region_tokenization':'retokenized decoded text; actual sampled token IDs stored separately',
        'retokenized_tokens':len(encoding['input_ids']), 'has_explicit_answer_marker':found,
        'region_token_counts':{name:int((regions == i).sum()) for i, name in enumerate(('reasoning_body', 'answer_marker', 'answer_suffix'))}}


def run(cfg, family, shard=0, smoke=False):
    root = Path(cfg['result_root']); split = cfg.get('evaluation_split', 'dev')
    source = root/'inputs'/('smoke.jsonl' if smoke else split+'.jsonl' if family == 'dose' else 'ablation.jsonl')
    specs = conditions(cfg, family)
    if smoke:
        by_name = {s['name']:s for s in conditions(cfg, 'dose')+conditions(cfg, 'ablation')}
        specs = [by_name[name] for name in cfg['smoke_conditions']]
    return run_registered(cfg, family, specs, source, shard=shard, smoke=smoke)


def run_registered(cfg, family, specs, source, *, shard=0, smoke=False,
                   smoke_family='dose', bundle_builder=build_bundle, extra_bindings=()):
    """Run explicit frozen conditions through the shared audited rollout loop.

    Callers bind their condition/input/direction definitions before invoking this
    runner. The historical run() supplies its original unchanged specifications.
    """
    import torch
    root = Path(cfg['result_root']); out = root/('smoke' if smoke else 'generation')/family/f'shard_{shard:02d}'
    if not 0 <= shard < cfg['shards'] or (smoke and shard != 0): raise ValueError('Invalid shard')
    if not smoke: verify(root/'smoke'/smoke_family/'shard_00/COMPLETE.json')
    if not specs or len({s['name'] for s in specs}) != len(specs):
        raise ValueError('Empty or repeated rollout conditions')
    split = cfg.get('evaluation_split', 'dev')
    questions = list(read_jsonl(source))
    if not smoke and any(r['question_split'] != split for r in questions):
        raise ValueError('Generation input contains a different question role')
    if not smoke: questions = [r for i, r in enumerate(questions) if i % cfg['shards'] == shard]
    out.mkdir(parents=True, exist_ok=False); model, tok, control = bundle_builder(cfg, out)
    settings = {k:cfg['generation'][k] for k in ('seed', 'max_new_tokens', 'temperature', 'top_p')}
    rows = []; batches = []; start = time.monotonic()
    with (out/'predictions.jsonl').open('x') as handle:
        for offset in range(0, len(questions), cfg['generation']['batch_size']):
            batch = questions[offset:offset+cfg['generation']['batch_size']]; base = None
            for spec in specs:
                batch_id = f'{offset:04d}__{spec["name"]}'
                torch.cuda.synchronize(); before = time.monotonic()
                generated = generate_condition(model, tok, control, spec, batch, settings)
                torch.cuda.synchronize(); elapsed = time.monotonic()-before
                batches.append({'batch_id':batch_id, 'condition':spec['name'], 'questions':len(batch),
                                'generation_wall_seconds':elapsed})
                if spec['rho'] == 0:
                    base = generated
                    if smoke:
                        replay = generate_condition(model, tok, control, spec, batch, settings)
                        if [r['token_ids'] for r in generated] != [r['token_ids'] for r in replay]: raise ValueError('Zero generation replay differs')
                for question, result in zip(batch, generated):
                    result['legacy_grade'] = {k:result[k] for k in ('is_correct', 'predicted_answer')}
                    result.update(grade_gsm8k_response(result['response'], question['answer'], timeout_seconds=cfg['grading_timeout_seconds']))
                    result.update(question=question['question'], gold_answer=question['answer'], family=family,
                        question_split=question['question_split'],
                        candidate_index=0, batch_id=batch_id, amortized_generation_wall_seconds=elapsed/len(batch),
                        generated_tokens=len(result['token_ids'])-int(result['token_ids'][-1] == tok.eos_token_id),
                        **generated_regions(tok, result['response'], cfg['answer_heading_pattern']))
                    if result['diagnostics']['max_delta_to_hidden_norm_fraction'] > spec['rho']+cfg['norm_rounding_tolerance']:
                        raise ValueError('Measured perturbation exceeds registered norm tolerance')
                    if spec['name'] == 'delayed_window':
                        original = next(r for r in base if r['problem_id'] == question['problem_id'])
                        n = min(cfg['window_tokens'], len(original['token_ids']))
                        if result['token_ids'][:n] != original['token_ids'][:n]: raise ValueError('Delayed intervention changed earlier tokens')
                    if spec['name'] == 'after_marker_window':
                        original = next(r for r in base if r['problem_id'] == question['problem_id'])
                        mark = original['diagnostics']['first_marker_completed_step']
                        n = len(original['token_ids']) if mark is None else mark+1
                        if result['token_ids'][:n] != original['token_ids'][:n]: raise ValueError('Answer-window intervention changed pre-marker tokens')
                    rows.append(result); handle.write(json.dumps(result)+'\n')
                handle.flush(); logging.info('SAE generation %s shard=%d batch=%d condition=%s records=%d', family, shard, offset, spec['name'], len(rows))
    for spec in specs: audit_cohort([r for r in rows if r['condition'] == spec['name']], [q['problem_id'] for q in questions])
    if len(rows) != len(questions)*len(specs): raise ValueError('Missing/extra generation records')
    write_jsonl(out/'batches.jsonl', batches)
    summary = {'questions':len(questions), 'predictions':len(rows), 'family':family, 'smoke_only':smoke,
        'conditions':{}, 'elapsed_seconds':time.monotonic()-start,
        'generation_batch_seconds':sum(b['generation_wall_seconds'] for b in batches),
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'cost_note':'Generation batch timing includes the inherited legacy answer parser. New grading and region parsing enter total elapsed time. Smoke replay is additional diagnostic cost.',
        'confirmation_accessed':not smoke and split == 'confirmation', 'evaluation_split':split,
        'formal_claim_allowed':False}
    for spec in specs:
        subset = [r for r in rows if r['condition'] == spec['name']]; n = len(subset)
        summary['conditions'][spec['name']] = {'n':n, 'correct':sum(r['is_correct'] for r in subset),
            'mean_generated_tokens':sum(r['generated_tokens'] for r in subset)/n,
            'cap_hits':sum(r['hit_max_new_tokens'] for r in subset),
            'marker_observed':sum(r['diagnostics']['marker_observed'] for r in subset)}
    save(out/'summary.json', summary)
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', Path(cfg['readback_root'])/'directions/COMPLETE.json',*extra_bindings,
        *sorted(out.glob('*'))], stage='sae_full_generation_controls', smoke_only=smoke, formal_claim_allowed=False)
