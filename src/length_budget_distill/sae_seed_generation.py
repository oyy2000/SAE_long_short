"""Exploratory equal-count generation after original eight-feature infeasibility.

The shared rollout and norm-matched controller are reused. Their activation
readback uses the historical dictionary and is explicitly not new-seed target
engagement; injected directions carry dictionary-specific provenance.
"""
from pathlib import Path
import shutil

import numpy as np

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import save, seal, verify
from .sae_generation_controls import build_bundle, run_registered

CODE=Path(__file__).resolve().parents[2]


def specifications(cfg, summaries):
    lookup={r['seed']:r for r in summaries}
    if set(lookup)!=set(cfg['seeds']) or len(lookup)!=len(summaries):
        raise ValueError('Seed summary support changed')
    n=cfg['feature_count'];reference=cfg['seeds'][0];historical=cfg['sae']['short_features']
    if n!=3 or len(historical)!=8:
        raise ValueError('This exploratory adaptation requires top 3 and the original eight-feature reference')
    common={'mode':'short','start':0,'end':None,'window':'full',
        'readback_dictionary':'historical_seed17_layer17_k64',
        'readback_scope':'Historical reference activations, not injected-seed target engagement'}
    def spec(name,direction,rho,count,dictionary,ids):
        return {**common,'name':name,'direction':direction,'rho':rho,'count':count,
            'dictionary':dictionary,'injected_feature_ids':ids,
            'measured_short_feature_ids':historical[:count]}
    result=[spec('unmodified',f'seed_{reference}_top_{n}',0.,n,f'seed_{reference}',None)]
    for seed in cfg['seeds']:
        ids=lookup[seed]['supplement_short_ids']
        if len(ids)!=n or len(set(ids))!=n or lookup[seed]['confirmed_short']<n:
            raise ValueError('Supplement feature eligibility changed')
        result.append(spec(f'seed_{seed}_top_{n}',f'seed_{seed}_top_{n}',cfg['rho'],n,f'seed_{seed}',ids))
    result.append(spec(f'seed_{reference}_top_8',f'seed_{reference}_top_8',cfg['rho'],8,f'seed_{reference}',historical))
    for name in ('answer_format','dense_reference_minus_generated'):
        result.append(spec(name,name,cfg['rho'],8,'discovery_control',None))
    return result


def prepare(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root']);parent=Path(cfg['parent_generation_root'])
    analysis=Path(cfg['seed_analysis_root']);launch=Path(cfg['launch_root'])/'FROZEN.json'
    if root.exists():raise FileExistsError(root)
    markers=[launch,parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json',analysis/'COMPLETE.json']
    for marker in markers:verify(marker)
    original=read_json(parent/'protocol/frozen_config.json')
    for key in ('teacher','sae','readback_root','generation','answer_heading_pattern',
                'norm_rounding_tolerance','grading_timeout_seconds','window_tokens'):
        cfg[key]=original[key]
    if cfg['rho']!=original['ablation_rho'] or cfg['seeds']!=[17,42,73,101]:
        raise ValueError('Supplement reference dose or seeds changed')
    cfg.update(code_root=str(root/'code'),doses=[cfg['rho']],ablation_rho=cfg['rho'],evaluation_split='confirmation')
    source=parent/'inputs/ablation.jsonl';smoke=parent/'inputs/smoke.jsonl'
    questions=list(read_jsonl(source));smoke_rows=list(read_jsonl(smoke))
    if len(questions)!=cfg['questions'] or len({r['problem_id'] for r in questions})!=len(questions):
        raise ValueError('Exploratory question cohort changed')
    if any(r['question_split']!='confirmation' for r in questions):raise ValueError('Wrong source role')
    if len(smoke_rows)!=cfg['smoke_questions'] or set(r['problem_id'] for r in questions)&set(r['problem_id'] for r in smoke_rows):
        raise ValueError('Smoke overlaps generation questions')
    summaries=list(read_jsonl(analysis/'seed_summary.jsonl'));specs=specifications(cfg,summaries)
    with np.load(analysis/'directions.npz',allow_pickle=False) as data:
        expected={s['direction'] for s in specs if s['dictionary'].startswith('seed_')}
        if set(data.files)!=expected:raise ValueError('Unexpected supplementary direction keys')
        for name in data.files:
            vector=data[name]
            if vector.ndim!=1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1.,atol=1e-6):
                raise ValueError('Invalid supplementary unit direction')
    root.mkdir(parents=True)
    write_jsonl(root/'inputs/evaluation.jsonl',questions);write_jsonl(root/'inputs/smoke.jsonl',smoke_rows)
    save(root/'inputs/conditions.json',{'conditions':specs})
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'protocol/FROZEN.json',[Path(config_path),*markers,source,smoke,analysis/'directions.npz',
        analysis/'seed_summary.jsonl',root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
        *sorted((root/'inputs').glob('*'))],stage='exploratory_equal_count_seed_generation_inputs',formal_claim_allowed=False)


def bundle(cfg,out):
    import torch
    if 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Supplement generation requires its registered H200 route')
    model,tok,control=build_bundle(cfg,out)
    with np.load(Path(cfg['seed_analysis_root'])/'directions.npz',allow_pickle=False) as data:
        control.directions.update({name:torch.tensor(data[name],device=model.device) for name in data.files})
    return model,tok,control


def generate(config_path,shard=0,smoke=False):
    cfg=read_json(config_path);root=Path(cfg['result_root']);analysis=Path(cfg['seed_analysis_root'])
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen supplementary generation source')
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json');verify(analysis/'COMPLETE.json')
    specs=specifications(cfg,list(read_jsonl(analysis/'seed_summary.jsonl')))
    if specs!=read_json(root/'inputs/conditions.json')['conditions']:raise ValueError('Supplement conditions changed')
    return run_registered(cfg,'seed_stability',specs,root/'inputs'/('smoke.jsonl' if smoke else 'evaluation.jsonl'),
        shard=shard,smoke=smoke,smoke_family='seed_stability',bundle_builder=bundle,
        extra_bindings=[analysis/'COMPLETE.json'])
