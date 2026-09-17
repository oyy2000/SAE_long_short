"""Repeat the original feature discovery rules for each newly trained SAE seed."""
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import logging
import os
import platform
import shutil
import subprocess

from .experiment_io import read_json, validated_artifact_marker
from .factorial import file_sha256, canonical_sha256
from .ncsu_reproduction import save, seal, verify, isolated_gpu_preflight

CODE=Path(__file__).resolve().parents[2]
SCORER='scripts/2_8_score_short_long_features.py'
LIBRARY='src/length_budget_distill/sae_feature_analysis.py'


def feature_selection(candidates, count):
    result={direction:[int(r['feature_id']) for r in sorted(candidates,key=lambda x:int(x['discovery_rank']))
        if r['confirmed'] and r['direction']==direction][:count] for direction in ('short','long')}
    return {'features':result,'required_count_per_direction':count,
        'both_directions_feasible':all(len(ids)==count for ids in result.values()),
        'short_direction_feasible':len(result['short'])==count,
        'selection':'Original discovery rank among held-out-confirmed features; no replacement by cosine matches.'}


def prepare(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root']);seeds=Path(cfg['seed_root'])
    if root.exists():raise FileExistsError(root)
    marker=Path(cfg['launch_root'])/'FROZEN.json';verify(marker)
    for name in ('FROZEN','SOURCES','LEGACY_SOURCES'):verify(seeds/'protocol'/f'{name}.json')
    parent=Path(read_json(seeds/'protocol/frozen_config.json')['parent_root']);legacy=seeds/'legacy_code'
    old_config_path=parent/'sae_features/protocol/frozen_protocol.json';old_config=read_json(old_config_path)
    old_score=parent/'sae_features/feature_scores/layer_17_k_064';summary=read_json(old_score/'scoring_summary.json')
    validated_artifact_marker(old_score/'FEATURE_SCORING_COMPLETE',expected_status='complete',
        hash_bindings={'summary_sha256':old_score/'scoring_summary.json'})
    if file_sha256(old_config_path)!=summary['config_sha256']:
        raise ValueError('Historical feature protocol changed')
    for relative,digest in [(SCORER,summary['source_code_sha256']),(LIBRARY,summary['library_code_sha256'])]:
        if file_sha256(legacy/relative)!=digest:raise ValueError('Historical feature scorer source changed')
    packages={'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','safetensors')}}
    if packages!=summary['runtime']['packages']:raise ValueError('Feature-scoring dependencies changed')
    bindings=[old_config_path,old_score/'FEATURE_SCORING_COMPLETE',old_score/'scoring_summary.json']
    for artifact in summary['artifacts'].values():
        p=Path(artifact['path'])
        if file_sha256(p)!=artifact['sha256']:raise ValueError('Historical feature scoring output changed')
        bindings.append(p)
    manifest=read_json(parent/'sae/token_samples/sample_manifest.json')
    chunks=next(r for r in manifest['layers'] if r['layer_index']==17)['source_chunks']
    for row in chunks:
        p=Path(row['path'])
        if file_sha256(p)!=row['sha256']:raise ValueError('Feature activation chunk changed')
        bindings.append(p)
    root.mkdir(parents=True)
    cfg.update(code_root=str(root/'code'),legacy_code_root=str(legacy),versions=packages,parent_root=str(parent))
    for seed in cfg['additional_seeds']:
        path=root/'inputs'/f'seed_{seed}';training_view=path/'training';training_view.mkdir(parents=True)
        (training_view/'layer_17_k_064').symlink_to(seeds/'training'/f'seed_{seed}',target_is_directory=True)
        view=deepcopy(old_config);view['parent_sae']['training_root']=str(training_view)
        # The unchanged parent config describes corpus/architecture, not a claim
        # that six new dictionaries were trained. Actual seed inputs are checked by score().
        save(path/'feature_config.json',view)
        bindings.append(path/'feature_config.json')
    save(root/'inputs/seed17_selection.json',feature_selection(read_json(old_score/'discovered_features.json')['candidates'],cfg['feature_count']))
    save(root/'inputs/source_scope.json',{'activation_chunks':len(chunks),
        'activation_bytes':sum(Path(r['path']).stat().st_size for r in chunks),
        'training_seed_views':{str(s):str(seeds/'training'/f'seed_{s}') for s in cfg['additional_seeds']},
        'adaptation':'Retain the original corpus/architecture protocol and original six-SAE completion as corpus provenance. Redirect only training_root to the separately validated new seed artifact; no new six-SAE pilot is claimed.'})
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',[Path(config_path),marker,seeds/'protocol/FROZEN.json',seeds/'protocol/LEGACY_SOURCES.json',
        root/'protocol/SOURCES.json',root/'protocol/frozen_config.json',*bindings,
        root/'inputs/seed17_selection.json',root/'inputs/source_scope.json'],stage='sae_seed_feature_inputs',stability_complete=False)
    logging.info('Same-corpus feature scoring frozen for seeds %s, %d activation chunks',cfg['additional_seeds'],len(chunks))


def score(config_path,seed):
    cfg=read_json(config_path);root=Path(cfg['result_root']);seeds=Path(cfg['seed_root']);legacy=Path(cfg['legacy_code_root'])
    if CODE!=Path(cfg['code_root']) or seed not in cfg['additional_seeds']:raise ValueError('Unregistered seed or source')
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json');verify(seeds/'protocol/LEGACY_SOURCES.json')
    training=seeds/'training'/f'seed_{seed}';verify(training/'COMPLETE.json')
    expected=read_json(seeds/'inputs'/f'seed_{seed}'/'training_config.json')
    metrics=read_json(training/'training_metrics.json')
    if expected['sae']['seed']!=seed or metrics['config_hash']!=canonical_sha256(expected):raise ValueError('Feature scorer uses a different seed')
    view=root/'inputs'/f'seed_{seed}'
    if (view/'training/layer_17_k_064').resolve()!=training.resolve():raise ValueError('Training view redirected')
    actual={'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','safetensors')}}
    if actual!=cfg['versions']:raise ValueError('Feature runtime changed')
    out=root/'feature_scores'/f'seed_{seed}';execution=root/'execution'/f'seed_{seed}'
    if out.exists() or execution.exists():raise FileExistsError('Preserve existing feature scoring attempt')
    execution.mkdir(parents=True);isolated_gpu_preflight(config_path,execution,expected_name='L40S')
    command=[cfg['runtime']['python'],str(legacy/SCORER),'--config',str(view/'feature_config.json'),
        '--layer-index','17','--k','64','--output-dir',str(out)]
    environment=os.environ.copy();environment['PYTHONPATH']=cfg['runtime']['overlay']+':'+str(legacy/'src')
    save(execution/'command.json',{'command':command,'actual_seed':seed,'actual_training_marker':str(training/'COMPLETE.json')})
    subprocess.run(command,env=environment,check=True)
    summary=read_json(out/'scoring_summary.json')
    validated_artifact_marker(out/'FEATURE_SCORING_COMPLETE',expected_status='complete',hash_bindings={'summary_sha256':out/'scoring_summary.json'})
    for row in summary['artifacts'].values():
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Feature artifact hash mismatch')
    if summary['input_evidence']['checkpoint_path']!=metrics['model_path']:
        raise ValueError('Feature scorer encoded a different checkpoint')
    selection=feature_selection(read_json(out/'discovered_features.json')['candidates'],cfg['feature_count'])
    save(execution/'selection.json',{'seed':seed,**selection,'discovery_questions':summary['discovery_question_count'],
        'confirmation_questions':summary['confirmation_question_count'],'generation_stability_complete':False})
    seal(execution/'COMPLETE.json',[root/'protocol/FROZEN.json',training/'COMPLETE.json',
        *sorted(execution.glob('*')),*sorted(out.glob('*'))],stage='sae_seed_feature_scoring',stability_complete=False)
