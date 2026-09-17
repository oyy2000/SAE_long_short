"""Train additional SAE seeds through the unchanged historical training entrypoint."""
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import time

from .experiment_io import (read_json, validated_artifact_marker, write_text_exclusive,
                            publish_files_hash_verified)
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify, isolated_gpu_preflight

CODE = Path(__file__).resolve().parents[2]
TRAINER = 'scripts/2_4_train_topk_sae.py'


def seeded_sample_view(parent_config, parent_manifest, seed, parent_config_path, parent_manifest_path):
    """Change only the training seed; activation sample records remain identical."""
    if parent_manifest['config_hash'] != canonical_sha256(parent_config):
        raise ValueError('Parent sample manifest uses a different protocol')
    if not isinstance(seed,int) or isinstance(seed,bool) or seed < 0:
        raise ValueError('SAE seed must be a nonnegative integer')
    config = deepcopy(parent_config); config['sae']['seed'] = seed
    manifest = deepcopy(parent_manifest); manifest['config_hash'] = canonical_sha256(config)
    manifest['training_seed_view'] = {'parent_config_path':str(parent_config_path),
        'parent_config_hash':canonical_sha256(parent_config), 'parent_manifest_path':str(parent_manifest_path),
        'parent_manifest_hash':canonical_sha256(parent_manifest), 'sae_seed':seed,
        'sample_tensors_or_normalizer_changed':False,
        'scope':'Metadata view for a training-only seed change; no resampling or extraction.'}
    return config, manifest


def versions():
    return {'python':platform.python_version(), **{p:importlib.metadata.version(p) for p in ('torch','safetensors')}}


def scratch_requirement(model_bytes, sae):
    """All periodic weights, final weights, one spare copy, and a 1-GiB margin."""
    if model_bytes <= 0 or sae['checkpoint_interval_steps'] <= 0:
        raise ValueError('Invalid checkpoint-storage sizing input')
    files = sae['max_steps']//sae['checkpoint_interval_steps'] + 2
    return {'checkpoint_sized_files':files, 'checkpoint_bytes':model_bytes, 'safety_bytes':2**30,
        'required_bytes':files*model_bytes+2**30, 'concurrent_seed_jobs_on_node':1}


def prepare(config_path):
    cfg = read_json(config_path); root = Path(cfg['result_root']); parent = Path(cfg['parent_root'])
    if root.exists(): raise FileExistsError(root)
    if cfg['seeds'] != [17,42,73,101] or cfg['layer_index'] != 17 or cfg['k'] != 64:
        raise ValueError('Unexpected SAE stability matrix')
    launch_marker = Path(cfg['launch_root'])/'FROZEN.json'; verify(launch_marker)
    verify(parent/'protocol/FROZEN.json')
    parent_config_path = parent/'sae/protocol/frozen_protocol.json'
    manifest_path = parent/'sae/token_samples/sample_manifest.json'
    parent_config = read_json(parent_config_path); manifest = read_json(manifest_path)
    metrics_path = parent/'sae/sae_training/layer_17_k_064/training_metrics.json'
    metrics = read_json(metrics_path); model_path = Path(metrics['model_path'])
    validated_artifact_marker(metrics_path.parent/'SAE_TRAINING_COMPLETE',expected_status='complete',
        hash_bindings={'training_metrics_sha256':metrics_path,'model_sha256':model_path})
    validated_artifact_marker(manifest_path.parent/'TOKEN_SAMPLES_COMPLETE',expected_status='complete',
        hash_bindings={'manifest_sha256':manifest_path})
    if (metrics['config_hash'] != canonical_sha256(parent_config) or parent_config['sae']['seed'] != 17 or
        metrics['config_sha256'] != file_sha256(parent_config_path) or
        metrics['sample_manifest_sha256'] != file_sha256(manifest_path)):
        raise ValueError('Historical seed17 training/input provenance changed')
    actual_versions = versions()
    if actual_versions != metrics['runtime']['packages']:
        raise ValueError('Historical SAE dependencies differ; register a fresh four-seed protocol')
    old_code = parent/'code'; sources = read_json(parent/'protocol/sources.json')
    if file_sha256(old_code/TRAINER) != metrics['source_code_sha256']:
        raise ValueError('Historical training entrypoint differs from the completed run')
    source_files = [p for folder in ('src','scripts') for p in (old_code/folder).rglob('*')
                    if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for p in source_files:
        if str(p) not in sources or file_sha256(p) != sources[str(p)]:
            raise ValueError('Historical source file is unbound or changed: '+str(p))
    layer = next(r for r in manifest['layers'] if r['layer_index'] == cfg['layer_index'])
    sample_paths = []
    for sample in layer['samples']:
        p = Path(sample['path'])
        if file_sha256(p) != sample['sha256']: raise ValueError('Activation sample changed')
        sample_paths.append(p)
    normalizer = Path(layer['normalizer_path'])
    if file_sha256(normalizer) != layer['normalizer_sha256']: raise ValueError('Activation normalization changed')
    root.mkdir(parents=True)
    for folder in ('src','scripts'):
        shutil.copytree(old_code/folder,root/'legacy_code'/folder,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    seal(root/'protocol/LEGACY_SOURCES.json',[p for p in (root/'legacy_code').rglob('*') if p.is_file() and not p.is_symlink()])
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    cfg.update(code_root=str(root/'code'),legacy_code_root=str(root/'legacy_code'),versions=actual_versions)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    for seed in cfg['seeds'][1:]:
        training, view = seeded_sample_view(parent_config,manifest,seed,parent_config_path,manifest_path)
        location = root/'inputs'/f'seed_{seed}'
        save(location/'training_config.json',training); save(location/'sample_manifest.json',view)
        write_text_exclusive(location/'TOKEN_SAMPLES_COMPLETE',
            f"status=complete\nconfig_hash={canonical_sha256(training)}\nmanifest_sha256={file_sha256(location/'sample_manifest.json')}\n")
    save(root/'inputs/seed17_reuse.json',{'seed':17,'reused':True,'model_path':str(model_path),
        'model_sha256':file_sha256(model_path),'metrics_path':str(metrics_path),
        'source_and_dependency_equality_verified':True,'versions':actual_versions,
        'sample_tokens':{r['split']:r['sampled_tokens'] for r in layer['samples']},
        'hardware_scope':'Historical L40S run reused; new runs also request L40S. Record actual UUID/driver; this does not isolate driver/kernel variability.'})
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',[Path(config_path),launch_marker,parent/'protocol/FROZEN.json',
        parent_config_path,manifest_path,metrics_path,model_path,*sample_paths,normalizer,
        root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',root/'protocol/LEGACY_SOURCES.json',
        *[p for p in (root/'inputs').rglob('*') if p.is_file()]],stage='sae_seed_training_inputs',stability_complete=False)
    logging.info('SAE seed17 reuse validated; seeds42/73/101 registered on identical activation samples')


def train(config_path, seed):
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen SAE stability wrapper')
    for name in ('FROZEN','SOURCES','LEGACY_SOURCES'): verify(root/'protocol'/f'{name}.json')
    extra_bindings=[Path(config_path)]
    if cfg.get('execution_overlay_marker'):
        overlay=Path(cfg['execution_overlay_marker']);verify(overlay);extra_bindings.append(overlay)
        parent=read_json(root/'protocol/frozen_config.json')
        if any(cfg[key]!=value for key,value in parent.items() if key!='code_root'):
            raise ValueError('Execution recovery changed a registered training/data setting')
    if seed not in cfg['seeds'][1:]: raise ValueError('Only the three additional SAE seeds may train here')
    if versions() != cfg['versions']: raise ValueError('SAE runtime versions changed')
    out = root/'training'/f'seed_{seed}'; destination = Path(cfg['checkpoint_root'])/f'seed_{seed}'
    if out.exists() or destination.exists(): raise FileExistsError('Preserve existing SAE training attempt')
    scratch_free = shutil.disk_usage(os.environ['TMPDIR']).free
    inputs = root/'inputs'/f'seed_{seed}'
    expected = read_json(inputs/'training_config.json')
    reference = Path(read_json(root/'inputs/seed17_reuse.json')['model_path'])
    storage = scratch_requirement(reference.stat().st_size,expected['sae'])
    required_scratch = max(storage['required_bytes'],cfg['runtime'].get('minimum_free_scratch_bytes',0))
    if scratch_free < required_scratch:
        raise RuntimeError(f'Insufficient job scratch: {scratch_free} < {required_scratch}')
    out.mkdir(parents=True)
    save(out/'storage_preflight.json',{**storage,'free_bytes':scratch_free,'required_bytes':required_scratch})
    isolated_gpu_preflight(config_path,out,expected_name='L40S')
    local = Path(os.environ['TMPDIR'])/f'sae_seed_{seed}'
    legacy = Path(cfg['legacy_code_root'])
    command = [cfg['runtime']['python'],str(legacy/TRAINER),'--config',str(inputs/'training_config.json'),
        '--sample-root',str(inputs),'--layer-index',str(cfg['layer_index']),'--k',str(cfg['k']),
        '--output-dir',str(local/'metrics'),'--checkpoint-dir',str(local/'checkpoints')]
    environment = os.environ.copy(); environment['PYTHONPATH'] = cfg['runtime']['overlay']+':'+str(legacy/'src')
    save(out/'execution.json',{'command':command,'seed':seed,'legacy_code_root':str(legacy),
        'job_id':os.environ['SLURM_JOB_ID'],'versions':versions(),'intermediate_checkpoints':str(local/'checkpoints')})
    started = time.monotonic(); subprocess.run(command,env=environment,check=True)
    import torch
    metrics_path = local/'metrics/training_metrics.json'; metrics = read_json(metrics_path)
    validated_artifact_marker(local/'metrics/SAE_TRAINING_COMPLETE',expected_status='complete',
        hash_bindings={'training_metrics_sha256':metrics_path,'model_sha256':local/'checkpoints/sae_model.safetensors'})
    expected = read_json(inputs/'training_config.json')
    if metrics['config_hash'] != canonical_sha256(expected) or metrics['max_steps'] != expected['sae']['max_steps']:
        raise ValueError('Actual SAE training differs from the seed configuration')
    if metrics['source_code_sha256'] != file_sha256(legacy/TRAINER): raise ValueError('Unexpected SAE trainer source')
    tensors = __import__('safetensors.torch',fromlist=['load_file']).load_file(str(local/'checkpoints/sae_model.safetensors'))
    if not all(bool(torch.isfinite(t).all()) for t in tensors.values()): raise ValueError('Nonfinite final SAE parameters')
    norms = torch.linalg.vector_norm(tensors['decoder_weight'].float(),dim=1)
    if not torch.allclose(norms,torch.ones_like(norms),atol=1e-5,rtol=1e-5): raise ValueError('SAE decoder directions are not normalized')
    hashes = publish_files_hash_verified(local/'checkpoints',destination,('sae_model.safetensors',),attempts=5,wait_seconds=3)
    shutil.copyfile(metrics_path,out/'original_training_metrics.json')
    published = {**metrics,'original_model_path':metrics['model_path'],'model_path':str(destination/'sae_model.safetensors'),
        'publication_note':'Only model_path metadata changed after hash-verified publication; original metrics retained.'}
    save(out/'training_metrics.json',published)
    marker = f"status=complete\nconfig_hash={metrics['config_hash']}\nmodel_sha256={hashes['sae_model.safetensors']}\ntraining_metrics_sha256={file_sha256(out/'training_metrics.json')}\nlayer_index=17\nk=64\nformal_claim_allowed=false\n"
    write_text_exclusive(out/'SAE_TRAINING_COMPLETE',marker); write_text_exclusive(destination/'SAE_TRAINING_COMPLETE',marker)
    save(out/'summary.json',{'seed':seed,'model_path':str(destination/'sae_model.safetensors'),
        'actual_training_steps':metrics['max_steps'],'trainer_seconds':metrics['elapsed_seconds'],
        'stage_seconds':time.monotonic()-started,'final_metrics':metrics['final_metrics'],
        'sample_tensors_changed':False,'feature_alignment_complete':False,'generation_stability_complete':False})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',root/'protocol/LEGACY_SOURCES.json',*extra_bindings,
        *sorted(out.glob('*')),destination/'sae_model.safetensors',destination/'SAE_TRAINING_COMPLETE'],
        stage='additional_sae_seed_training',stability_complete=False)
