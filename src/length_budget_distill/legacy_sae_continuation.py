"""Gate-locked SAE retraining on hash-identical historical activation samples."""
from __future__ import annotations
import concurrent.futures
import copy
import importlib.util
import itertools
import json
import logging
from pathlib import Path
import queue
import subprocess
import sys

from .legacy_replication import ROOT, config_load, root_for, source_hashes, copy_checked, marker_read, marker_write, gpu_capacity
from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256


def require_replication(config):
    source_hashes(config)
    root = root_for(config)
    mark = marker_read(root / 'replication/REPLICATION_COMPLETE')
    report_path = root / 'replication/analysis/replication_report.json'
    report = read_json(report_path)
    if mark.get('status') != 'complete' or mark.get('gate_status') != 'passed' or report['gate']['status'] != 'passed' or mark.get('report_sha256') != file_sha256(report_path):
        raise RuntimeError('Historical replication gate has not passed; SAE execution is prohibited.')
    return {'replication_marker_sha256': file_sha256(root / 'replication/REPLICATION_COMPLETE'), 'replication_report_sha256': file_sha256(report_path)}


def prepare_and_submit(config):
    gate = require_replication(config)
    root = root_for(config)
    out = root / 'sae'
    out.mkdir(exist_ok=False)
    overlay = read_json(ROOT / config['sae']['parent_config'])
    old_root = ROOT / overlay['outputs']['result_root']
    old_protocol_path = old_root / 'protocol/frozen_protocol.json'
    old_protocol = read_json(old_protocol_path)
    imported = source_hashes(config)
    original_raw_hashes = sorted(r['sha256'] for r in old_protocol['source_pool']['raw_shards'])
    imported_raw = [r for r in imported['files'] if '/imported/raw/' in r['path']]
    if sorted(r['sha256'] for r in imported_raw) != original_raw_hashes:
        raise ValueError('Cached activations do not refer to the copied historical raw pool.')
    corpus_path = Path(old_protocol['sampling_ablation']['parent_sae']['corpus_path'])
    corpus = copy_checked(corpus_path, out / 'corpus/mixed_trajectories.jsonl')
    cached = {}
    for condition in config['sae']['conditions']:
        original_manifest = old_root / f'token_samples/{condition}/sample_manifest.json'
        sample = read_json(original_manifest)
        mark = marker_read(original_manifest.parent / 'TOKEN_SAMPLES_COMPLETE')
        if mark.get('status') != 'complete' or mark.get('manifest_sha256') != file_sha256(original_manifest) or sample['config_hash'] != canonical_sha256(old_protocol) or sample['corpus_sha256'] != corpus['sha256']:
            raise ValueError('Cached activation sample provenance mismatch.')
        layer = copy.deepcopy(sample['layers'][0])
        for item in layer['samples']:
            evidence = copy_checked(item['path'], out / f'cached_samples/{condition}/{Path(item["path"]).name}', item['sha256'])
            item['path'] = evidence['path']
        evidence = copy_checked(layer['normalizer_path'], out / f'cached_samples/{condition}/normalizer.safetensors', layer['normalizer_sha256'])
        layer['normalizer_path'] = evidence['path']
        # Raw residual chunks remain read-only shared artifacts, hash-checked here.
        for chunk in layer['source_chunks']:
            if file_sha256(chunk['path']) != chunk['registered_sha256']:
                raise ValueError(f'Cached residual chunk changed: {chunk["path"]}')
        cached[condition] = (sample, layer, original_manifest)
    tasks = []
    for seed in config['sae']['seeds']:
        seed_root = out / f'seed_{seed}'
        protocol_dir = seed_root / 'protocol'
        protocol_dir.mkdir(parents=True)
        protocol = copy.deepcopy(old_protocol)
        protocol['experiment_name'] = f'{config["experiment_name"]}__sae_seed_{seed}'
        protocol['sae']['seed'] = seed
        protocol['sampling_ablation']['primary_sae']['training_seed'] = seed
        protocol['sampling_ablation']['conditions'] = [r for r in overlay['conditions'] if r['name'] in config['sae']['conditions']]
        protocol['sampling_ablation']['parent_sae']['corpus_path'] = corpus['path']
        protocol['sampling_ablation']['outputs'] = {'result_root': str(seed_root), 'checkpoint_root': str(seed_root / 'checkpoints'), 'figure_root': str(seed_root / 'figures')}
        protocol['legacy_replication_parent'] = {**gate, 'import_manifest_sha256': file_sha256(root / 'IMPORT_COMPLETE.json'), 'cached_sample_parent_protocol_path': str(old_protocol_path), 'cached_sample_parent_protocol_sha256': file_sha256(old_protocol_path), 'raw_inputs_are_hash_identical': True}
        protocol_path = protocol_dir / 'frozen_protocol.json'
        write_json_exclusive(protocol_path, protocol)
        digest = canonical_sha256(protocol)
        write_json_exclusive(protocol_dir / 'protocol_manifest.json', {'status': 'frozen', 'config_hash': digest, 'config_sha256': file_sha256(protocol_path), **gate})
        marker_write(protocol_dir / 'PROTOCOL_FROZEN', {'status': 'frozen', 'config_hash': digest, 'config_sha256': file_sha256(protocol_path), 'manifest_sha256': file_sha256(protocol_dir / 'protocol_manifest.json')})
        for condition, (sample, layer, original_manifest) in cached.items():
            sample_dir = seed_root / f'token_samples/{condition}'
            sample_dir.mkdir(parents=True)
            derived = {**copy.deepcopy(sample), 'config_hash': digest, 'config_path': str(protocol_path), 'config_sha256': file_sha256(protocol_path), 'layers': [layer], 'corpus_path': corpus['path'], 'reuse_evidence': {'source_manifest_path': str(original_manifest), 'source_manifest_sha256': file_sha256(original_manifest), 'transformation': 'none; sample bytes unchanged; only training seed and publication root differ'}}
            sample_path = sample_dir / 'sample_manifest.json'
            write_json_exclusive(sample_path, derived)
            marker_write(sample_dir / 'TOKEN_SAMPLES_COMPLETE', {'status': 'complete', 'config_hash': digest, 'manifest_sha256': file_sha256(sample_path)})
            tasks.append({'seed': seed, 'condition': condition, 'config_path': str(protocol_path), 'sample_root': str(sample_dir), 'training_dir': str(seed_root / f'sae_training/{condition}'), 'checkpoint_dir': str(seed_root / f'checkpoints/{condition}')})
    source_files = ['scripts/2_4_train_topk_sae.py', 'scripts/2_17_score_sae_sampling_condition.py', 'scripts/2_18_analyze_sae_sampling_ablation.py', 'src/length_budget_distill/topk_sae.py', 'src/length_budget_distill/sae_ablation.py', 'src/length_budget_distill/sae_feature_analysis.py', 'src/length_budget_distill/legacy_sae_continuation.py']
    write_json_exclusive(out / 'SAE_PREPARED.json', {'tasks': tasks, 'gate': gate, 'source_hashes': {str(ROOT / f): file_sha256(ROOT / f) for f in source_files}})
    # Avoid the currently saturated C30 and independently occupied C32. Both
    # workers still use memory-fit and all available disjoint local GPU slots.
    jobs = []
    for shard, node in enumerate([n for n in config['nodes'] if n['node'] in ('c31', 'c49')]):
        command = ['sbatch', '--parsable', '--partition', node['partition'], '--nodelist', node['node'], '--export', f'ALL,LEGACY_CONFIG={config["_path"]},SAE_SHARD={shard},SAE_SHARDS=2', str(ROOT / 'scripts/slurm/1_35_legacy_sae_worker.sh')]
        jobs.append(subprocess.check_output(command, text=True).strip().split(';')[0])
    audit = subprocess.check_output(['sbatch', '--parsable', '--partition', 'a6000', '--nodelist', 'c31', '--dependency', 'afterany:' + ':'.join(jobs), '--export', f'ALL,LEGACY_CONFIG={config["_path"]},SAE_ACTION=audit', str(ROOT / 'scripts/slurm/1_34_legacy_sae_continuation.sh')], text=True).strip().split(';')[0]
    write_json_exclusive(out / 'SAE_SUBMITTED.json', {'workers': jobs, 'audit_job': audit, 'task_count': len(tasks), 'status': 'submitted'})


def _validate_sources(config):
    manifest = read_json(root_for(config) / 'sae/SAE_PREPARED.json')
    for filename, digest in manifest['source_hashes'].items():
        if file_sha256(filename) != digest:
            raise ValueError(f'SAE source changed after freezing: {filename}')
    return manifest


def worker(config, shard, shards, gpu_ids):
    import os
    import time
    require_replication(config)
    manifest = _validate_sources(config)
    pending = queue.Queue()
    for index, task in enumerate(manifest['tasks']):
        # Rotate the distribution-to-node assignment across SAE seeds, rather
        # than confounding all full-trace runs with one GPU architecture.
        width = len(config['sae']['conditions'])
        if (index // width + index % width) % shards == shard:
            pending.put(task)
    count = pending.qsize()
    def run_gpu(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            while not gpu_capacity(gpu, 12000):
                logging.info('Waiting for SAE memory-fit gpu=%s', gpu)
                time.sleep(30)
            _validate_sources(config)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            name = f'{task["condition"]}__seed_{task["seed"]}'
            logdir = root_for(config) / 'sae/logs'
            logdir.mkdir(exist_ok=True)
            commands = [
                [sys.executable, str(ROOT / 'scripts/2_4_train_topk_sae.py'), '--config', task['config_path'], '--sample-root', task['sample_root'], '--layer-index', str(config['sae']['layer_index']), '--k', str(config['sae']['k']), '--output-dir', task['training_dir'], '--checkpoint-dir', task['checkpoint_dir']],
                [sys.executable, str(ROOT / 'scripts/2_17_score_sae_sampling_condition.py'), '--config', task['config_path'], '--condition', task['condition']]
            ]
            for stage, command in zip(('train', 'score'), commands):
                subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=index,memory.used,memory.free', '--format=csv'], check=True)
                with (logdir / f'{name}_{stage}.log').open('x') as handle:
                    subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
            pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, len(gpu_ids))) as executor:
        list(executor.map(run_gpu, gpu_ids[:count]))


def audit(config):
    import numpy as np
    from .sae_ablation import mutual_decoder_matches
    require_replication(config)
    manifest = _validate_sources(config)
    root = root_for(config) / 'sae'
    spec = importlib.util.spec_from_file_location('shared_sae_visual_analysis', ROOT / 'scripts/2_18_analyze_sae_sampling_ablation.py')
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    runs, summaries = {}, []
    for task in manifest['tasks']:
        protocol = read_json(task['config_path'])
        model_path = Path(task['checkpoint_dir']) / 'sae_model.safetensors'
        training = Path(task['training_dir'])
        mark = marker_read(training / 'SAE_TRAINING_COMPLETE')
        if mark.get('status') != 'complete' or mark.get('model_sha256') != file_sha256(model_path) or mark.get('training_metrics_sha256') != file_sha256(training / 'training_metrics.json'):
            raise ValueError('SAE completion evidence mismatch.')
        metrics = read_json(training / 'training_metrics.json')
        if metrics['config_hash'] != canonical_sha256(protocol):
            raise ValueError('SAE checkpoint protocol mismatch.')
        run = shared._load_condition(Path(task['training_dir']).parents[1], task['condition'], protocol)
        runs[(task['condition'], task['seed'])] = run
        summaries.append({'condition': task['condition'], 'seed': task['seed'], 'test_explained_variance': metrics['final_metrics']['test']['explained_variance'], 'confirmed': sum(bool(r['confirmed']) for r in run['candidates']), 'training_log': metrics['training_log']})
    matches, stable = {}, []
    for condition in config['sae']['conditions']:
        pair_maps = {}
        for a, b in itertools.combinations(config['sae']['seeds'], 2):
            left, right = runs[(condition, a)], runs[(condition, b)]
            if [r['corpus_index'] for r in left['test_rows']] != [r['corpus_index'] for r in right['test_rows']]:
                raise ValueError('Cross-seed trace ordering differs.')
            pair = mutual_decoder_matches(left_name=str(a), right_name=str(b), left_candidates=left['candidates'], right_candidates=right['candidates'], left_decoders=left['decoder'], right_decoders=right['decoder'], left_test_activations=left['test_activations'], right_test_activations=right['test_activations'], minimum_decoder_cosine=.5, minimum_activation_correlation=.5, require_same_direction=True)
            matches[f'{condition}__{a}_{b}'] = pair
            pair_maps[(a,b)] = {int(r['left_feature_id']): r for r in pair if r['stable_match']}
        a,b,c = config['sae']['seeds']
        for fid, ab in pair_maps[(a,b)].items():
            ac = pair_maps[(a,c)].get(fid)
            bc = pair_maps[(b,c)].get(int(ab['right_feature_id']))
            if ac and bc and int(ac['right_feature_id']) == int(bc['right_feature_id']):
                mapped = {a: fid, b: int(ab['right_feature_id']), c: int(ac['right_feature_id'])}
                if all(next(r for r in runs[(condition,s)]['candidates'] if r['feature_id'] == f)['confirmed'] for s,f in mapped.items()):
                    stable.append({'condition': condition, 'direction': ab['left_direction'], 'feature_ids_by_seed': mapped})
    output = root / 'analysis'
    output.mkdir(exist_ok=False)
    write_json_exclusive(output / 'sae_summary.json', {'status': 'complete', 'models': summaries, 'pair_matches': matches, 'stable_features': stable, 'claim_boundary': 'Length-associated candidates only; lexical falsification and causal intervention remain required.', 'formal_claim_allowed': False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, condition in enumerate(config['sae']['conditions']):
        for r in (x for x in summaries if x['condition'] == condition):
            axes[i].plot([x['step'] for x in r['training_log']], [x['dev_explained_variance'] for x in r['training_log']], label=f'seed {r["seed"]}')
        axes[i].set(title=condition.replace('_', ' '), xlabel='Training step', ylabel='Dev explained variance')
        axes[i].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'sae_training_curves.png', dpi=180)
    fig.savefig(output / 'sae_training_curves.pdf')
    plt.close(fig)
    marker_write(root / 'SAE_TRAINING_AND_SCORING_COMPLETE', {'status': 'complete', 'model_count': len(summaries), 'summary_sha256': file_sha256(output / 'sae_summary.json'), 'stable_feature_count': len(stable), 'intervention_status': 'not_started_requires_lexical_and_causal_screening'})


def register(config, dependency):
    root = root_for(config)
    destination = root / 'SAE_CONTINUATION_REGISTERED.json'
    if destination.exists():
        raise FileExistsError(destination)
    job = subprocess.check_output(['sbatch', '--parsable', '--partition', 'a6000', '--nodelist', 'c31', '--dependency', f'afterok:{dependency}', '--export', f'ALL,LEGACY_CONFIG={config["_path"]},SAE_ACTION=prepare', str(ROOT / 'scripts/slurm/1_34_legacy_sae_continuation.sh')], text=True).strip().split(';')[0]
    write_json_exclusive(destination, {'job_id': job, 'afterok': str(dependency), 'requires_hash_verified_replication_pass': True, 'source_sha256': file_sha256(Path(__file__)), 'status': 'dependency_queued_not_running'})
    print(job, flush=True)
