"""Audited lexical probes and paired intervention calibration after SAE training."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import time

from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256, runtime_metadata
from .legacy_replication import ROOT, config_load, root_for, marker_read, marker_write, gpu_capacity
from .legacy_sae_continuation import require_replication, _validate_sources
from .sae_intervention import InterventionSpec


def settings(path):
    overlay = read_json(path)
    parent = config_load(str(ROOT / overlay['parent_config']))
    return overlay, parent, root_for(parent) / overlay['output_subdirectory']


def seed_for(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31-1)


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(path):
    config = read_json(path)
    for filename, digest in config['sources'].items():
        if file_sha256(filename) != digest:
            raise ValueError(f'Calibration source changed: {filename}')
    for item in config['inputs']:
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError(f'Calibration input changed: {item["path"]}')
    return config


def register(path, dependency):
    overlay, parent, out = settings(path)
    destination = root_for(parent) / 'SAE_CALIBRATION_REGISTERED.json'
    if destination.exists():
        raise FileExistsError(destination)
    job = subprocess.check_output(['sbatch', '--parsable', '--partition=a6000', '--nodelist=c31',
        f'--dependency=afterok:{dependency}', '--export',
        f'ALL,CALIBRATION_CONFIG={Path(path).resolve()},CALIBRATION_ACTION=prepare',
        str(ROOT / 'scripts/slurm/1_37_legacy_sae_calibration.sh')], text=True).strip().split(';')[0]
    write_json_exclusive(destination, {'job_id': job, 'afterok': str(dependency),
        'overlay_sha256': file_sha256(path), 'status': 'dependency_queued_not_running',
        'scope': 'lexical probes and dev calibration only; main generation and SFT remain separate'})
    print(job, flush=True)


def prepare(path):
    overlay, parent, out = settings(path)
    require_replication(parent)
    _validate_sources(parent)
    sae_root = root_for(parent) / 'sae'
    summary_path = sae_root / 'analysis/sae_summary.json'
    marker_path = sae_root / 'SAE_TRAINING_AND_SCORING_COMPLETE'
    mark = marker_read(marker_path)
    if mark.get('status') != 'complete' or mark.get('summary_sha256') != file_sha256(summary_path):
        raise ValueError('SAE training/scoring has not passed its completion audit.')
    corpus_path = sae_root / 'corpus/mixed_trajectories.jsonl'
    corpus = read_rows(corpus_path)
    dev_ids = sorted({r['problem_id'] for r in corpus if r['question_split'] == 'dev'},
                     key=lambda q: seed_for(overlay['base_seed'], q, 'dev-selection'))
    dev_ids = dev_ids[:parent['intervention']['calibration_questions']]
    if len(dev_ids) != parent['intervention']['calibration_questions']:
        raise ValueError('Not enough distinct SAE dev questions.')
    tasks, selections, inputs = [], {}, [summary_path, marker_path, corpus_path, Path(path).resolve(),
                                       ROOT / overlay['parent_config']]
    summary = read_json(summary_path)
    reference = str(overlay['reference_sae_seed'])
    for condition in parent['sae']['conditions']:
        score_dir = sae_root / f'seed_{reference}/condition_scores/{condition}'
        checkpoint = sae_root / f'seed_{reference}/checkpoints/{condition}/sae_model.safetensors'
        discovered_path = score_dir / 'discovered_features.json'
        candidates = read_json(discovered_path)['candidates']
        stable = summary['stable_features']
        ids_by_direction = {}
        for direction in ('short', 'long'):
            eligible = {int(r['feature_ids_by_seed'][reference]) for r in stable
                        if r['condition'] == condition and r['direction'] == direction}
            ordered = sorted((r for r in candidates if r['feature_id'] in eligible),
                             key=lambda r: (r['discovery_rank'], r['feature_id']))
            ids_by_direction[direction] = [r['feature_id'] for r in ordered[:overlay['maximum_features_per_direction']]]
        excluded = {r['feature_id'] for r in candidates}
        rng = random.Random(seed_for(overlay['base_seed'], condition, 'random-features'))
        pool = [i for i in range(overlay['dictionary_feature_count']) if i not in excluded]
        ids_by_direction['random'] = rng.sample(pool, len(ids_by_direction['short']))
        selections[condition] = {**ids_by_direction, 'checkpoint': str(checkpoint)}
        inputs.extend([checkpoint, discovered_path])
        # Missing directions are recorded, never silently replaced by unstable features.
        modes = [d for d in ('short', 'long') if len(ids_by_direction[d]) >= overlay['minimum_stable_features_per_direction']]
        selections[condition]['available_directions'] = modes
        if not modes:
            continue
        for prefix in parent['intervention']['prefix_lengths']:
            for shard in range(overlay['question_shards']):
                tasks.append({'condition': condition, 'prefix_tokens': prefix, 'shard': shard,
                              'problem_ids': dev_ids[shard::overlay['question_shards']]})
    source_files = ['src/length_budget_distill/legacy_sae_calibration.py',
                    'src/length_budget_distill/sae_lexical_probe.py',
                    'src/length_budget_distill/sae_paired_intervention.py',
                    'src/length_budget_distill/sae_intervention.py',
                    'src/length_budget_distill/verifiers.py',
                    'scripts/1_37_legacy_sae_calibration.py',
                    'scripts/slurm/1_37_legacy_sae_calibration.sh',
                    'scripts/slurm/1_38_legacy_sae_calibration_worker.sh']
    teacher_path = ROOT / overlay['teacher_config']
    inputs.append(teacher_path)
    config = {'overlay': overlay, 'parent_intervention': parent['intervention'],
              'teacher': read_json(teacher_path)['teacher'], 'layer_index': parent['sae']['layer_index'],
              'k': parent['sae']['k'], 'tasks': tasks, 'features': selections, 'dev_problem_ids': dev_ids,
              'corpus_path': str(corpus_path), 'output_root': str(out),
              'sources': {str(ROOT / f): file_sha256(ROOT / f) for f in source_files},
              'inputs': [{'path': str(p), 'sha256': file_sha256(p)} for p in inputs],
              'status': 'prepared', 'formal_claim_allowed': False}
    out.mkdir(exist_ok=False)
    protocol = out / 'frozen_protocol.json'
    write_json_exclusive(protocol, config)
    if not tasks:
        write_json_exclusive(out / 'NO_STABLE_FEATURES.json', {'status': 'not_launched',
            'reason': 'No direction has a confirmed three-seed stable feature; thresholds were not relaxed.'})
        return
    jobs = []
    for index, node in enumerate(overlay['worker_nodes']):
        jobs.append(subprocess.check_output(['sbatch', '--parsable', '--partition', node['partition'],
            '--nodelist', node['node'], '--export',
            f'ALL,CALIBRATION_CONFIG={protocol},CALIBRATION_WORKER={index},CALIBRATION_WORKERS={len(overlay["worker_nodes"])}',
            str(ROOT / 'scripts/slurm/1_38_legacy_sae_calibration_worker.sh')], text=True).strip().split(';')[0])
    audit = subprocess.check_output(['sbatch', '--parsable', '--partition=a6000', '--nodelist=c31',
        '--dependency=afterany:' + ':'.join(jobs), '--export',
        f'ALL,CALIBRATION_CONFIG={protocol},CALIBRATION_ACTION=analyze',
        str(ROOT / 'scripts/slurm/1_37_legacy_sae_calibration.sh')], text=True).strip().split(';')[0]
    write_json_exclusive(out / 'SUBMITTED.json', {'workers': jobs, 'analysis_job': audit, 'task_count': len(tasks)})


def specifications(config, condition):
    selected = config['features'][condition]
    setting = config['parent_intervention']
    rows = [InterventionSpec('no_steering', 'none', 0.)]
    for direction, mode, grid in [('short', 'short_enhance', setting['short_strength_grid']),
                                  ('long', 'long_suppress', setting['long_strength_grid']),
                                  ('short', 'random_enhance', setting['short_strength_grid'])]:
        if direction in selected['available_directions']:
            rows.extend(InterventionSpec(f'{mode}__{strength:g}', mode, strength) for strength in grid)
    return rows


def worker(path, index, count, gpu_ids):
    config = validate(path)
    pending = queue.Queue()
    for i in range(len(config['tasks'])):
        if i % count == index:
            pending.put(i)
    def run_gpu(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            while not gpu_capacity(gpu, config['overlay']['minimum_free_mib']):
                print(f'Waiting for memory-fit gpu={gpu}', flush=True)
                time.sleep(30)
            logs = Path(config['output_root']) / 'logs'
            logs.mkdir(exist_ok=True)
            subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=index,memory.used,memory.free', '--format=csv'], check=True)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
            with (logs / f'task_{task:03d}.log').open('x') as handle:
                subprocess.run([sys.executable, str(ROOT / 'scripts/1_37_legacy_sae_calibration.py'),
                                'generate', '--config', str(path), '--task', str(task)],
                               stdout=handle, stderr=subprocess.STDOUT, check=True, env=env, cwd=ROOT)
            pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(pending.qsize(), len(gpu_ids))) as pool:
        list(pool.map(run_gpu, gpu_ids))


def generate(path, task_index):
    from .sae_paired_intervention import MeasuredSAEController, generate_cache_cloned_branches
    from .verifiers import extract_final_answer, verify_answer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    config = validate(path)
    task = config['tasks'][task_index]
    teacher = config['teacher']
    tokenizer = AutoTokenizer.from_pretrained(teacher['snapshot_path'], local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(teacher['snapshot_path'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True).to('cuda:0').eval()
    selected = config['features'][task['condition']]
    all_features = selected['short'] + selected['long'] + selected['random']
    controller = MeasuredSAEController(torch_module=torch, checkpoint_path=selected['checkpoint'],
        layer_module=model.model.layers[config['layer_index']], short_feature_ids=selected['short'],
        long_feature_ids=selected['long'], random_feature_ids=selected['random'],
        measured_feature_ids=all_features, k=config['k'],
        maximum_delta_fraction=config['parent_intervention']['max_delta_norm_fraction'], device=model.device, dtype=torch.bfloat16)
    corpus = read_rows(config['corpus_path'])
    problems = {}
    for row in corpus:
        if row['problem_id'] in task['problem_ids']:
            if row['question_split'] != 'dev':
                raise ValueError('Non-dev question in calibration.')
            problems.setdefault(row['problem_id'], row)
    out = Path(config['output_root']) / f'shards/task_{task_index:03d}'
    out.mkdir(parents=True, exist_ok=False)
    # One lexical diagnostic per dictionary. Other shards are disjoint generation.
    if task['shard'] == 0 and task['prefix_tokens'] == config['parent_intervention']['prefix_lengths'][0]:
        from .sae_lexical_probe import lexical_probe
        lexical_probe(model, tokenizer, controller, corpus, config, task['condition'], out)
    specs = specifications(config, task['condition'])
    rows = [problems[q] for q in task['problem_ids']]
    records = out / 'generations.jsonl'
    count = 0
    started = time.time()
    with records.open('x') as handle:
        for candidate in range(config['parent_intervention']['calibration_candidates']):
            for offset in range(0, len(rows), config['overlay']['batch_size']):
                batch = rows[offset:offset+config['overlay']['batch_size']]
                seed_parts = (config['overlay']['base_seed'], task['prefix_tokens'], candidate,
                              *(r['problem_id'] for r in batch))
                prefix_seed, continuation_seed = seed_for(*seed_parts, 'prefix'), seed_for(*seed_parts, 'continuation')
                # Measure each branch's own targeted features, not the union.
                # The generator iterates lazily, so switch measured IDs at activation.
                from contextlib import contextmanager
                original_activate = controller.activate
                @contextmanager
                def measured_activate(spec):
                    key = {'short_enhance': 'short', 'long_suppress': 'long', 'random_enhance': 'random'}.get(spec.mode)
                    controller.measured_ids = torch.tensor(selected[key] if key else all_features,
                                                         dtype=torch.long, device=model.device)
                    with original_activate(spec) as active:
                        yield active
                controller.activate = measured_activate
                try:
                    generated = generate_cache_cloned_branches(torch_module=torch, model=model, tokenizer=tokenizer,
                        branches=[(spec, controller) for spec in specs], prompts=[r['prompt'] for r in batch],
                        prefix_seed=prefix_seed, continuation_seed=continuation_seed,
                        common_prefix_tokens=task['prefix_tokens'], max_new_tokens=teacher['max_new_tokens'],
                        temperature=teacher['temperature'], top_p=teacher['top_p'])
                finally:
                    controller.activate = original_activate
                for i, problem in enumerate(batch):
                    hashes = set()
                    for spec in specs:
                        cell = generated[spec.name][i]
                        prefix_hash = canonical_sha256(cell['response_token_ids'][:cell['common_prefix_token_count']])
                        hashes.add(prefix_hash)
                        predicted = extract_final_answer(cell['response'])
                        record = {**cell, 'problem_id': problem['problem_id'], 'candidate_index': candidate,
                            'condition': task['condition'], 'branch': spec.name, 'mode': spec.mode, 'strength': spec.strength,
                            'prefix_tokens': task['prefix_tokens'], 'common_prefix_sha256': prefix_hash,
                            'prefix_seed': prefix_seed, 'continuation_seed': continuation_seed,
                            'question': problem['question'], 'prompt': problem['prompt'], 'answer': problem['answer'],
                            'predicted_answer': predicted, 'is_correct': bool(verify_answer(predicted, problem['answer']))}
                        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1
                    if len(hashes) != 1:
                        raise ValueError('Paired branches have different prefixes.')
                handle.flush()
                print(json.dumps({'task': task_index, 'candidate': candidate, 'offset': offset,
                                  'records': count, 'elapsed_seconds': time.time()-started}), flush=True)
    expected = len(rows) * config['parent_intervention']['calibration_candidates'] * len(specs)
    if count != expected:
        raise ValueError('Calibration record count mismatch.')
    write_json_exclusive(out / 'manifest.json', {'status': 'complete', 'task_index': task_index,
        'config_sha256': file_sha256(path), 'records': count, 'records_sha256': file_sha256(records),
        'runtime': runtime_metadata(), 'elapsed_seconds': time.time()-started})
    marker_write(out / 'GENERATION_COMPLETE', {'status': 'complete', 'manifest_sha256': file_sha256(out / 'manifest.json')})


def analyze(path):
    from collections import defaultdict
    config = validate(path)
    out = Path(config['output_root'])
    groups, seen = defaultdict(list), set()
    paired_hashes = defaultdict(set)
    for i, task in enumerate(config['tasks']):
        directory = out / f'shards/task_{i:03d}'
        mark = marker_read(directory / 'GENERATION_COMPLETE')
        manifest = read_json(directory / 'manifest.json')
        if (mark.get('status') != 'complete' or mark['manifest_sha256'] != file_sha256(directory / 'manifest.json')
            or manifest['config_sha256'] != file_sha256(path)
            or manifest['records_sha256'] != file_sha256(directory / 'generations.jsonl')):
            raise ValueError('Calibration shard evidence mismatch.')
        rows = read_rows(directory / 'generations.jsonl')
        expected = {(q, c, spec.name) for q in task['problem_ids']
                    for c in range(config['parent_intervention']['calibration_candidates'])
                    for spec in specifications(config, task['condition'])}
        actual = {(r['problem_id'], r['candidate_index'], r['branch']) for r in rows}
        if actual != expected or len(actual) != len(rows):
            raise ValueError('Missing, extra, or duplicate calibration records.')
        for r in rows:
            key = (r['condition'], r['prefix_tokens'], r['problem_id'], r['candidate_index'], r['branch'])
            if key in seen:
                raise ValueError('Duplicate across calibration shards.')
            seen.add(key)
            paired_hashes[key[:-1]].add(r['common_prefix_sha256'])
            groups[(r['condition'], r['prefix_tokens'], r['branch'])].append(r)
    if any(len(hashes) != 1 for hashes in paired_hashes.values()):
        raise ValueError('Paired prefix audit failed.')
    summaries = []
    for (condition, prefix, branch), rows in groups.items():
        correct = sum(r['is_correct'] for r in rows)
        tokens = sum(r['output_token_count'] for r in rows)
        diagnostics = [r['intervention_diagnostics'] for r in rows if not r['prefix_completed_before_intervention']]
        before = sum(d['mean_target_activation_before'] for d in diagnostics) / max(1, len(diagnostics))
        after = sum(d['mean_target_activation_after'] for d in diagnostics) / max(1, len(diagnostics))
        summaries.append({'condition': condition, 'prefix_tokens': prefix, 'branch': branch,
            'mode': rows[0]['mode'], 'strength': rows[0]['strength'], 'records': len(rows),
            'accuracy': correct/len(rows), 'mean_length': tokens/len(rows),
            'correct_mean_length': sum(r['output_token_count'] for r in rows if r['is_correct']) / max(1, correct),
            'correct_usable_per_million_response_tokens': correct*1e6/max(1, tokens),
            'coverage': len({r['problem_id'] for r in rows if r['is_correct']})/len(config['dev_problem_ids']),
            'target_activation_before': before, 'target_activation_after': after,
            'hit_max_new_tokens_rate': sum(r['hit_max_new_tokens'] for r in rows)/len(rows),
            'diagnostics_scope': 'batch aggregate, replicated per example; descriptive, no token-level CI',
            'cost_scope': 'response tokens including duplicated shared prefixes; not amortized wall-clock cost'})
    selected = []
    for condition in config['features']:
        for mode in ('short_enhance', 'long_suppress'):
            eligible = []
            for r in summaries:
                if r['condition'] != condition or r['mode'] != mode:
                    continue
                baseline = next(b for b in summaries if b['condition'] == condition and b['prefix_tokens'] == r['prefix_tokens'] and b['mode'] == 'none')
                correct_safe = r['accuracy'] >= baseline['accuracy'] - config['parent_intervention']['max_accuracy_drop']
                activated = (r['target_activation_after'] > r['target_activation_before']) if mode == 'short_enhance' else (r['target_activation_after'] < r['target_activation_before'])
                if correct_safe and activated:
                    eligible.append(r)
            winner = max(eligible, key=lambda r: (r['correct_usable_per_million_response_tokens'], r['accuracy'], -r['strength'])) if eligible else None
            selected.append({'condition': condition, 'mode': mode, 'selected': winner,
                             'status': 'calibration_candidate' if winner else 'no_eligible_intervention'})
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=False)
    write_json_exclusive(analysis / 'calibration_summary.json', {'status': 'complete', 'groups': summaries,
        'selection': selected, 'formal_claim_allowed': False,
        'main_generation_status': 'not_started_pending_lexical_and_calibration_review'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in config['features']:
        for mode in ('none', 'short_enhance', 'long_suppress', 'random_enhance'):
            rows = [r for r in summaries if r['condition'] == condition and r['mode'] == mode]
            if rows:
                axes[0].scatter([r['mean_length'] for r in rows], [100*r['accuracy'] for r in rows], label=f'{condition}: {mode}', s=22)
                axes[1].scatter([r['target_activation_before'] for r in rows], [r['target_activation_after'] for r in rows], s=22)
    axes[0].set(xlabel='Mean output tokens (all traces)', ylabel='Correctness (%)')
    axes[1].set(xlabel='Target activation before intervention', ylabel='Target activation after intervention')
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(analysis / 'calibration_effects.png', dpi=180)
    fig.savefig(analysis / 'calibration_effects.pdf')
    plt.close(fig)
    marker_write(out / 'CALIBRATION_COMPLETE', {'status': 'complete', 'records': len(seen),
        'summary_sha256': file_sha256(analysis / 'calibration_summary.json'),
        'main_generation_status': 'not_started_pending_review'})
