"""Hash-bound historical-recipe replication, isolated from the SAE pilot trainer."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import time

from .experiment_io import publish_files_hash_verified, read_json, write_json_exclusive
from .factorial import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def config_load(filename):
    config = read_json(path(filename))
    config['_path'] = str(path(filename).resolve())
    return config


def root_for(config):
    return path(config['result_root'])


def json_lines(filename):
    with Path(filename).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def marker_write(filename, values):
    with Path(filename).open('x') as handle:
        handle.write(''.join(f'{key}={value}\n' for key, value in values.items()))


def marker_read(filename):
    return dict(line.rstrip('\n').split('=', 1) for line in Path(filename).read_text().splitlines() if '=' in line)


def copy_checked(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    digest = file_sha256(source)
    if expected and digest != expected:
        raise ValueError(f'Source hash mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError(f'Conflicting destination: {destination}')
    else:
        # Exclusive creation avoids overwriting either an earlier copy or user files.
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if file_sha256(destination) != digest:
            raise ValueError(f'Copied hash mismatch: {destination}')
    return {'source': str(source), 'path': str(destination), 'sha256': digest}


def source_hashes(config):
    root = root_for(config)
    manifest = read_json(root / 'IMPORT_COMPLETE.json')
    for row in manifest['files']:
        if file_sha256(row['path']) != row['sha256']:
            raise ValueError(f'Imported file changed: {row["path"]}')
    if file_sha256(config['_path']) != manifest['config_sha256']:
        raise ValueError('Experiment config changed after import.')
    runtime_sources = root / 'RUNTIME_SOURCES.json'
    if runtime_sources.exists():
        for filename, digest in read_json(runtime_sources)['hashes'].items():
            if file_sha256(filename) != digest:
                raise ValueError(f'Runtime source changed after submission: {filename}')
    return manifest


def prepare(config):
    root, storage = root_for(config), path(config['storage_root'])
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'Independent result root already exists: {root}')
    storage.mkdir(parents=True, exist_ok=False)
    root.symlink_to(storage, target_is_directory=True)
    for child in ('protocol', 'imported', 'legacy_code', 'replication/configs', 'replication/checkpoints', 'replication/evaluation', 'replication/logs'):
        (root / child).mkdir(parents=True, exist_ok=True)
    legacy = path(config['legacy_project'])
    matrix = legacy / config['legacy_matrix']
    records = []
    for relative in config['legacy_sources']:
        records.append(copy_checked(legacy / relative, root / 'legacy_code' / relative))
    records.append(copy_checked(config['_path'], root / 'protocol/frozen_protocol.json'))
    records.append(copy_checked(matrix / 'protocol/frozen_protocol.json', root / 'imported/historical_protocol.json'))
    records.append(copy_checked(matrix / 'sft_data/dataset_manifest.json', root / 'imported/historical_dataset_manifest.json'))
    records.append(copy_checked(matrix / 'eval/predictions/base.jsonl', root / 'imported/historical_base_predictions.jsonl'))
    raw_config = read_json(path(config['raw_pool_config']))
    raw_ids, raw_count = {}, 0
    for i, row in enumerate(raw_config['source_pool']['raw_shards']):
        item = copy_checked(row['path'], root / f'imported/raw/shard_{i:02d}.jsonl', row['sha256'])
        rows = json_lines(item['path'])
        if len(rows) != row['records']:
            raise ValueError('Raw shard count mismatch.')
        for trace in rows:
            pid = trace.get('problem_id') or trace.get('metadata', {}).get('problem_id')
            raw_ids[pid] = raw_ids.get(pid, 0) + 1
        raw_count += len(rows)
        records.append(item)
    if raw_count != config['expected_raw_trace_count'] or len(raw_ids) != config['expected_question_count'] or set(raw_ids.values()) != {16}:
        raise ValueError(f'Raw pool cardinality mismatch: {raw_count}, {len(raw_ids)}')
    cohorts = []
    runs = []
    for rank in config['ranks']:
        imported = root / f'imported/sft/{rank}.jsonl'
        old_name = f'equal_example__qwen2p5_7b__relative_{rank}__seed_17'
        original = read_json(matrix / f'training/configs/{old_name}.json')
        records.append(copy_checked(original['data']['train_path'], imported, original['factorial_metadata']['train_sha256']))
        rows = json_lines(imported)
        ids = [r['metadata']['problem_id'] for r in rows]
        if len(ids) != config['expected_question_count'] or len(set(ids)) != len(ids):
            raise ValueError('Historical SFT cohort count or uniqueness failed.')
        cohorts.append(ids)
        for seed in config['seeds']:
            historical_name = f'equal_example__qwen2p5_7b__relative_{rank}__seed_{seed}'
            original = read_json(matrix / f'training/configs/{historical_name}.json')
            records.append(copy_checked(matrix / f'training/configs/{historical_name}.json', root / f'imported/run_configs/{rank}_{seed}.json'))
            name = f'{rank}__seed_{seed}'
            run = json.loads(json.dumps(original))
            run['experiment_name'] = name
            run['data']['train_path'] = str(imported)
            run['training']['output_dir'] = str(root / f'replication/checkpoints/{name}')
            # Pin the already-used revision, without changing optimizer/loss defaults.
            run['training']['model_init_kwargs'] = {'revision': config['student_revision'], 'local_files_only': True}
            run['student']['tokenizer_kwargs'] = {'revision': config['student_revision'], 'local_files_only': True}
            run['replication_evidence'] = {'rank': rank, 'seed': seed, 'train_sha256': file_sha256(imported), 'legacy_training_sha256': file_sha256(root / 'legacy_code/src/length_budget_distill/training.py'), 'expected_steps': (len(ids) + 3) // 4}
            run_path = root / f'replication/configs/{name}.json'
            write_json_exclusive(run_path, run)
            runs.append({'name': name, 'rank': rank, 'seed': seed, 'config_path': str(run_path), 'config_sha256': file_sha256(run_path)})
    if not all(ids == cohorts[0] for ids in cohorts) or set(cohorts[0]) != set(raw_ids):
        raise ValueError('Rank/raw question supports differ.')
    old_predictions = json_lines(root / 'imported/historical_base_predictions.jsonl')
    if len(old_predictions) != config['evaluation']['count'] or len({r['problem_id'] for r in old_predictions}) != len(old_predictions):
        raise ValueError('Historical evaluation cohort invalid.')
    # Preserve historical gold answers, not model predictions. This is a pre-sliced
    # local copy of test[50:1319], so the unchanged evaluator receives start-index=0.
    eval_rows = [{'id': r['problem_id'], 'question': r['question'], 'answer': r['gold_answer']} for r in old_predictions]
    eval_path = root / 'imported/locked_evaluation_questions.jsonl'
    with eval_path.open('x') as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    records.append({'source': str(root / 'imported/historical_base_predictions.jsonl'), 'path': str(eval_path), 'sha256': file_sha256(eval_path)})
    eval_config = {'dataset': {'source': 'local_jsonl', 'path': str(eval_path), 'question_field': 'question', 'answer_field': 'answer'}, 'logical_split': config['evaluation']['logical_split']}
    write_json_exclusive(root / 'replication/eval_config.json', eval_config)
    manifest = {'status': 'complete', 'config_sha256': file_sha256(config['_path']), 'raw_trace_count': raw_count, 'question_count': len(cohorts[0]), 'files': records, 'runs': runs, 'eval_config_sha256': file_sha256(root / 'replication/eval_config.json'), 'seed_policy': 'set_seed before legacy model construction; preserve legacy Trainer sampling', 'formal_claim_allowed': False}
    write_json_exclusive(root / 'IMPORT_COMPLETE.json', manifest)
    return manifest


def check_versions(config):
    from importlib.metadata import version
    found = {key: version(key) for key in config['versions']}
    for key, expected in config['versions'].items():
        if found[key].split('+')[0] != expected:
            raise ValueError(f'Environment mismatch: {key}={found[key]}, expected {expected}')
    return found


def _load_legacy_training(filename):
    spec = importlib.util.spec_from_file_location('frozen_legacy_training', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(config, name, runtime):
    manifest = source_hashes(config)
    entry = next(row for row in manifest['runs'] if row['name'] == name)
    if file_sha256(entry['config_path']) != entry['config_sha256']:
        raise ValueError('Run config hash mismatch.')
    run = read_json(entry['config_path'])
    evidence = run['replication_evidence']
    output = path(run['training']['output_dir'])
    if output.exists():
        validate_training(config, name)
        logging.info('validated_training_skip name=%s', name)
        return
    versions = check_versions(config)
    import torch
    import peft
    from transformers import set_seed
    set_seed(int(entry['seed']))
    runtime = Path(runtime)
    if runtime.exists():
        raise FileExistsError(runtime)
    os.environ['LBD_RUNTIME_OUTPUT_DIR'] = str(runtime)
    initial = {}
    original = peft.get_peft_model
    def capture_initial(*args, **kwargs):
        model = original(*args, **kwargs)
        state = peft.get_peft_model_state_dict(model)
        digest = hashlib.sha256()
        for key, tensor in sorted(state.items()):
            digest.update(key.encode())
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        initial['initial_adapter_sha256'] = digest.hexdigest()
        initial['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model
    peft.get_peft_model = capture_initial
    started = time.monotonic()
    try:
        trainer = _load_legacy_training(root_for(config) / 'legacy_code/src/length_budget_distill/training.py').run_trl_sft(run)
    finally:
        peft.get_peft_model = original
    if trainer.state.global_step != evidence['expected_steps'] or not initial:
        raise ValueError('Legacy training step/initialization audit failed.')
    rows = json_lines(run['data']['train_path'])
    metrics = {'status': 'complete', 'run_name': name, 'seed': entry['seed'], 'rank': entry['rank'], 'record_count': len(rows), 'optimizer_steps': trainer.state.global_step, 'completion_token_updates': sum(r['metadata']['solution_token_count'] for r in rows), 'elapsed_seconds': time.monotonic() - started, 'log_history': trainer.state.log_history, 'versions': versions, 'train_sha256': evidence['train_sha256'], 'run_config_sha256': entry['config_sha256'], 'training_source_sha256': evidence['legacy_training_sha256'], 'launcher_source_sha256': file_sha256(Path(__file__)), 'runtime_sources_sha256': file_sha256(root_for(config) / 'RUNTIME_SOURCES.json'), **initial}
    write_json_exclusive(runtime / 'training_metrics.json', metrics)
    hashes = publish_files_hash_verified(runtime, output, ('adapter_config.json', 'adapter_model.safetensors', 'training_metrics.json'))
    marker_write(output / 'TRAIN_COMPLETE', {'status': 'complete', 'run_config_sha256': entry['config_sha256'], 'train_sha256': evidence['train_sha256'], 'training_source_sha256': evidence['legacy_training_sha256'], **{f'{Path(k).stem}_sha256': v for k, v in hashes.items()}})


def validate_training(config, name):
    root = root_for(config)
    manifest = read_json(root / 'IMPORT_COMPLETE.json')
    entry = next(row for row in manifest['runs'] if row['name'] == name)
    output = root / f'replication/checkpoints/{name}'
    mark = marker_read(output / 'TRAIN_COMPLETE')
    run = read_json(entry['config_path'])
    expected = run['replication_evidence']
    if mark.get('status') != 'complete' or mark.get('run_config_sha256') != file_sha256(entry['config_path']) or file_sha256(entry['config_path']) != entry['config_sha256']:
        raise ValueError('Training marker/config invalid.')
    if mark.get('train_sha256') != file_sha256(run['data']['train_path']) or mark.get('training_source_sha256') != expected['legacy_training_sha256']:
        raise ValueError('Training input/source mismatch.')
    for filename in ('adapter_config.json', 'adapter_model.safetensors', 'training_metrics.json'):
        if mark[f'{Path(filename).stem}_sha256'] != file_sha256(output / filename):
            raise ValueError(f'Adapter artifact changed: {filename}')
    return read_json(output / 'training_metrics.json')


def evaluate(config, name, runtime):
    manifest = source_hashes(config)
    root = root_for(config)
    destination = root / f'replication/evaluation/{name}'
    if destination.exists():
        validate_evaluation(config, name)
        return
    if name != 'base':
        validate_training(config, name)
    check_versions(config)
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(root / 'legacy_code/scripts/4_1_eval_model.py'), '--config', str(root / 'replication/eval_config.json'), '--model-name', 'Qwen/Qwen2.5-1.5B-Instruct', '--split', 'test', '--start-index', '0', '--limit', str(config['evaluation']['count']), '--output-jsonl', str(runtime / 'predictions.jsonl'), '--summary-json', str(runtime / 'summary.json'), '--max-new-tokens', str(config['evaluation']['max_new_tokens']), '--batch-size', str(config['evaluation']['batch_size']), '--temperature', '0', '--top-p', '1', '--torch-dtype', 'bfloat16']
    adapter_hash = None
    if name != 'base':
        adapter = root / f'replication/checkpoints/{name}'
        staged = runtime / 'adapter'
        for f in ('adapter_config.json', 'adapter_model.safetensors'):
            copy_checked(adapter / f, staged / f)
        adapter_hash = file_sha256(staged / 'adapter_model.safetensors')
        command += ['--adapter-path', str(staged)]
    subprocess.run(command, check=True, cwd=root / 'legacy_code')
    rows = json_lines(runtime / 'predictions.jsonl')
    gold = json_lines(root / 'imported/locked_evaluation_questions.jsonl')
    if len(rows) != len(gold) or any((r['problem_id'], r['question'], r['gold_answer']) != (g['id'], g['question'], g['answer']) for r, g in zip(rows, gold)):
        raise ValueError('Evaluation cohort mismatch.')
    hashes = publish_files_hash_verified(runtime, destination, ('predictions.jsonl', 'summary.json'))
    marker_write(destination / 'EVALUATION_COMPLETE', {'status': 'complete', 'model_name': name, 'adapter_sha256': adapter_hash or 'base', 'eval_config_sha256': manifest['eval_config_sha256'], **{f'{Path(k).stem}_sha256': v for k, v in hashes.items()}})


def validate_evaluation(config, name):
    root = root_for(config)
    output = root / f'replication/evaluation/{name}'
    mark = marker_read(output / 'EVALUATION_COMPLETE')
    if mark.get('status') != 'complete' or mark.get('model_name') != name or mark.get('eval_config_sha256') != file_sha256(root / 'replication/eval_config.json'):
        raise ValueError('Evaluation marker invalid.')
    for filename in ('predictions.jsonl', 'summary.json'):
        if mark[f'{Path(filename).stem}_sha256'] != file_sha256(output / filename):
            raise ValueError('Evaluation artifact changed.')
    if name != 'base' and mark['adapter_sha256'] != file_sha256(root / f'replication/checkpoints/{name}/adapter_model.safetensors'):
        raise ValueError('Evaluated adapter hash differs from current adapter.')
    rows = json_lines(output / 'predictions.jsonl')
    expected = json_lines(root / 'imported/locked_evaluation_questions.jsonl')
    if len(rows) != len(expected) or any((r['problem_id'], r['question'], r['gold_answer']) != (g['id'], g['question'], g['answer']) for r, g in zip(rows, expected)):
        raise ValueError('Evaluation question identity mismatch.')
    return rows


def gpu_capacity(gpu, minimum):
    command = ['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits']
    for observation in range(2):
        if int(subprocess.check_output(command, text=True).strip()) < minimum:
            return False
        if observation == 0:
            time.sleep(2)
    return True


def worker(config, shard, shards, gpu_ids, runtime):
    manifest = source_hashes(config)
    tasks = manifest['runs'] + [{'name': 'base'}]
    tasks = [r for i, r in enumerate(tasks) if i % shards == shard]
    work = queue.Queue()
    for entry in tasks:
        work.put(entry)
    def on_gpu(gpu):
        failures = []
        while True:
            try:
                entry = work.get_nowait()
            except queue.Empty:
                return failures
            name = entry['name']
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            for stage in (['evaluate'] if name == 'base' else ['train', 'evaluate']):
                while not gpu_capacity(gpu, config['minimum_free_mib']):
                    logging.info('Waiting for memory-fit gpu=%s task=%s stage=%s', gpu, name, stage)
                    time.sleep(30)
                subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], check=True)
                command = [sys.executable, str(ROOT / 'scripts/1_31_legacy_replication.py'), stage, '--config', config['_path'], '--name', name, '--runtime', str(Path(runtime) / name / stage)]
                logfile = root_for(config) / f'replication/logs/{name}_{stage}.log'
                with logfile.open('a') as handle:
                    result = subprocess.run(command, env=env, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    failures.append({'name': name, 'stage': stage, 'returncode': result.returncode})
                    break
            work.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), len(gpu_ids))) as executor:
        failures = sum(list(executor.map(on_gpu, gpu_ids[:len(tasks)])), [])
    output = root_for(config) / f'replication/worker_{shard}.json'
    write_json_exclusive(output, {'status': 'failed' if failures else 'complete', 'tasks': tasks, 'failures': failures, 'node': os.uname().nodename, 'job_id': os.environ.get('SLURM_JOB_ID')})
    if failures:
        raise RuntimeError(f'Worker failures: {failures}')


def gate_decision(short, medium, long, base, settings):
    checks = {'historical_short_within_tolerance': abs(short - settings['historical_short_accuracy']) <= settings['tolerance'] + 1e-12, 'short_above_base': short > base, 'short_above_long': short > long}
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks, 'short': short, 'medium': medium, 'long': long, 'base': base}


def analyze(config):
    from .ranked_multiseed_analysis import crossed_seed_problem_bootstrap
    import statistics
    root = root_for(config)
    manifest = source_hashes(config)
    evaluations, metrics = {}, []
    for entry in manifest['runs']:
        name = entry['name']
        training = validate_training(config, name)
        rows = validate_evaluation(config, name)
        evaluations[name] = {r['problem_id']: r for r in rows}
        metrics.append({'name': name, 'rank': entry['rank'], 'seed': entry['seed'], 'accuracy': statistics.mean(r['is_correct'] for r in rows), 'mean_output_tokens': statistics.mean(r['output_token_count'] for r in rows), **{'training': training}})
    base_rows = validate_evaluation(config, 'base')
    base = {r['problem_id']: r for r in base_rows}
    means = {rank: statistics.mean(r['accuracy'] for r in metrics if r['rank'] == rank) for rank in config['ranks']}
    paired = {}
    for contrast, other in [('short_minus_long', 'long'), ('short_minus_base', 'base')]:
        effects = {}
        for seed in config['seeds']:
            a = evaluations[f'short__seed_{seed}']
            b = base if other == 'base' else evaluations[f'{other}__seed_{seed}']
            effects[seed] = {pid: float(a[pid]['is_correct']) - float(b[pid]['is_correct']) for pid in a}
        paired[contrast] = crossed_seed_problem_bootstrap(effects, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
    initial = {seed: {r['training']['initial_adapter_sha256'] for r in metrics if r['seed'] == seed} for seed in config['seeds']}
    if any(len(values) != 1 for values in initial.values()):
        raise ValueError('Initial LoRA weights differ between same-seed rank conditions.')
    base_accuracy = statistics.mean(r['is_correct'] for r in base_rows)
    decision = gate_decision(means['short'], means['medium'], means['long'], base_accuracy, config['replication_gate'])
    output = root / 'replication/analysis'
    output.mkdir(exist_ok=False)
    report = {'status': 'complete', 'gate': decision, 'runs': metrics, 'paired': paired, 'rank_means': means, 'base_accuracy': base_accuracy, 'formal_claim_allowed': False}
    write_json_exclusive(output / 'replication_report.json', report)
    _plot_replication(config, metrics, means, base_accuracy, output)
    lines = ['# Historical recipe replication', '', f'Gate: {decision["status"]}', '', f'Base accuracy: {100*base_accuracy:.2f}%', '']
    lines += [f'- {rank}: {100*means[rank]:.2f}%' for rank in config['ranks']]
    lines += ['', 'All rank conditions: original 881 questions, final one-epoch legacy TRL adapters.', 'Historical observed GSM8K evaluation; exploratory, not independent confirmation.', '', '![Replication accuracy](replication_accuracy.png)', '']
    (output / 'replication_report.md').write_text('\n'.join(lines))
    marker_write(root / 'replication/REPLICATION_COMPLETE', {'status': 'complete', 'gate_status': decision['status'], 'report_sha256': file_sha256(output / 'replication_report.json'), 'config_sha256': manifest['config_sha256']})
    print(json.dumps(decision, indent=2), flush=True)
    if decision['status'] != 'passed':
        raise SystemExit(3)


def _plot_replication(config, metrics, means, base, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(6, 4))
    ranks = config['ranks']
    ax.bar(ranks, [100*means[r] for r in ranks], color=['#4C78A8', '#72B7B2', '#F58518'])
    for i, rank in enumerate(ranks):
        values = [100*r['accuracy'] for r in metrics if r['rank'] == rank]
        ax.scatter([i]*len(values), values, color='black', s=18, zorder=3)
    ax.axhline(100*base, color='#555555', linestyle='--', label='Base student')
    ax.set(ylabel='GSM8K accuracy (%)', ylim=(55, 80), title='Original 881-question recipe replication')
    ax.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / 'replication_accuracy.png', dpi=180)
    figure.savefig(output / 'replication_accuracy.pdf')
    plt.close(figure)


def submit(config):
    source_hashes(config)
    root = root_for(config)
    if (root / 'SUBMITTED.json').exists():
        raise FileExistsError('Submission already registered.')
    sources = [Path(__file__), ROOT / 'scripts/1_31_legacy_replication.py', ROOT / 'scripts/slurm/1_31_legacy_replication_worker.sh', ROOT / 'scripts/slurm/1_32_legacy_replication_analysis.sh']
    write_json_exclusive(root / 'RUNTIME_SOURCES.json', {'hashes': {str(p): file_sha256(p) for p in sources}})
    jobs = []
    for index, node in enumerate(config['nodes']):
        command = ['sbatch', '--parsable', '--partition', node['partition'], '--nodelist', node['node'], '--export', f'ALL,LEGACY_CONFIG={config["_path"]},LEGACY_SHARD={index},LEGACY_SHARDS={len(config["nodes"])}', str(ROOT / 'scripts/slurm/1_31_legacy_replication_worker.sh')]
        result = subprocess.check_output(command, text=True).strip().split(';')[0]
        jobs.append({'job_id': result, **node, 'shard': index})
        write_json_exclusive(root / f'SUBMISSION_{index}.json', jobs[-1])
    dependencies = ':'.join(j['job_id'] for j in jobs)
    command = ['sbatch', '--parsable', '--partition', 'a6000', '--nodelist', 'c31', '--dependency', f'afterany:{dependencies}', '--export', f'ALL,LEGACY_CONFIG={config["_path"]}', str(ROOT / 'scripts/slurm/1_32_legacy_replication_analysis.sh')]
    analysis_job = subprocess.check_output(command, text=True).strip().split(';')[0]
    payload = {'status': 'submitted', 'workers': jobs, 'analysis_job': analysis_job, 'sae_status': 'not_submitted_waiting_for_replication_gate'}
    write_json_exclusive(root / 'SUBMITTED.json', payload)
    print(json.dumps(payload, indent=2), flush=True)
