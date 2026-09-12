"""Compare a local smoke adapter with a recovered historical short-data run.

Reuse the sealed legacy trainer, evaluator, publication helpers and paired
bootstrap. Data recovery is accepted only with the historical byte hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from . import legacy_replication as legacy
from .experiment_io import publish_files_hash_verified, read_json, write_json_exclusive
from .factorial import file_sha256
from .utility_analysis import paired_question_bootstrap

ROOT = legacy.ROOT


def require_hash(filename, expected):
    if file_sha256(filename) != expected:
        raise ValueError(f'Hash mismatch: {filename}')


def recovered_jsonl(filename, rows, expected):
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'Recovered bytes differ from historical input: {filename}')
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open('xb') as handle:
        handle.write(payload)
    return {'path': str(filename), 'sha256': expected}


def prepare(config):
    import pyarrow as pa

    root = legacy.root_for(config)
    legacy.check_versions(config)
    require_hash(Path(config['student_snapshot']) / 'model.safetensors', config['student_model_sha256'])
    audit = read_json(legacy.path(config['source_audit']))
    if audit['status'] != 'passed':
        raise ValueError('Historical source audit did not pass.')
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = legacy.path(config['checkpoint_root'])
    checkpoint.mkdir(parents=True, exist_ok=False)
    (root / 'replication').mkdir()
    (root / 'replication/checkpoints').symlink_to(checkpoint, target_is_directory=True)
    (root / 'logs').mkdir()
    files = []
    sources = {r['path'].split('/legacy_code/')[1]: r['sha256']
               for r in audit['hash_checks'] if '/legacy_code/' in r['path']}
    for name, digest in sources.items():
        files.append(legacy.copy_checked(Path(config['source_project']) / name,
                                         root / 'legacy_code' / name, digest))
    with Path(config['short_arrow']).open('rb') as handle:
        rows = pa.ipc.open_stream(handle).read_all().to_pylist()
    if len(rows) != 881 or len({r['metadata']['problem_id'] for r in rows}) != 881:
        raise ValueError('Recovered short cohort is not the original 881 unique questions.')
    files.append(recovered_jsonl(root / 'imported/sft/short.jsonl', rows, config['short_sha256']))
    predictions = legacy.json_lines(legacy.path(config['fixture_source']))
    gold = [{'id': r['problem_id'], 'question': r['question'], 'answer': r['gold_answer']}
            for r in predictions]
    if [r['id'] for r in gold] != [f'hf-{i:06d}' for i in range(50, 1319)]:
        raise ValueError('Locked evaluation IDs or ordering changed.')
    files.append(recovered_jsonl(root / 'imported/locked_evaluation_questions.jsonl', gold,
                                 config['fixture_sha256']))
    smoke = Path(config['smoke_adapter'])
    smoke_mark = legacy.marker_read(smoke / 'TRAIN_COMPLETE')
    smoke_metrics = read_json(smoke / 'training_metrics.json')
    require_hash(smoke_metrics['train_path'], smoke_mark['train_sha256'])
    if (smoke_metrics['record_count'], smoke_metrics['optimizer_steps']) != (4, 1):
        raise ValueError('Local smoke artifact changed scope.')
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'training_metrics.json'):
        files.append(legacy.copy_checked(smoke / name, root / 'imported/local_smoke' / name,
                                         smoke_mark[f'{Path(name).stem}_sha256']))
    files.append(legacy.copy_checked(smoke / 'TRAIN_COMPLETE', root / 'imported/local_smoke/TRAIN_COMPLETE'))
    eval_config = {'dataset': {'source': 'local_jsonl',
                              'path': str(root / 'imported/locked_evaluation_questions.jsonl'),
                              'question_field': 'question', 'answer_field': 'answer'},
                   'logical_split': config['evaluation']['logical_split']}
    write_json_exclusive(root / 'replication/eval_config.json', eval_config)
    name = f"short__seed_{config['seed']}"
    training = dict(config['training'], seed=config['seed'], data_seed=config['seed'],
                    output_dir=str(root / 'replication/checkpoints' / name),
                    model_init_kwargs={'revision': config['student_revision'], 'local_files_only': True})
    run = {'experiment_name': name,
           'data': {'train_path': str(root / 'imported/sft/short.jsonl'),
                    'eval_path': None, 'text_format': 'prompt_completion'},
           'student': {'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
                       'tokenizer_name': 'Qwen/Qwen2.5-1.5B-Instruct',
                       'trust_remote_code': False, 'torch_dtype': 'bfloat16',
                       'use_lora': True, 'lora': config['lora'],
                       'tokenizer_kwargs': {'revision': config['student_revision'], 'local_files_only': True}},
           'training': training,
           'replication_evidence': {'rank': 'short', 'seed': config['seed'],
                                    'train_sha256': config['short_sha256'], 'expected_steps': 221,
                                    'legacy_training_sha256': sources['src/length_budget_distill/training.py']}}
    run_path = root / 'replication/configs' / (name + '.json')
    write_json_exclusive(run_path, run)
    sealed = [Path(__file__), ROOT / 'scripts/1_41_check_local_short_gain.py',
              ROOT / 'scripts/slurm/1_41_local_short_gain_check.sh', Path(legacy.__file__)]
    write_json_exclusive(root / 'RUNTIME_SOURCES.json', {'hashes': {str(p): file_sha256(p) for p in sealed}})
    files.append(legacy.copy_checked(config['_path'], root / 'protocol/frozen_protocol.json'))
    write_json_exclusive(root / 'IMPORT_COMPLETE.json', {
        'status': 'complete', 'config_sha256': file_sha256(config['_path']), 'files': files,
        'runs': [{'name': name, 'rank': 'short', 'seed': config['seed'],
                  'config_path': str(run_path), 'config_sha256': file_sha256(run_path)}],
        'eval_config_sha256': file_sha256(root / 'replication/eval_config.json'),
        'recovery': {'short_arrow': config['short_arrow'], 'short_arrow_sha256': file_sha256(config['short_arrow']),
                     'fixture_source': str(legacy.path(config['fixture_source'])),
                     'fixture_source_sha256': file_sha256(legacy.path(config['fixture_source'])),
                     'historical_audit_sha256': file_sha256(legacy.path(config['source_audit']))},
        'student_model_sha256': config['student_model_sha256'],
        'student_snapshot_hashes': {str(p): file_sha256(p) for p in Path(config['student_snapshot']).iterdir()
                                   if p.suffix in ('.json', '.txt')},
        'formal_claim_allowed': False})


def validate_evaluation(config, arm):
    root = legacy.root_for(config)
    directory = root / 'evaluation' / arm
    marker = legacy.marker_read(directory / 'EVALUATION_COMPLETE')
    if marker['config_sha256'] != file_sha256(config['_path']):
        raise ValueError('Evaluation protocol changed.')
    for name in ('predictions.jsonl', 'summary.json'):
        require_hash(directory / name, marker[f'{Path(name).stem}_sha256'])
    if arm != 'base':
        adapter = root / ('imported/local_smoke' if arm == 'local_smoke' else 'replication/checkpoints/' + arm)
        require_hash(adapter / 'adapter_model.safetensors', marker['adapter_sha256'])
    return validate_prediction_files(config, directory)


def validate_prediction_files(config, directory):
    root = legacy.root_for(config)
    rows = legacy.json_lines(directory / 'predictions.jsonl')
    gold = legacy.json_lines(root / 'imported/locked_evaluation_questions.jsonl')
    if len(rows) != len(gold) or len({r['problem_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicated evaluation cohort.')
    if any((r['problem_id'], r['question'], r['gold_answer']) != (g['id'], g['question'], g['answer'])
           for r, g in zip(rows, gold)):
        raise ValueError('Evaluation questions, gold answers, or ordering changed.')
    summary = read_json(directory / 'summary.json')
    if summary['correct'] != sum(r['is_correct'] for r in rows) or summary['n'] != len(rows):
        raise ValueError('Evaluation summary disagrees with predictions.')
    return rows


def evaluate(config, arm, runtime):
    legacy.source_hashes(config)
    legacy.check_versions(config)
    root = legacy.root_for(config)
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(root / 'legacy_code/scripts/4_1_eval_model.py'),
               '--config', str(root / 'replication/eval_config.json'),
               '--model-name', config['student_snapshot'], '--split', 'test', '--start-index', '0',
               '--limit', str(config['evaluation']['count']), '--max-new-tokens', str(config['evaluation']['max_new_tokens']),
               '--batch-size', str(config['evaluation']['batch_size']), '--temperature', '0', '--top-p', '1',
               '--torch-dtype', 'bfloat16', '--output-jsonl', str(runtime / 'predictions.jsonl'),
               '--summary-json', str(runtime / 'summary.json')]
    adapter_hash = 'base'
    if arm != 'base':
        if arm != 'local_smoke':
            legacy.validate_training(config, arm)
        adapter = root / ('imported/local_smoke' if arm == 'local_smoke' else 'replication/checkpoints/' + arm)
        adapter_hash = file_sha256(adapter / 'adapter_model.safetensors')
        command.extend(['--adapter-path', str(adapter)])
    subprocess.run(command, check=True, cwd=root / 'legacy_code')
    validate_prediction_files(config, runtime)
    destination = root / 'evaluation' / arm
    hashes = publish_files_hash_verified(runtime, destination, ('predictions.jsonl', 'summary.json'))
    legacy.marker_write(destination / 'EVALUATION_COMPLETE', {
        'status': 'complete', 'arm': arm, 'config_sha256': file_sha256(config['_path']),
        'adapter_sha256': adapter_hash, **{f'{Path(k).stem}_sha256': v for k, v in hashes.items()}})
    validate_evaluation(config, arm)


def worker(config, arm):
    if arm not in config['arms']:
        raise ValueError('Unregistered arm.')
    runtime = Path(config['runtime_root']) / arm
    if arm.startswith('short__'):
        legacy.train(config, arm, runtime / 'training')
    evaluate(config, arm, runtime / 'evaluation')
    # Only remove this new arm's disposable runtime after successful publication.
    shutil.rmtree(runtime)


def analyze(config):
    legacy.source_hashes(config)
    training = legacy.validate_training(config, f"short__seed_{config['seed']}")
    root = legacy.root_for(config)
    rows = {arm: validate_evaluation(config, arm) for arm in config['arms']}
    metrics = {}
    for arm, values in rows.items():
        metrics[arm] = {'n': len(values), 'correct': sum(r['is_correct'] for r in values),
                        'accuracy': statistics.fmean(r['is_correct'] for r in values),
                        'mean_output_tokens': statistics.fmean(r['output_token_count'] for r in values),
                        'empty_extracted_answer_count': sum(not str(r['predicted_answer'] or '').strip() for r in values)}
    paired = {}
    for arm in config['arms'][1:]:
        left = {r['problem_id']: float(r['is_correct']) for r in rows[arm]}
        right = {r['problem_id']: float(r['is_correct']) for r in rows['base']}
        paired[arm + '_minus_base'] = paired_question_bootstrap(
            left, right, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
        paired[arm + '_minus_base']['corrected_base_errors'] = sum(left[k] > right[k] for k in left)
        paired[arm + '_minus_base']['lost_base_correct'] = sum(left[k] < right[k] for k in left)
    short = metrics[f"short__seed_{config['seed']}"]['accuracy']
    report = {'status': 'complete', 'metrics': metrics, 'paired': paired, 'training': training,
              'historical_short_mean': config['historical_short_mean'],
              'short_minus_historical_mean': short - config['historical_short_mean'],
              'within_one_percentage_point_of_historical_mean': abs(short - config['historical_short_mean']) <= config['historical_short_mean_tolerance'],
              'source_manifest_sha256': file_sha256(root / 'IMPORT_COMPLETE.json'),
              'config_sha256': file_sha256(config['_path']), 'formal_claim_allowed': False,
              'claim_boundary': config['claim_boundary'],
              'comparison_scope': 'Same current runtime and paired test problems. Single training seed; CIs do not estimate training-seed variability.'}
    destination = root / 'analysis'
    write_json_exclusive(destination / 'report.json', report)
    lines = ['# C31 本地环境与 short 数据涨点验证', '',
             '当前三个条件使用同一固定模型权重、原评估代码和 GSM8K test[50:1319]。', '',
             '| 条件 | 正确题数 / 1269 | 正确率 | 相对当前 base | 平均输出 token |',
             '| --- | ---: | ---: | ---: | ---: |']
    for arm, m in metrics.items():
        delta = m['accuracy'] - metrics['base']['accuracy']
        lines.append(f"| {arm} | {m['correct']} | {m['accuracy']:.2%} | {delta * 100:+.2f} 个百分点 | {m['mean_output_tokens']:.2f} |")
    lines += ['', '## 配对涨跌幅', '']
    for arm, effect in paired.items():
        lines.append(f"- {arm}: {100*effect['estimate']:+.2f} 个百分点，配对 95% bootstrap CI [{100*effect['ci_low']:+.2f}, {100*effect['ci_high']:+.2f}]；纠正 base 错题 {effect['corrected_base_errors']} 道、损失 base 对题 {effect['lost_base_correct']} 道。")
    lines += ['', '## 数据与结论边界', '',
              '- 原本位于 /mnt/local 的 adapter 仅训练 4 条样本、1 步，属于环境测试；这次对它新增完整评估。',
              '- 新 short adapter 使用从 C31 Arrow 缓存逐字节恢复的原 881 题数据，原 legacy TRL 配方、seed 17、1 epoch、221 steps；新数据与最终 adapter 均发布到 home 项目目录。',
              '- 70.03% 是历史三种子均值；本轮为单种子复查，不能估计训练种子变异，也不重新选择种子或调参以追平历史分数。',
              '- 准确率和平均输出长度来自逐题预测。原评估器未保存原始停止标记，不能将重编码长度直接解释为精确 EOS/截断率。',
              '- 本次 C31/C32 的 BeeGFS 均未挂载；数据恢复及执行不依赖 BeeGFS。', '']
    (destination / 'report.md').write_text('\n'.join(lines))
    legacy.marker_write(root / 'LOCAL_SHORT_GAIN_CHECK_COMPLETE', {
        'status': 'complete', 'report_sha256': file_sha256(destination / 'report.json'),
        'config_sha256': file_sha256(config['_path']), 'training_runs': 1, 'evaluation_count': 3,
        'formal_claim_allowed': 'false'})
    print(json.dumps({'metrics': metrics, 'paired': paired}, ensure_ascii=False, indent=2))
