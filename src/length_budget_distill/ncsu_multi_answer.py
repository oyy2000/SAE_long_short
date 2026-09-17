"""Fixed-question answer multiplicity, paired supplementation and repeat controls.

The generator hook, SFT implementation, evaluator, bootstrap and plotting style
are reused from the completed NCSU experiment. Only candidate multiplicity and
registered repetition change. Frozen historical copies are never modified.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
from collections import Counter, defaultdict

import numpy as np
from . import ncsu_reproduction as common
from . import ncsu_intervention as intervention
from .experiment_io import read_json
from .factorial import file_sha256
from .records import read_jsonl, write_jsonl
from .sae_norm_intervention import generate_condition
from .verifiers import extract_final_answer, verify_answer


def unique_correct(rows):
    """Earliest-index text deduplication, followed by the registered length rank."""
    unique = {}
    for row in sorted(rows, key=lambda r: r['candidate_index']):
        if row['is_correct'] and not row['hit_max_new_tokens']:
            unique.setdefault(row['response'].strip(), row)
    return sorted(unique.values(), key=lambda r: (r['output_token_count'], r['candidate_index']))


def grouped_pool(rows, question_ids, conditions):
    result = {p: {c: [] for c in conditions} for p in question_ids}
    for row in rows:
        result[row['problem_id']][row['condition']].append(row)
    return result


def pool_shortfalls(pool, required):
    return {p: {c: len(unique_correct(rows)) for c, rows in arms.items()}
            for p, arms in pool.items() if any(len(unique_correct(rows)) < required for rows in arms.values())}


def check_candidate_records(rows, questions, cfg, *, initial_only=False):
    """Validate pairing, contiguous candidate IDs, live-token norms and golds."""
    conditions = cfg['conditions']
    maximum = cfg['supplement']['initial_candidates'] if initial_only else cfg['supplement']['max_candidates_per_question_condition']
    observed = set()
    by_question = defaultdict(lambda: defaultdict(set))
    streams = {}
    for row in rows:
        pid, arm, index = row['problem_id'], row['condition'], row['candidate_index']
        key = (pid, arm, index)
        if key in observed or pid not in questions or arm not in conditions or not 0 <= index < maximum:
            raise ValueError('Duplicate or unexpected candidate key')
        observed.add(key); by_question[pid][arm].add(index)
        gold = questions[pid]['answer']
        if row['gold_answer'] != gold or row['is_correct'] != verify_answer(extract_final_answer(row['response']), gold):
            raise ValueError('Candidate gold/verifier mismatch')
        stream_key = (pid, index)
        expected_seed = int.from_bytes(hashlib.sha256(
            f"{cfg['intervention']['generation_seed'] + cfg['supplement']['seed_offset'] + index}:{pid}".encode()).digest()[:4], 'little')
        if row['seed'] != expected_seed or (stream_key in streams and streams[stream_key] != row['seed']):
            raise ValueError('Unpaired or unexpected generation random stream')
        streams[stream_key] = row['seed']
        spec = next(s for s in cfg['main_specs'] if s['name'] == arm)
        if row['spec'] != spec:
            raise ValueError('Changed intervention specification')
        diag = row['diagnostics']; maximum_norm = diag['max_delta_to_hidden_norm_fraction']
        if not math.isfinite(maximum_norm) or not 0 <= maximum_norm <= spec['rho'] + .005:
            raise ValueError('Actual norm violates tolerance')
        if row['output_token_count'] != len(row['token_ids']):
            raise ValueError('Candidate token count mismatch')
        if spec['rho'] and diag['modified_positions'] != len(row['token_ids']):
            raise ValueError('Intervention live-position coverage mismatch')
    if set(by_question) != set(questions):
        raise ValueError('Missing question in candidate pool')
    for pid, arms in by_question.items():
        indices = arms[conditions[0]]
        if any(arms[c] != indices for c in conditions) or indices != set(range(len(indices))):
            raise ValueError('Missing or unpaired candidate indices')
        if len(indices) < cfg['supplement']['initial_candidates'] or len(indices) % cfg['supplement']['round_size']:
            raise ValueError('Incomplete candidate round')
        if initial_only and len(indices) != maximum:
            raise ValueError('Initial candidate count mismatch')


def contrast_definitions():
    definitions = {}
    for k in (1, 2, 4):
        definitions[f'sae_minus_no_steering_k{k}'] = {
            f'k{k}_unique__selected_target': 1, f'k{k}_unique__no_steering': -1}
    for k in (2, 4):
        definitions[f'method_gap_k{k}_minus_k1'] = {
            f'k{k}_unique__selected_target': 1, f'k{k}_unique__no_steering': -1,
            'k1_unique__selected_target': -1, 'k1_unique__no_steering': 1}
    for arm in ('no_steering', 'selected_target'):
        for k in (2, 4):
            definitions[f'{arm}_unique_minus_repeat_k{k}'] = {
                f'k{k}_unique__{arm}': 1, f'k{k}_repeat__{arm}': -1}
    return definitions


def prepare(config_path):
    overlay = read_json(config_path)
    parent_config_path = common.resolve(overlay['parent_config'])
    cfg = read_json(parent_config_path)
    parent_root = Path(cfg['result_root'])
    common.verify(parent_root/'MERGE_COMPLETE.json')
    common.verify(parent_root/'STUDENT_COMPLETE.json')
    parent_data = read_json(parent_root/'data_audit.json')
    support = sorted(parent_data['support'])
    root = common.resolve(overlay['result_root'])
    if root.exists():
        raise FileExistsError(root)
    runtime = {**cfg['runtime'], **overlay['runtime']}
    cfg.update(overlay)
    cfg.update(result_root=str(root), checkpoint_root=str(common.resolve(overlay['checkpoint_root'])),
               code_root=str(root/'code'), parent_student_root=str(parent_root), parent_config=str(parent_config_path), runtime=runtime)
    cfg['scope'] = 'Fixed 878-question, 1/2/4 unique-answer versus repeated-shortest SFT comparison'
    cfg['selection'] = {'method': 'shortest_k_unique_correct_uncapped', 'ties': 'candidate_index', 'pool': 'same supplemented pool at every k'}
    cfg['ranks'] = [f"{v['name']}__{c}" for v in cfg['variants'] for c in cfg['conditions']]
    cfg['contrasts'] = contrast_definitions()
    cfg['main_specs'] = [s for s in cfg['main_specs'] if s['name'] in cfg['conditions']]
    cfg['support'] = support
    cfg['teacher']['num_rollouts'] = cfg['supplement']['round_size']
    cfg['intervention']['student_conditions'] = cfg['conditions']
    cfg['intervention']['student_budgets'] = ['equal_examples']
    if len(support) != cfg['question_count'] or len(set(support)) != len(support):
        raise ValueError('Registered parent support mismatch')
    questions = {r['problem_id']: r for r in read_jsonl(parent_root/'inputs/questions.jsonl') if r['problem_id'] in support}
    rows = []
    bindings = [parent_root/'MERGE_COMPLETE.json', parent_root/'STUDENT_COMPLETE.json', parent_config_path, parent_root/'data_audit.json']
    for index in range(read_json(parent_config_path)['intervention']['shards']):
        directory = parent_root/'generation'/f'shard_{index}'
        common.verify(directory/'COMPLETE.json')
        bindings.extend([directory/'COMPLETE.json', directory/'predictions.jsonl'])
        rows.extend(r for r in read_jsonl(directory/'predictions.jsonl') if r['problem_id'] in questions and r['condition'] in cfg['conditions'])
    check_candidate_records(rows, questions, cfg, initial_only=True)
    pool = grouped_pool(rows, support, cfg['conditions'])
    shortfalls = pool_shortfalls(pool, cfg['supplement']['required_unique_correct'])
    cfg['supplement_questions'] = sorted(shortfalls)
    write_jsonl(root/'inputs/questions.jsonl', [questions[p] for p in support])
    write_jsonl(root/'inputs/initial_candidates.jsonl', rows)
    for name in ('evaluation.jsonl', 'model_hashes.json'):
        shutil.copy2(parent_root/'inputs'/name, root/'inputs'/name)
    registry = read_json(root/'inputs/model_hashes.json')
    verified = {}
    for role, files in registry.items():
        for path, expected in files.items():
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError('Base model hash mismatch: '+path)
            verified[path] = actual
    teacher_frozen = read_json(Path(cfg['teacher_root'])/'protocol/FROZEN.json')
    if file_sha256(cfg['sae_checkpoint']) != teacher_frozen['hashes'][cfg['sae_checkpoint']]:
        raise ValueError('SAE checkpoint changed')
    common.save(root/'setup/model_preflight.json', {'status': 'passed', 'verified_hashes': verified})
    common.save(root/'inputs/shortfalls.json', {'status': 'complete', 'questions': shortfalls, 'count': len(shortfalls)})
    for name in ('src', 'scripts', 'configs'):
        shutil.copytree(common.CODE/name, root/'code'/name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    files = [p for name in ('src', 'scripts', 'configs') for p in (root/'code'/name).rglob('*') if p.is_file() and not p.is_symlink()]
    common.save(root/'protocol/sources.json', common.evidence(files))
    common.save(root/'protocol/frozen_protocol.json', cfg)
    common.seal(root/'protocol/FROZEN.json', bindings + [root/'protocol/sources.json', root/'protocol/frozen_protocol.json',
        root/'inputs/questions.jsonl', root/'inputs/initial_candidates.jsonl', root/'inputs/evaluation.jsonl',
        root/'inputs/model_hashes.json', root/'inputs/shortfalls.json', root/'setup/model_preflight.json', Path(cfg['sae_checkpoint'])],
        source_config_sha256=file_sha256(config_path), formal_claim_allowed=False)
    logging.info('Frozen %s: %s questions, %s need supplementation, %s training runs', root, len(support), len(shortfalls), len(cfg['ranks'])*len(cfg['student_seeds']))


def supplement(cfg, index):
    root = Path(cfg['result_root']); spec = cfg['supplement']
    if not 0 <= index < spec['shards']:
        raise ValueError('Invalid shard')
    directory = root/'supplement'/f'shard_{index}'
    if (directory/'COMPLETE.json').exists():
        common.verify(directory/'COMPLETE.json'); return
    if directory.exists():
        raise FileExistsError('Partial supplementation requires explicit recovery')
    pids = [p for j, p in enumerate(cfg['supplement_questions']) if j % spec['shards'] == index]
    all_questions = {r['problem_id']: r for r in read_jsonl(root/'inputs/questions.jsonl')}
    initial = list(read_jsonl(root/'inputs/initial_candidates.jsonl'))
    pool = grouped_pool(initial, cfg['support'], cfg['conditions'])
    directory.mkdir(parents=True)
    generated = []; rounds = []
    if pids:
        model, tokenizer = common.teacher_bundle(cfg)
        controller = intervention.controller(cfg, model)
        active = pids
        with (directory/'predictions.jsonl').open('x') as handle:
            for begin in range(spec['initial_candidates'], spec['max_candidates_per_question_condition'], spec['round_size']):
                if not active: break
                round_pids = list(active)
                for start in range(0, len(active), spec['batch_size']):
                    batch = [all_questions[p] for p in active[start:start+spec['batch_size']]]
                    for candidate in range(begin, begin+spec['round_size']):
                        settings = intervention.settings(cfg)
                        settings['seed'] += spec['seed_offset'] + candidate
                        for arm in cfg['main_specs']:
                            rows = generate_condition(model, tokenizer, controller, arm, batch, settings)
                            for row in rows:
                                row['candidate_index'] = candidate
                                pool[row['problem_id']][row['condition']].append(row)
                                generated.append(row)
                                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                            handle.flush()
                active = [p for p in active if p in pool_shortfalls({p: pool[p]}, spec['required_unique_correct'])]
                rounds.append({'first_candidate': begin, 'problem_ids': round_pids, 'remaining': active[:]})
                logging.info('Supplement shard=%s candidate_end=%s records=%s remaining=%s', index, begin+spec['round_size'], len(generated), len(active))
    else:
        write_jsonl(directory/'predictions.jsonl', [])
    relevant = [r for r in initial if r['problem_id'] in pids] + generated
    if pids:
        check_candidate_records(relevant, {p: all_questions[p] for p in pids}, cfg)
    shortages = pool_shortfalls({p: pool[p] for p in pids}, spec['required_unique_correct'])
    common.save(directory/'manifest.json', {'status': 'complete', 'problem_ids': pids, 'records': len(generated),
        'rounds': rounds, 'shortfalls': shortages, 'candidates_complete': not shortages})
    common.seal(directory/'COMPLETE.json', [directory/'predictions.jsonl', directory/'manifest.json', root/'protocol/FROZEN.json'],
                records=len(generated), candidates_complete=not shortages, formal_claim_allowed=False)


def make_datasets(pool, questions, variants, conditions, tokenizer):
    """Build nested distinct-text datasets and explicit repetitions from one pool."""
    selected = {p: {c: unique_correct(pool[p][c])[:4] for c in conditions} for p in sorted(questions)}
    if any(len(rows) != 4 for arms in selected.values() for rows in arms.values()):
        raise ValueError('Four unique correct answers are required for every question and arm')
    datasets = {}
    for variant in variants:
        for arm in conditions:
            rows = []
            for pid in sorted(questions):
                chosen = selected[pid][arm][:variant['k']] if variant['mode'] == 'unique' else [selected[pid][arm][0]]*variant['k']
                for slot, source in enumerate(chosen):
                    tokens = len(tokenizer.encode(source['response'], add_special_tokens=False))
                    rows.append({'problem_id': pid, 'trace_id': f"{pid}:{arm}:{source['candidate_index']}:{variant['name']}:{slot}",
                        'source_candidate_index': source['candidate_index'], 'answer_slot': slot,
                        'occurrence_index': slot if variant['mode'] == 'repeat' else 0,
                        'prompt': questions[pid]['student_prompt'], 'completion': source['response'],
                        'completion_token_count': tokens, 'metadata': {'problem_id': pid, 'solution_token_count': tokens,
                        'is_correct': True, 'variant': variant['name'], 'condition': arm}})
            datasets[f"{variant['name']}__{arm}"] = rows
    return datasets


def load_complete_pool(cfg):
    root = Path(cfg['result_root'])
    rows = list(read_jsonl(root/'inputs/initial_candidates.jsonl')); markers = []
    for index in range(cfg['supplement']['shards']):
        directory = root/'supplement'/f'shard_{index}'
        marker = directory/'COMPLETE.json'; common.verify(marker); markers.append(marker)
        part = list(read_jsonl(directory/'predictions.jsonl'))
        manifest = read_json(directory/'manifest.json')
        if len(part) != manifest['records']:
            raise ValueError('Supplement manifest count mismatch')
        expected_pids = [p for j,p in enumerate(cfg['supplement_questions']) if j%cfg['supplement']['shards']==index]
        if manifest['problem_ids'] != expected_pids or any(r['problem_id'] not in expected_pids for r in part):
            raise ValueError('Supplement shard support mismatch')
        rows.extend(part)
    questions = {r['problem_id']:r for r in read_jsonl(root/'inputs/questions.jsonl')}
    check_candidate_records(rows, questions, cfg)
    pool = grouped_pool(rows, cfg['support'], cfg['conditions'])
    # Reconstruct the adaptive stopping rule independently from persisted rows.
    for pid, arms in pool.items():
        count = len(arms[cfg['conditions'][0]])
        for endpoint in range(cfg['supplement']['initial_candidates'], count, cfg['supplement']['round_size']):
            prefix = {pid: {c: [r for r in records if r['candidate_index'] < endpoint] for c,records in arms.items()}}
            if not pool_shortfalls(prefix, cfg['supplement']['required_unique_correct']):
                raise ValueError('Generated beyond the registered stopping point')
    return rows, questions, pool, markers


def build(cfg):
    from transformers import AutoTokenizer
    root = Path(cfg['result_root'])
    rows, questions, pool, markers = load_complete_pool(cfg)
    shortages = pool_shortfalls(pool, cfg['supplement']['required_unique_correct'])
    if shortages:
        common.save(root/'DATA_INCOMPLETE.json', {'status':'insufficient_candidates', 'shortfalls':shortages,
                    'training_allowed':False, 'formal_claim_allowed':False})
        raise RuntimeError('Candidate limit reached; registered cohort retained and training not triggered')
    tok = AutoTokenizer.from_pretrained(cfg['student']['snapshot_path'], local_files_only=True)
    datasets = make_datasets(pool, questions, cfg['variants'], cfg['conditions'], tok)
    budgets = []; bindings = markers + [root/'protocol/FROZEN.json']
    for rank, data in datasets.items():
        path = root/'sft_data'/f'{rank}.jsonl'; write_jsonl(path, data); bindings.append(path)
        variant_name, arm = rank.split('__')
        variant = next(v for v in cfg['variants'] if v['name']==variant_name)
        exposure = Counter(r['problem_id'] for r in data)
        if set(exposure) != set(cfg['support']) or set(exposure.values()) != {variant['k']}:
            raise ValueError('Question exposure mismatch')
        budgets.append({'rank':rank, 'variant':variant_name, 'condition':arm, 'k':variant['k'], 'mode':variant['mode'],
            'records':len(data), 'questions':len(exposure), 'completion_tokens':sum(r['completion_token_count'] for r in data),
            'steps':math.ceil(len(data)/cfg['training']['per_device_train_batch_size']),
            'unique_texts':sum(len({r['completion'].strip() for r in data if r['problem_id']==p}) for p in exposure)})
    old_k1 = {c:{r['problem_id']:r['completion'] for r in read_jsonl(Path(cfg['parent_student_root'])/'sft_data'/f'equal_examples__{c}.jsonl')} for c in cfg['conditions']}
    changed = {c:[r['problem_id'] for r in datasets[f'k1_unique__{c}'] if r['completion'] != old_k1[c][r['problem_id']]] for c in cfg['conditions']}
    common.save(root/'data_audit.json', {'status':'complete', 'question_count':len(questions), 'support':cfg['support'],
        'initial_candidates':len(list(read_jsonl(root/'inputs/initial_candidates.jsonl'))), 'pooled_candidates':len(rows),
        'supplemented_questions':len(cfg['supplement_questions']), 'changed_k1_questions':changed,
        'budgets':budgets, 'formal_claim_allowed':False})
    common.seal(root/'MERGE_COMPLETE.json', bindings+[root/'data_audit.json'], formal_claim_allowed=False)
    logging.info('Built %s datasets; original question support and multiplicities verified', len(datasets))


def analyze(cfg):
    from .ranked_multiseed_analysis import crossed_seed_problem_bootstrap
    from .sae_feature_analysis import holm_adjust
    root = Path(cfg['result_root']); common.verify(root/'MERGE_COMPLETE.json')
    expected = [r['problem_id'] for r in read_jsonl(root/'inputs/evaluation.jsonl')]
    names = ['base']+[f'{rank}__seed_{seed}' for rank in cfg['ranks'] for seed in cfg['student_seeds']]
    metrics = {}; predictions = {}; bindings = [root/'MERGE_COMPLETE.json']
    for name in names:
        directory = root/'student_evaluation'/name
        marker = directory/'COMPLETE.json'; common.verify(marker); bindings.append(marker)
        rows = list(read_jsonl(directory/'predictions.jsonl'))
        if [r['problem_id'] for r in rows] != expected:
            raise ValueError('Evaluation support/order mismatch')
        predictions[name] = {r['problem_id']:float(r['is_correct']) for r in rows}
        metrics[name] = read_json(directory/'metrics.json')
    contrasts = {}
    for name, terms in cfg['contrasts'].items():
        effects = {seed:{p:sum(coef*predictions[f'{rank}__seed_{seed}'][p] for rank,coef in terms.items()) for p in expected}
                   for seed in cfg['student_seeds']}
        contrasts[name] = crossed_seed_problem_bootstrap(effects, samples=cfg['bootstrap_samples'], seed=cfg['bootstrap_seed'])
        logging.info('Computed registered contrast %s', name)
    for value, adjusted in zip(contrasts.values(), holm_adjust([v['bootstrap_p_value'] for v in contrasts.values()])):
        value['holm_p_value'] = float(adjusted)
    gains = {str(k): all(contrasts[n]['ci_low']>0 and contrasts[n]['holm_p_value']<.05
                for n in (f'sae_minus_no_steering_k{k}', f'method_gap_k{k}_minus_k1')) for k in (2,4)}
    directory = root/'analysis'
    common.save(directory/'summary.json', {'status':'complete', 'metrics':metrics, 'contrasts':contrasts,
        'increased_sae_advantage_supported':gains, 'claim_boundary':cfg['claim_boundary'], 'formal_claim_allowed':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'figure.facecolor':'white'})
    colors = {'no_steering':'#6c757d', 'selected_target':'#2a9d8f'}
    labels = {'no_steering':'No steering', 'selected_target':'SAE'}
    def mean_metric(variant, arm, key):
        return float(np.mean([metrics[f'{variant}__{arm}__seed_{s}'][key] for s in cfg['student_seeds']]))
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm in cfg['conditions']:
        axes[0].plot([1,2,4], [100*mean_metric(f'k{k}_unique',arm,'accuracy') for k in (1,2,4)],
                     marker='o',color=colors[arm],label=labels[arm]+' distinct texts')
        axes[0].plot([1,2,4], [100*mean_metric('k1_unique' if k==1 else f'k{k}_repeat',arm,'accuracy') for k in (1,2,4)],
                     marker='s',linestyle='--',color=colors[arm],label=labels[arm]+' repeated shortest')
        axes[1].plot([1,2,4], [mean_metric(f'k{k}_unique',arm,'mean_output_tokens') for k in (1,2,4)],
                     marker='o',color=colors[arm],label=labels[arm])
    axes[0].axhline(100*metrics['base']['accuracy'],linestyle=':',color='black',label='Base')
    for ax in axes:ax.set_xticks([1,2,4]);ax.set_xlabel('Training records per question');ax.legend(fontsize=8)
    axes[0].set_ylabel('GSM8K accuracy (%)');axes[1].set_ylabel('Student mean output tokens')
    fig.savefig(directory/'answer_multiplicity.png',dpi=180);fig.savefig(directory/'answer_multiplicity.pdf');plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5),constrained_layout=True)
    labels_ci=list(contrasts)
    for i,n in enumerate(labels_ci):
        v=contrasts[n];x=100*v['estimate']
        ax.errorbar(x,i,xerr=[[x-100*v['ci_low']],[100*v['ci_high']-x]],fmt='o',color='#2a9d8f',capsize=3)
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(range(len(labels_ci)),labels_ci);ax.invert_yaxis()
    ax.set_xlabel('Accuracy difference (percentage points), unadjusted 95% CI')
    fig.savefig(directory/'paired_contrasts.png',dpi=180);fig.savefig(directory/'paired_contrasts.pdf');plt.close(fig)
    report=['# 固定878题的多答案训练结果','','30个 adapter 与31组完整评估已完成；每组评估1,269题。','',
            '| 数据方案 | 无干预准确率 | SAE准确率 | 无干预输出tokens | SAE输出tokens |', '| --- | ---: | ---: | ---: | ---: |']
    for v in cfg['variants']:
        n=v['name'];report.append(f"| {n} | {100*mean_metric(n,'no_steering','accuracy'):.2f}% | {100*mean_metric(n,'selected_target','accuracy'):.2f}% | {mean_metric(n,'no_steering','mean_output_tokens'):.2f} | {mean_metric(n,'selected_target','mean_output_tokens'):.2f} |")
    report.extend(['',f"本轮 Base：{100*metrics['base']['accuracy']:.2f}%。",'', '![答案数量曲线](answer_multiplicity.png)','',
        '| 登记比较 | 差值（百分点） | 95%区间 | Holm p |','| --- | ---: | ---: | ---: |'])
    for n,v in contrasts.items():
        report.append(f"| {n} | {100*v['estimate']:+.2f} | [{100*v['ci_low']:+.2f}, {100*v['ci_high']:+.2f}] | {v['holm_p_value']:.4f} |")
    report.extend(['','![配对比较](paired_contrasts.png)','',
        f'每题2条／4条扩大SAE优势的登记联合判定：{gains}。须方法优势与相对单答案的差距扩大同时通过检验。',
        '', '九项比较共同作Holm校正；区间为未校正的训练seed×配对题目bootstrap。不同文本不等于不同数学解法；重复对照匹配记录数、题目曝光和优化步数，不匹配监督token。', '',cfg['claim_boundary'],''])
    (directory/'report_zh.md').write_text('\n'.join(report))
    common.seal(root/'ANALYSIS_COMPLETE.json', bindings+[directory/'summary.json',directory/'report_zh.md',
        directory/'answer_multiplicity.png',directory/'answer_multiplicity.pdf',directory/'paired_contrasts.png',directory/'paired_contrasts.pdf'],formal_claim_allowed=False)


def audit(cfg):
    from transformers import AutoTokenizer
    from .student_evaluation import summarize_predictions
    root=Path(cfg['result_root'])
    checked={};markers=set();comparison_count=0
    def hashes(mapping):
        nonlocal comparison_count
        for path,expected in mapping.items():
            if path not in checked:checked[path]=file_sha256(path)
            if checked[path]!=expected:raise ValueError('Final audit hash mismatch: '+path)
            comparison_count+=1
    def marker(path):
        path=Path(path)
        if str(path) in markers:return
        doc=read_json(path)
        if doc.get('status')!='complete':raise ValueError('Incomplete marker')
        markers.add(str(path));hashes(doc['hashes'])
        for child in doc['hashes']:
            if Path(child).suffix=='.json':
                obj=read_json(child)
                if isinstance(obj,dict) and 'status' in obj and 'hashes' in obj:marker(child)
    marker(root/'ANALYSIS_COMPLETE.json');marker(root/'protocol/FROZEN.json')
    hashes(read_json(root/'protocol/sources.json'))
    for mapping in read_json(root/'inputs/model_hashes.json').values():hashes(mapping)
    logging.info('Hash audit complete: %s distinct files, %s markers',len(checked),len(markers))
    all_rows,questions,pool,_=load_complete_pool(cfg)
    if pool_shortfalls(pool,cfg['supplement']['required_unique_correct']):raise ValueError('Incomplete four-answer support')
    tok=AutoTokenizer.from_pretrained(cfg['student']['snapshot_path'],local_files_only=True)
    reconstructed=make_datasets(pool,questions,cfg['variants'],cfg['conditions'],tok)
    for rank,expected in reconstructed.items():
        actual=list(read_jsonl(root/'sft_data'/f'{rank}.jsonl'))
        if actual!=expected:raise ValueError('Training data mismatch: '+rank)
        for row in actual:
            if not verify_answer(extract_final_answer(row['completion']),questions[row['problem_id']]['answer']):
                raise ValueError('Invalid selected training answer')
    fixtures=list(read_jsonl(root/'inputs/evaluation.jsonl'))
    if [r['source_index'] for r in fixtures]!=list(range(50,1319)):raise ValueError('Changed locked split')
    expected_ids=[r['problem_id'] for r in fixtures];golds={r['problem_id']:r['answer'] for r in fixtures}
    summary=read_json(root/'analysis/summary.json');names=['base']+[f'{r}__seed_{s}' for r in cfg['ranks'] for s in cfg['student_seeds']]
    count=0;adapters=[]
    for name in names:
        directory=root/'student_evaluation'/name
        rows=list(read_jsonl(directory/'predictions.jsonl'))
        if [r['problem_id'] for r in rows]!=expected_ids:raise ValueError('Missing/duplicate evaluation rows')
        for row in rows:
            gold=golds[row['problem_id']]
            if row['gold_answer']!=gold or row['is_correct']!=verify_answer(extract_final_answer(row['prediction_text']),gold):
                raise ValueError('Evaluation verifier mismatch')
        actual=summarize_predictions(rows)
        for k,v in actual.items():
            if abs(v-summary['metrics'][name][k])>1e-10:raise ValueError('Metric mismatch')
        count+=len(rows)
        if name=='base':continue
        rank,seed=name.split('__seed_')
        adapter=Path(cfg['checkpoint_root'])/'students'/name
        m=read_json(adapter/'training_metrics.json')
        if m['records']!=len(reconstructed[rank]) or m['steps']!=math.ceil(m['records']/cfg['training']['per_device_train_batch_size']) or m['seed']!=int(seed):
            raise ValueError('Training steps, records or seed mismatch')
        if not any(r.get('learning_rate',0)>0 for r in m['log_history']):raise ValueError('No positive learning rate')
        if not all(math.isfinite(v) for r in m['log_history'] for v in r.values() if isinstance(v,(float,int))):raise ValueError('Nonfinite training')
        adapters.append({'name':name,'records':m['records'],'steps':m['steps']})
    common.save(root/'FINAL_AUDIT.json', {'status':'passed','hash_comparisons':comparison_count,'unique_verified_files':len(checked),
        'verified_markers':len(markers),'verified_hashes':checked,'questions':len(questions),'pooled_candidates':len(all_rows),
        'training_data_variants':len(reconstructed),'adapters':adapters,'adapter_count':len(adapters),
        'evaluation_runs':len(names),'evaluation_records':count,'formal_claim_allowed':False})
    common.seal(root/'EXPERIMENT_COMPLETE.json',[root/'ANALYSIS_COMPLETE.json',root/'FINAL_AUDIT.json'],
                outcome='multi_answer_and_repeat_controls_complete',formal_claim_allowed=False)
    logging.info('Experiment complete: %s adapters, %s evaluations, %s prediction records',len(adapters),len(names),count)


def queue(cfg,stage,*,index=0,name='base',deps=()):
    root=Path(cfg['result_root']);runtime=cfg['runtime'];gpu=stage in ('supplement','student')
    (root/'logs').mkdir(exist_ok=True)
    cmd=['sbatch','--parsable',f"--account={runtime['gpu_account'] if gpu else runtime['cpu_account']}",
        f"--partition={runtime['gpu_partition'] if gpu else 'compute_partners'}",f"--qos={runtime['gpu_qos'] if gpu else 'short'}",
        '--nodes=1','--ntasks=1',f"--cpus-per-task={runtime['cpus']}",f"--mem={runtime['memory']}",
        f"--time={runtime['gpu_time'] if gpu else runtime['cpu_time']}",f'--job-name=nma_{stage}_{name if stage=="student" else index}',
        f'--output={root}/logs/{stage}_{name if stage=="student" else index}_%j.log']
    if gpu:cmd.append(f"--gres=gpu:{runtime['gpu_type']}:1")
    if deps:cmd.extend(['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes'])
    cmd.extend([str(Path(cfg['code_root'])/'scripts/slurm/4_42_ncsu_multi_answer.sh'),cfg['code_root'],
        str(root/'protocol/frozen_protocol.json'),runtime['python'],runtime['overlay'],stage,'--index',str(index),'--name',name])
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not job.isdigit():raise ValueError(job)
    with (root/'submission_ledger.jsonl').open('a') as handle:
        handle.write(json.dumps({'job_id':job,'stage':stage,'index':index,'name':name,'dependencies':list(deps),'command':cmd})+'\n')
        handle.flush();os.fsync(handle.fileno())
    logging.info('Submitted %s %s job=%s',stage,name if stage=='student' else index,job)
    return job


def dispatch(args):
    if args.stage=='prepare':return prepare(args.config)
    cfg=read_json(args.config);root=Path(cfg['result_root'])
    common.verify(root/'protocol/FROZEN.json')
    for path,digest in read_json(root/'protocol/sources.json').items():
        if file_sha256(path)!=digest:raise ValueError('Frozen source changed: '+path)
    logging.info('stage=%s index=%s name=%s experiment=%s',args.stage,args.index,args.name,cfg['experiment_name'])
    if args.stage=='submit':
        if (root/'submission_ledger.jsonl').exists():raise FileExistsError('Submission ledger already exists; inspect recorded DAG')
        supplements=[queue(cfg,'supplement',index=i) for i in range(cfg['supplement']['shards'])]
        merge=queue(cfg,'build',deps=supplements)
        base=queue(cfg,'student',name='base',deps=[merge])
        students=[queue(cfg,'student',name=f'{rank}__seed_{seed}',deps=[merge]) for rank in cfg['ranks'] for seed in cfg['student_seeds']]
        analysis=queue(cfg,'analyze',deps=[base,*students])
        final=queue(cfg,'audit',deps=[analysis])
        return common.save(root/'submission.json',{'status':'submitted_not_completed','supplements':supplements,'build':merge,
            'base':base,'students':students,'analysis':analysis,'final':final})
    if args.stage in ('supplement','student'):common.admission(cfg)
    if args.stage=='supplement':return supplement(cfg,args.index)
    if args.stage=='student':return common.student(cfg,args.name)
    return globals()[args.stage](cfg)
