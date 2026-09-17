"""Exploratory interventions on the independently completed NCSU SAE.

Uses existing measured norm hooks, item-paired sampling and paired statistics.
The local Phase-6 orchestration is not reusable because its dictionaries, feature
counts, paths and historical-data imports are specific to the old C31 experiment.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from . import ncsu_reproduction as parent
from .experiment_io import read_json
from .factorial import file_sha256, read_key_value_marker
from .records import read_jsonl, write_jsonl
from .sae_norm_intervention import NormMatchedController, generate_condition
from .utility_analysis import paired_question_bootstrap


def specs(cfg):
    i = cfg['intervention']
    common = dict(dictionary='layer17_k64', count=i['feature_count_per_side'], start=i['start'])
    result = [dict(common, name='no_steering', mode='short', rho=0.)]
    for mode in i['modes']:
        for rho in i['strengths']:
            for random in (False, True):
                name = f'{"random_" if random else ""}{mode}_rho{rho:g}'
                result.append(dict(common, name=name, mode=mode, rho=rho, random_control=random))
    return result


class MatchedController(NormMatchedController):
    """Random controls match the target's number and signed grouping of columns."""
    def begin(self, spec, batch_size):
        super().begin(spec, batch_size)
        if spec.get('random_control'):
            count = spec['count']
            def unit(ids):
                v = self.decoder[ids].float().sum(0)
                return v / v.norm().clamp_min(1e-12)
            direction = unit(self.random_ids_list[:count])
            if spec['mode'] == 'joint':
                direction = direction - unit(self.random_ids_list[count:2*count])
            self.direction = direction / direction.norm().clamp_min(1e-12)


def prepare(config_path):
    cfg = read_json(config_path)
    root = parent.resolve(cfg['result_root'])
    old = parent.resolve(cfg['parent_root'])
    parent.verify(old/'EXPERIMENT_COMPLETE.json')
    frozen = read_json(old/'protocol/frozen_protocol.json')
    cfg.update(result_root=str(root), parent_root=str(old), checkpoint_root=str(parent.resolve(cfg['checkpoint_root'])))
    for key in ('teacher', 'student'):
        cfg[key]['snapshot_path'] = frozen[key]['snapshot_path']
    cfg['code_root'] = str(root/'code')
    if root.exists():
        raise FileExistsError(root)
    table = old/'sae_features/analysis/primary_heldout_features.csv'
    with table.open() as handle:
        features = [r for r in csv.DictReader(handle) if r['confirmed'] == 'True']
    count = cfg['intervention']['feature_count_per_side']
    selected = {direction: [int(r['feature_id']) for r in sorted(features, key=lambda r:int(r['discovery_rank']))
                           if r['direction'] == direction][:count] for direction in ('short','long')}
    if any(len(ids) != count for ids in selected.values()):
        raise ValueError('Insufficient confirmed parent features')
    sae_run = old/'sae/sae_training/layer_17_k_064'
    checkpoint = Path(read_json(sae_run/'training_metrics.json')['model_path'])
    if file_sha256(checkpoint) != read_key_value_marker(sae_run/'SAE_TRAINING_COMPLETE')['model_sha256']:
        raise ValueError('Parent SAE weight hash mismatch')
    selected['random'] = np.random.default_rng(cfg['intervention']['random_seed']).choice(
        sorted(set(range(28672))-set(selected['short']+selected['long'])), 2*count, replace=False).tolist()
    cfg['sae_checkpoint'] = str(checkpoint)
    cfg['feature_ids'] = selected
    questions = list(read_jsonl(old/'inputs/questions.jsonl'))
    assignments = read_json(old/'sae/corpus/question_split.json')['assignments']
    for r in questions:
        r.update(question_split=assignments[r['problem_id']], gold_answer=r['raw_answer'])
    cfg['cohorts'] = {}
    for split in ('dev','test'):
        rows = sorted((r for r in questions if r['question_split']==split),
                      key=lambda r:hashlib.sha256(f"2026091303:{r['problem_id']}".encode()).hexdigest())
        cfg['cohorts'][split] = [r['problem_id'] for r in rows[:cfg['intervention'][split+'_questions']]]
        if len(cfg['cohorts'][split]) != cfg['intervention'][split+'_questions']:
            raise ValueError('Incomplete cohort')
    write_jsonl(root/'inputs/questions.jsonl', questions)
    for filename in ('evaluation.jsonl','model_hashes.json','smoke_evaluation.json'):
        shutil.copy2(old/'inputs'/filename, root/'inputs'/filename)
    for directory in ('src','scripts','configs'):
        shutil.copytree(parent.CODE/directory, root/'code'/directory,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'), symlinks=True)
    sources = [p for directory in ('src','scripts','configs') for p in (root/'code'/directory).rglob('*') if p.is_file()]
    parent.save(root/'protocol/sources.json', parent.evidence(sources))
    parent.save(root/'protocol/frozen_protocol.json', cfg)
    parent.seal(root/'protocol/FROZEN.json', [root/'protocol/sources.json', root/'protocol/frozen_protocol.json',
        root/'inputs/questions.jsonl', root/'inputs/evaluation.jsonl',root/'inputs/model_hashes.json',
        root/'inputs/smoke_evaluation.json', checkpoint, table, old/'EXPERIMENT_COMPLETE.json',
        old/'sae/corpus/question_split.json', old/'sft_data/short.jsonl'], formal_claim_allowed=False)
    logging.info('Frozen %s: %s', root, selected)


def controller(cfg, model):
    import torch
    ids = cfg['feature_ids']
    return MatchedController(short_ids=ids['short'],long_ids=ids['long'],random_ids=ids['random'],
        torch_module=torch,checkpoint_path=cfg['sae_checkpoint'],layer_module=model.model.layers[cfg['intervention']['layer']],
        k=cfg['intervention']['k'],maximum_delta_fraction=max(cfg['intervention']['strengths']),
        device=model.device,dtype=torch.bfloat16)


def settings(cfg):
    return dict(seed=cfg['intervention']['generation_seed'],max_new_tokens=cfg['teacher']['max_new_tokens'],
                temperature=cfg['teacher']['temperature'],top_p=cfg['teacher']['top_p'])


def smoke(cfg):
    root = Path(cfg['result_root'])
    # Smoke examples come only from the parent's official test[:50] fixture.
    rows = read_json(root/'inputs/smoke_evaluation.json')['questions']
    for row in rows:
        row.update(question_split='smoke',gold_answer=row['raw_answer'])
    model, tok = parent.teacher_bundle(cfg)
    ctrl = controller(cfg,model)
    options = specs(cfg)
    runs = []
    baseline = generate_condition(model,tok,ctrl,options[0],rows,settings(cfg))
    duplicate = generate_condition(model,tok,ctrl,options[0],rows,settings(cfg))
    if [r['token_ids'] for r in baseline] != [r['token_ids'] for r in duplicate]:
        raise ValueError('Paired baseline replay is not deterministic')
    runs.extend(baseline)
    for spec in (options[1], options[2], options[7], options[8], dict(options[7],name='late_joint_smoke',start=16)):
        generated = generate_condition(model,tok,ctrl,spec,rows,settings(cfg))
        for base,row in zip(baseline,generated):
            diag = row['diagnostics']
            if diag['modified_positions'] == 0 or diag['mean_delta_to_hidden_norm_fraction'] < spec['rho']*.9:
                raise ValueError('Intervention did not reach the residual stream')
            if diag['max_delta_to_hidden_norm_fraction'] > spec['rho']+.005:
                raise ValueError('Measured intervention exceeds norm tolerance')
            if spec['start'] and base['token_ids'][:16] != row['token_ids'][:16]:
                raise ValueError('Delayed hook changed pre-intervention tokens')
        runs.extend(generated)
    write_jsonl(root/'smoke/predictions.jsonl',runs)
    parent.seal(root/'smoke/COMPLETE.json',[root/'protocol/FROZEN.json',root/'smoke/predictions.jsonl'],
                checks=['baseline replay','per-row actual norm','live positions','first-logit hook','delayed prefix equality'])


def condition_specs(cfg, split):
    if split == 'dev':
        return specs(cfg)
    root = Path(cfg['result_root'])
    parent.verify(root/'analysis/dev/COMPLETE.json')
    selected = read_json(root/'analysis/dev/summary.json')['selected']
    if selected is None:
        return []
    all_specs = specs(cfg)
    return [all_specs[0]] + [s for s in all_specs if s['name'] in (selected,'random_'+selected)]


def generate(cfg, split, index):
    root = Path(cfg['result_root'])
    parent.verify(root/'smoke/COMPLETE.json')
    conditions = condition_specs(cfg,split)
    if not conditions:
        logging.info('No dev condition passed: confirmation not triggered')
        return
    directory = root/'generation'/split/f'shard_{index}'
    if directory.exists():
        raise FileExistsError('Existing generation requires explicit recovery')
    cohorts = cfg['cohorts'][split]
    by_id = {r['problem_id']:r for r in read_jsonl(root/'inputs/questions.jsonl')}
    rows = [by_id[p] for j,p in enumerate(cohorts) if j%cfg['intervention']['shards']==index]
    model,tok = parent.teacher_bundle(cfg)
    ctrl = controller(cfg,model)
    directory.mkdir(parents=True)
    records = 0
    with (directory/'predictions.jsonl').open('x') as handle:
        for start in range(0,len(rows),cfg['intervention']['batch_size']):
            batch = rows[start:start+cfg['intervention']['batch_size']]
            for spec in conditions:
                for r in generate_condition(model,tok,ctrl,spec,batch,settings(cfg)):
                    handle.write(json.dumps(r,ensure_ascii=False)+'\n')
                    records += 1
                handle.flush()
                logging.info('split=%s shard=%s records=%s condition=%s',split,index,records,spec['name'])
    parent.seal(directory/'COMPLETE.json',[directory/'predictions.jsonl',root/'protocol/FROZEN.json'],
                records=records, problem_ids=[r['problem_id'] for r in rows])


def audit_rows(cfg, split):
    from .verifiers import extract_final_answer, verify_answer
    root = Path(cfg['result_root'])
    rows,markers = [],[]
    for i in range(cfg['intervention']['shards']):
        marker = root/'generation'/split/f'shard_{i}'/'COMPLETE.json'
        doc = parent.verify(marker)
        part = list(read_jsonl(marker.parent/'predictions.jsonl'))
        if len(part)!=doc['records']:
            raise ValueError('Shard count mismatch')
        rows.extend(part);markers.append(marker)
    expected = {(p,s['name']) for p in cfg['cohorts'][split] for s in condition_specs(cfg,split)}
    observed = [(r['problem_id'],r['condition']) for r in rows]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError('Missing/duplicate/unexpected generation keys')
    questions = {r['problem_id']:r for r in read_jsonl(root/'inputs/questions.jsonl')}
    seeds = {}
    for r in rows:
        gold = questions[r['problem_id']]['answer']
        if r['gold_answer']!=gold or r['is_correct']!=verify_answer(extract_final_answer(r['response']),gold):
            raise ValueError('Answer audit mismatch')
        if r['problem_id'] in seeds and seeds[r['problem_id']]!=r['seed']:
            raise ValueError('Unpaired sampling')
        seeds[r['problem_id']]=r['seed']
        if not np.isfinite(r['diagnostics']['max_delta_to_hidden_norm_fraction']):
            raise ValueError('Nonfinite intervention')
        if r['diagnostics']['max_delta_to_hidden_norm_fraction'] > r['spec']['rho']+.005:
            raise ValueError('Norm limit exceeded')
    return rows,markers


def analyze(cfg,split):
    root = Path(cfg['result_root'])
    conditions = condition_specs(cfg,split)
    if not conditions:
        logging.info('No confirmation required; dev termination marker already published')
        return
    rows,markers = audit_rows(cfg,split)
    grouped = {s['name']:{r['problem_id']:r for r in rows if r['condition']==s['name']} for s in conditions}
    metrics = {name:dict(accuracy=float(np.mean([r['is_correct'] for r in data.values()])),
        mean_tokens=float(np.mean([r['output_token_count'] for r in data.values()])),
        cap_rate=float(np.mean([r['hit_max_new_tokens'] for r in data.values()]))) for name,data in grouped.items()}
    def effect(left,right,field):
        return paired_question_bootstrap({p:float(r[field]) for p,r in grouped[left].items()},
            {p:float(r[field]) for p,r in grouped[right].items()},samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
    comparisons = {}
    for s in conditions:
        name=s['name']
        if name=='no_steering' or s.get('random_control'):
            continue
        length=effect(name,'no_steering','output_token_count')
        accuracy=effect(name,'no_steering','is_correct')
        random_length=effect(name,'random_'+name,'output_token_count')
        reduction=1-metrics[name]['mean_tokens']/metrics['no_steering']['mean_tokens']
        gate = dict(length_reduction=reduction>=cfg['intervention']['minimum_relative_length_reduction'],
            accuracy_drop=accuracy['estimate']>=-cfg['intervention']['maximum_accuracy_drop'],
            paired_length_ci=length['ci_high']<0,random_comparison=random_length['estimate']<0)
        comparisons[name]=dict(length=length,accuracy=accuracy,random_length=random_length,
                               relative_length_reduction=reduction,gate=gate,passed=all(gate.values()))
    eligible=[name for name,v in comparisons.items() if v['passed']]
    selected=min(eligible,key=lambda name:(metrics[name]['mean_tokens'],-metrics[name]['accuracy'],name)) if eligible else None
    summary=dict(status='complete',split=split,records=len(rows),unique_questions=len(cfg['cohorts'][split]),
        metrics=metrics,comparisons=comparisons,selected=selected,student_gate_passed=split=='test' and selected is not None,
        claim_boundary=cfg['claim_boundary'],formal_claim_allowed=False)
    directory=root/'analysis'/split
    parent.save(directory/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    names=list(metrics)
    colors=['#6c757d' if n=='no_steering' else '#e9c46a' if n.startswith('random_') else '#2a9d8f' for n in names]
    for ax,field,label in zip(axes,['mean_tokens','accuracy'],['Mean output tokens','Answer accuracy (%)']):
        ax.barh(names,[metrics[n][field]*(100 if field=='accuracy' else 1) for n in names],color=colors)
        ax.set_xlabel(label);ax.invert_yaxis()
    fig.savefig(directory/'teacher_comparison.png',dpi=180);fig.savefig(directory/'teacher_comparison.pdf');plt.close(fig)
    report=['# NCSU SAE 干预状态','',f'阶段：{split}；{len(rows)} 条输出，{len(cfg["cohorts"][split])} 道配对题，完整性审计通过。','',
        f'入选条件：`{selected}`。' if selected else '没有条件通过预登记的教师干预门槛；后续学生蒸馏不触发。','',
        '| 条件 | 正确率 | 平均输出 token |','| --- | ---: | ---: |']
    report.extend(f'| {n} | {m["accuracy"]*100:.2f}% | {m["mean_tokens"]:.2f} |' for n,m in metrics.items())
    report.extend(['','![教师干预比较](teacher_comparison.png)','','门槛：平均缩短至少 5%，正确率点估计下降不超过 5 个百分点，配对长度 95% 区间上界小于零，并且比同范数随机对照更短。正确率点估计门槛不构成统计非劣证明；dev 扫描区间未经多重比较校正，只用于探索性筛选。','',cfg['claim_boundary'],''])
    (directory/'report_zh.md').write_text('\n'.join(report))
    bindings=markers+[root/'protocol/FROZEN.json',directory/'summary.json',directory/'report_zh.md',directory/'teacher_comparison.png',directory/'teacher_comparison.pdf']
    parent.seal(directory/'COMPLETE.json',bindings,formal_claim_allowed=False)
    if selected is None:
        parent.seal(root/'INTERVENTION_COMPLETE.json',[directory/'COMPLETE.json'],outcome='stopped_at_teacher_gate',
                    stopped_stage=split,student_followup='not_triggered',formal_claim_allowed=False)


def submit(cfg,stage,split,index,deps=()):
    root=Path(cfg['result_root']);rt=cfg['runtime']
    gpu=stage in ('smoke','generate')
    (root/'logs').mkdir(exist_ok=True)
    cmd=['sbatch','--parsable',f"--account={rt['gpu_account'] if gpu else rt['cpu_account']}",
        f"--partition={rt['gpu_partition'] if gpu else 'compute_partners'}",f"--qos={rt['gpu_qos'] if gpu else 'short'}",
        '--nodes=1','--ntasks=1',f"--cpus-per-task={rt['cpus']}",f"--mem={rt['memory']}",
        f"--time={rt['gpu_time'] if gpu else '01:00:00'}",f'--job-name=nsi_{stage}_{split}_{index}',
        f'--output={root}/logs/{stage}_{split}_{index}_%j.log']
    if gpu:cmd.append(f"--gres=gpu:{rt['gpu_type']}:1")
    if deps:cmd.extend(['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes'])
    cmd.extend([str(Path(cfg['code_root'])/'scripts/slurm/3_40_ncsu_sae_intervention.sh'),cfg['code_root'],
        str(root/'protocol/frozen_protocol.json'),rt['python'],rt['overlay'],stage,'--split',split,'--index',str(index)])
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not job.isdigit():raise ValueError(job)
    with (root/'submission_ledger.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(job_id=job,stage=stage,split=split,index=index,deps=list(deps),command=cmd))+'\n')
        handle.flush();os.fsync(handle.fileno())
    logging.info('Submitted %s %s %s job=%s',stage,split,index,job)
    return job


def dispatch(args):
    if args.stage=='prepare':return prepare(args.config)
    cfg=read_json(args.config);root=Path(cfg['result_root'])
    parent.verify(root/'protocol/FROZEN.json')
    for path,digest in read_json(root/'protocol/sources.json').items():
        if file_sha256(path)!=digest:raise ValueError('Changed frozen source: '+path)
    logging.info('stage=%s split=%s shard=%s config=%s',args.stage,args.split,args.index,args.config)
    if args.stage=='submit':
        if (root/'submission_ledger.jsonl').exists():raise FileExistsError('Submission already recorded')
        return submit(cfg,'smoke','dev',0)
    if args.stage in ('smoke','generate'):
        if args.stage=='generate' and not condition_specs(cfg,args.split):return
        if not 0<=args.index<cfg['intervention']['shards']:raise ValueError('Invalid shard')
        parent.admission(cfg)
    if args.stage=='smoke':return smoke(cfg)
    if args.stage=='generate':return generate(cfg,args.split,args.index)
    return analyze(cfg,args.split)
