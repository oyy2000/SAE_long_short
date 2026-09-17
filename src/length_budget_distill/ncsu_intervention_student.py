"""Conditional full-cohort generation and controlled SFT after teacher gates.

Reuses the original NCSU student recipe/evaluator and the established whole-trace
repeat helper. It never trains a natural control on a different question support.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
from collections import Counter, defaultdict

import numpy as np
from . import ncsu_reproduction as parent
from . import ncsu_intervention as intervention
from .experiment_io import read_json
from .factorial import file_sha256
from .records import read_jsonl, write_jsonl
from .sae_norm_intervention import generate_condition
from .verifiers import extract_final_answer, verify_answer


def prepare(config_path):
    cfg=read_json(config_path)
    teacher_root=Path(cfg['result_root'])
    for split in ('dev','test'):
        parent.verify(teacher_root/'analysis'/split/'COMPLETE.json')
    summary=read_json(teacher_root/'analysis/test/summary.json')
    if not summary['student_gate_passed']:raise ValueError('Teacher test gate did not pass')
    root=teacher_root/'student_followup'
    if root.exists():raise FileExistsError(root)
    cfg.update(teacher_root=str(teacher_root),result_root=str(root),code_root=str(root/'code'))
    cfg['checkpoint_root']=str(Path(cfg['checkpoint_root'])/'student_followup')
    cfg['runtime'].update(read_json(teacher_root/'setup/scheduling_h200.json')['runtime_overlay'])
    cfg['main_specs']=[dict(s,name={'no_steering':'no_steering',summary['selected']:'selected_target',
                      'random_'+summary['selected']:'matched_random'}[s['name']])
                       for s in intervention.condition_specs(read_json(config_path),'test')]
    cfg['ranks']=[f'{budget}__{condition}' for budget in cfg['intervention']['student_budgets']
                  for condition in cfg['intervention']['student_conditions']]
    for directory in ('src','scripts','configs'):
        shutil.copytree(parent.CODE/directory,root/'code'/directory,symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    (root/'inputs').mkdir()
    for name in ('questions.jsonl','evaluation.jsonl','model_hashes.json'):
        shutil.copy2(teacher_root/'inputs'/name,root/'inputs'/name)
    files=[p for d in ('src','scripts','configs') for p in (root/'code'/d).rglob('*') if p.is_file() and not p.is_symlink()]
    parent.save(root/'protocol/sources.json',parent.evidence(files))
    parent.save(root/'protocol/frozen_protocol.json',cfg)
    parent.seal(root/'protocol/FROZEN.json',[root/'protocol/sources.json',root/'protocol/frozen_protocol.json',
        root/'inputs/questions.jsonl',root/'inputs/evaluation.jsonl',root/'inputs/model_hashes.json',
        teacher_root/'analysis/test/COMPLETE.json',teacher_root/'analysis/dev/COMPLETE.json',
        Path(cfg['parent_root'])/'sft_data/short.jsonl',Path(cfg['sae_checkpoint'])],formal_claim_allowed=False)


def generate(cfg,index):
    root=Path(cfg['result_root'])
    directory=root/'generation'/f'shard_{index}'
    if directory.exists():raise FileExistsError(directory)
    all_rows=list(read_jsonl(root/'inputs/questions.jsonl'))
    rows=[r for j,r in enumerate(all_rows) if j%cfg['intervention']['shards']==index]
    model,tok=parent.teacher_bundle(cfg)
    ctrl=intervention.controller(cfg,model)
    directory.mkdir(parents=True)
    n=0
    with (directory/'predictions.jsonl').open('x') as handle:
        for start in range(0,len(rows),cfg['intervention']['batch_size']):
            batch=rows[start:start+cfg['intervention']['batch_size']]
            for candidate in range(cfg['intervention']['main_candidates']):
                settings=intervention.settings(cfg)
                # Independent new draws; same candidate-specific stream in all arms.
                settings['seed']+=10000+candidate
                for spec in cfg['main_specs']:
                    for row in generate_condition(model,tok,ctrl,spec,batch,settings):
                        row['candidate_index']=candidate
                        handle.write(json.dumps(row,ensure_ascii=False)+'\n');n+=1
                    handle.flush()
            logging.info('main shard=%s records=%s questions=%s/%s',index,n,min(start+len(batch),len(rows)),len(rows))
    parent.seal(directory/'COMPLETE.json',[directory/'predictions.jsonl',root/'protocol/FROZEN.json'],
                records=n,problem_ids=[r['problem_id'] for r in rows])


def select_common(questions,predictions,natural,candidates):
    """Audit every key/verifier/stream before choosing common correct support."""
    conditions=['no_steering','selected_target','matched_random']
    expected={(p,c,i) for p in questions for c in conditions for i in range(candidates)}
    observed=[(r['problem_id'],r['condition'],r['candidate_index']) for r in predictions]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError('Missing, duplicate or unexpected main candidate keys')
    grouped={c:defaultdict(list) for c in conditions}
    seeds={}
    for row in predictions:
        pid=row['problem_id'];gold=questions[pid]['answer']
        if row['gold_answer']!=gold or row['is_correct']!=verify_answer(extract_final_answer(row['response']),gold):
            raise ValueError('Verifier or gold mismatch')
        key=(pid,row['candidate_index'])
        if key in seeds and seeds[key]!=row['seed']:raise ValueError('Candidate streams are not paired')
        seeds[key]=row['seed']
        if row['is_correct'] and not row['hit_max_new_tokens']:
            grouped[row['condition']][pid].append(row)
    support=sorted(set(natural).intersection(*(set(v) for v in grouped.values())))
    if not support:raise ValueError('Empty common support')
    selected={c:[] for c in conditions}
    for c in conditions:
        for pid in support:
            unique={}
            for row in sorted(grouped[c][pid],key=lambda r:r['candidate_index']):
                unique.setdefault(row['response'].strip(),row)
            selected[c].append(min(unique.values(),key=lambda r:(r['output_token_count'],r['candidate_index'])))
    return support,selected,{c:len(v) for c,v in grouped.items()}


def build(cfg):
    from transformers import AutoTokenizer
    root=Path(cfg['result_root'])
    predictions=[];markers=[]
    for index in range(cfg['intervention']['shards']):
        marker=root/'generation'/f'shard_{index}'/'COMPLETE.json'
        doc=parent.verify(marker)
        part=list(read_jsonl(marker.parent/'predictions.jsonl'))
        if len(part)!=doc['records']:raise ValueError('Shard manifest record count mismatch')
        predictions.extend(part);markers.append(marker)
    questions={r['problem_id']:r for r in read_jsonl(root/'inputs/questions.jsonl')}
    natural={r['metadata']['problem_id']:r for r in read_jsonl(Path(cfg['parent_root'])/'sft_data/short.jsonl')}
    support,selected,counts=select_common(questions,predictions,natural,cfg['intervention']['main_candidates'])
    tok=AutoTokenizer.from_pretrained(cfg['student']['snapshot_path'],local_files_only=True)
    datasets={name:[] for name in cfg['intervention']['student_conditions']}
    for name,rows in selected.items():
        for row in rows:
            pid=row['problem_id']
            datasets[name].append(dict(problem_id=pid,trace_id=f'{pid}:{name}:{row["candidate_index"]}',occurrence_index=0,
                prompt=questions[pid]['student_prompt'],completion=row['response'],source_candidate_index=row['candidate_index']))
    for pid in support:
        row=natural[pid]
        if row['prompt']!=questions[pid]['student_prompt'] or not verify_answer(extract_final_answer(row['completion']),questions[pid]['answer']):
            raise ValueError('Natural control prompt/gold mismatch')
        datasets['natural_short'].append(dict(problem_id=pid,trace_id=row['metadata']['trace_id'],occurrence_index=0,
            prompt=row['prompt'],completion=row['completion'],source_candidate_index=-1))
    for name,rows in datasets.items():
        if [r['problem_id'] for r in rows]!=support:raise ValueError('Different training supports')
        for row in rows:
            n=len(tok.encode(row['completion'],add_special_tokens=False))
            row['completion_token_count']=n
            row['metadata']={'problem_id':row['problem_id'],'solution_token_count':n,'is_correct':True}
    target=max(sum(r['completion_token_count'] for r in rows) for rows in datasets.values())
    helper=parent.CODE/'scripts/4_0_build_intervention_sft_datasets.py'
    spec=importlib.util.spec_from_file_location('registered_repeat_helper',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    regimes={'equal_examples':datasets,'equal_target_tokens':{name:module._repeat_to_target(rows,target,
             seed=cfg['intervention']['generation_seed'],condition=name) for name,rows in datasets.items()}}
    bindings=markers+[root/'protocol/FROZEN.json',helper]
    budgets=[]
    for regime,cells in regimes.items():
        for name,rows in cells.items():
            filename=root/'sft_data'/f'{regime}__{name}.jsonl'
            write_jsonl(filename,rows);bindings.append(filename)
            budgets.append(dict(regime=regime,condition=name,rows=len(rows),unique_questions=len(support),
                completion_tokens=sum(r['completion_token_count'] for r in rows),steps=math.ceil(len(rows)/4),
                question_weights=dict(Counter(r['problem_id'] for r in rows))))
    totals=[b['completion_tokens'] for b in budgets if b['regime']=='equal_target_tokens']
    if max(totals)-min(totals)>cfg['teacher']['max_new_tokens']:raise ValueError('Token budget mismatch exceeds one trace')
    teacher_metrics={name:dict(records=sum(r['condition']==name for r in predictions),
        accuracy=float(np.mean([r['is_correct'] for r in predictions if r['condition']==name])),
        mean_tokens=float(np.mean([r['output_token_count'] for r in predictions if r['condition']==name]))) for name in selected}
    parent.save(root/'data_audit.json',dict(status='complete',support=support,support_count=len(support),
        excluded_questions=sorted(set(questions)-set(support)),correct_support_by_condition=counts,
        records=len(predictions),budgets=budgets,teacher_metrics=teacher_metrics,
        full_generation_does_not_reselect_intervention=True,formal_claim_allowed=False))
    bindings.append(root/'data_audit.json')
    parent.seal(root/'MERGE_COMPLETE.json',bindings,formal_claim_allowed=False)


def analyze(cfg):
    from .ranked_multiseed_analysis import crossed_seed_problem_bootstrap
    from .sae_feature_analysis import holm_adjust
    root=Path(cfg['result_root'])
    parent.verify(root/'MERGE_COMPLETE.json')
    names=['base']+[f'{rank}__seed_{seed}' for rank in cfg['ranks'] for seed in cfg['student_seeds']]
    predictions={};metrics={};markers=[]
    expected={r['problem_id'] for r in read_jsonl(root/'inputs/evaluation.jsonl')}
    for name in names:
        directory=root/'student_evaluation'/name
        marker=directory/'COMPLETE.json';parent.verify(marker);markers.append(marker)
        rows=list(read_jsonl(directory/'predictions.jsonl'))
        if len(rows)!=len(expected) or {r['problem_id'] for r in rows}!=expected:raise ValueError('Evaluation support mismatch')
        predictions[name]={r['problem_id']:float(r['is_correct']) for r in rows}
        metrics[name]=read_json(directory/'metrics.json')
    contrasts={}
    for budget in cfg['intervention']['student_budgets']:
        for other in ('no_steering','matched_random','natural_short'):
            effects={seed:{pid:predictions[f'{budget}__selected_target__seed_{seed}'][pid]-
                        predictions[f'{budget}__{other}__seed_{seed}'][pid] for pid in sorted(expected)} for seed in cfg['student_seeds']}
            contrasts[f'{budget}__target_minus_{other}']=crossed_seed_problem_bootstrap(effects,
                samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
    for value,p in zip(contrasts.values(),holm_adjust([v['bootstrap_p_value'] for v in contrasts.values()])):
        value['holm_p_value']=float(p)
    directory=root/'analysis'
    parent.save(directory/'summary.json',dict(status='complete',metrics=metrics,contrasts=contrasts,
        data_audit=read_json(root/'data_audit.json'),formal_claim_allowed=False,claim_boundary=cfg['claim_boundary']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,budget in zip(axes,cfg['intervention']['student_budgets']):
        for j,name in enumerate(cfg['intervention']['student_conditions']):
            ys=[100*metrics[f'{budget}__{name}__seed_{seed}']['accuracy'] for seed in cfg['student_seeds']]
            ax.bar(j,np.mean(ys),color=['#6c757d','#2a9d8f','#e9c46a','#e76f51'][j]);ax.scatter([j]*len(ys),ys,color='black',s=16)
        ax.axhline(100*metrics['base']['accuracy'],color='gray',linestyle='--',label='Base')
        ax.set_xticks(range(4),['No steering','SAE target','Random','Natural short'],rotation=15)
        ax.set_ylabel('GSM8K accuracy (%)');ax.set_title(budget);ax.legend()
    fig.savefig(directory/'student_comparison.png',dpi=180);fig.savefig(directory/'student_comparison.pdf');plt.close(fig)
    report=['# NCSU SAE 干预学生结果','','全量候选生成、共同支持训练和完整评估已完成。','',
        f'共同支持：{read_json(root/"data_audit.json")["support_count"]}/881 题。三个训练 seed；每个模型评估 1,269 题。','',
        '| 预算 | 条件 | 平均准确率 |','| --- | --- | ---: |']
    for budget in cfg['intervention']['student_budgets']:
        for name in cfg['intervention']['student_conditions']:
            mean=np.mean([metrics[f'{budget}__{name}__seed_{seed}']['accuracy'] for seed in cfg['student_seeds']])
            report.append(f'| {budget} | {name} | {100*mean:.2f}% |')
    report.extend(['',f'同环境 Base：{100*metrics["base"]["accuracy"]:.2f}%。','','![学生比较](student_comparison.png)','',
        '| 配对比较 | 差值（百分点） | 95% 区间 | Holm p |','| --- | ---: | ---: | ---: |'])
    for name,v in contrasts.items():report.append(f'| {name} | {100*v["estimate"]:+.2f} | [{100*v["ci_low"]:+.2f}, {100*v["ci_high"]:+.2f}] | {v["holm_p_value"]:.4f} |')
    report.extend(['','置信区间为训练 seed 与题目的交叉 bootstrap；六项计划比较一起作 Holm 校正。等 token 采用完整轨迹重复，训练步数和题目权重仍不完全相同。','',cfg['claim_boundary'],''])
    (directory/'report_zh.md').write_text('\n'.join(report))
    parent.seal(root/'STUDENT_COMPLETE.json',markers+[root/'MERGE_COMPLETE.json',directory/'summary.json',directory/'report_zh.md',
        directory/'student_comparison.png',directory/'student_comparison.pdf'],formal_claim_allowed=False)
    parent.seal(Path(cfg['teacher_root'])/'INTERVENTION_COMPLETE.json',[root/'STUDENT_COMPLETE.json',
        Path(cfg['teacher_root'])/'analysis/test/COMPLETE.json',Path(cfg['teacher_root'])/'analysis/dev/COMPLETE.json'],
        outcome='teacher_and_student_followup_complete',formal_claim_allowed=False)


def queue(cfg,stage,index=0,name='base',deps=()):
    root=Path(cfg['result_root']);rt=cfg['runtime'];gpu=stage in ('generate','student')
    (root/'logs').mkdir(exist_ok=True)
    cmd=['sbatch','--parsable',f"--account={rt['gpu_account'] if gpu else rt['cpu_account']}",
        f"--partition={rt['gpu_partition'] if gpu else 'compute_partners'}",f"--qos={rt['gpu_qos'] if gpu else 'short'}",
        '--nodes=1','--ntasks=1',f"--cpus-per-task={rt['cpus']}",f"--mem={rt['memory']}",
        '--time=02:00:00',f'--job-name=nsi_sft_{stage}_{name if stage=="student" else index}',
        f'--output={root}/logs/{stage}_{name if stage=="student" else index}_%j.log']
    if gpu:cmd.append(f"--gres=gpu:{rt['gpu_type']}:1")
    if deps:cmd.extend(['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes'])
    cmd.extend([str(Path(cfg['code_root'])/'scripts/slurm/4_40_ncsu_intervention_student.sh'),cfg['code_root'],
        str(root/'protocol/frozen_protocol.json'),rt['python'],rt['overlay'],stage,'--index',str(index),'--name',name])
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not job.isdigit():raise ValueError(job)
    with (root/'submission_ledger.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(job_id=job,stage=stage,index=index,name=name,deps=list(deps),command=cmd))+'\n')
        handle.flush();os.fsync(handle.fileno())
    logging.info('Submitted stage=%s name=%s job=%s',stage,name,job)
    return job


def dispatch(args):
    if args.stage=='prepare':return prepare(args.config)
    cfg=read_json(args.config);root=Path(cfg['result_root'])
    parent.verify(root/'protocol/FROZEN.json')
    for path,digest in read_json(root/'protocol/sources.json').items():
        if file_sha256(path)!=digest:raise ValueError('Changed frozen source: '+path)
    logging.info('stage=%s index=%s name=%s',args.stage,args.index,args.name)
    if args.stage=='submit':
        if (root/'submission_ledger.jsonl').exists():raise FileExistsError('Already submitted')
        generation=[queue(cfg,'generate',i) for i in range(cfg['intervention']['shards'])]
        merge=queue(cfg,'build',deps=generation)
        base=queue(cfg,'student',name='base')
        students=[queue(cfg,'student',name=f'{rank}__seed_{seed}',deps=[merge]) for rank in cfg['ranks'] for seed in cfg['student_seeds']]
        final=queue(cfg,'analyze',deps=[base,*students])
        return parent.save(root/'submission.json',dict(status='submitted_not_completed',generation=generation,merge=merge,base=base,students=students,final=final))
    if args.stage in ('generate','student'):parent.admission(cfg)
    if args.stage=='generate':return generate(cfg,args.index)
    if args.stage=='student':return parent.student(cfg,args.name)
    return globals()[args.stage](cfg)


def audit_delivery(cfg, output):
    """Read back the completed evidence graph and audit all registered records.

    This post-run audit adds no generation, training, tuning or model selection.
    It preserves the frozen execution copies and their original completion marks.
    """
    from .student_evaluation import summarize_predictions
    from .utility_analysis import paired_question_bootstrap
    from transformers import AutoTokenizer
    root=Path(cfg['result_root']);teacher_root=Path(cfg['teacher_root'])
    cached={};visited=set();comparisons=0
    def digest(path):
        path=str(Path(path).resolve())
        if path not in cached:cached[path]=file_sha256(path)
        return cached[path]
    def check_hashes(mapping):
        nonlocal comparisons
        for path,expected in mapping.items():
            if digest(path)!=expected:raise ValueError('Hash mismatch: '+path)
            comparisons+=1
    def walk(path):
        path=Path(path).resolve()
        if str(path) in visited:return
        visited.add(str(path))
        doc=read_json(path)
        if doc.get('status')!='complete':raise ValueError('Incomplete marker: '+str(path))
        check_hashes(doc['hashes'])
        for child in doc['hashes']:
            if Path(child).suffix=='.json':
                obj=read_json(child)
                if isinstance(obj,dict) and 'hashes' in obj and 'status' in obj:
                    walk(child)
    walk(teacher_root/'INTERVENTION_COMPLETE.json')
    for protocol_root in (root,teacher_root):
        walk(protocol_root/'protocol/FROZEN.json')
        check_hashes(read_json(protocol_root/'protocol/sources.json'))
        for mapping in read_json(protocol_root/'inputs/model_hashes.json').values():check_hashes(mapping)
    logging.info('Evidence hashes verified: %s distinct files',len(cached))
    questions={r['problem_id']:r for r in read_jsonl(root/'inputs/questions.jsonl')}
    teacher_predictions=[]
    for i in range(cfg['intervention']['shards']):
        teacher_predictions.extend(read_jsonl(root/'generation'/f'shard_{i}'/'predictions.jsonl'))
    natural={r['metadata']['problem_id']:r for r in read_jsonl(Path(cfg['parent_root'])/'sft_data/short.jsonl')}
    support,selected,counts=select_common(questions,teacher_predictions,natural,cfg['intervention']['main_candidates'])
    data=read_json(root/'data_audit.json')
    if support!=data['support'] or counts!=data['correct_support_by_condition']:raise ValueError('Common support audit mismatch')
    diagnostics={}
    for name in ('no_steering','selected_target','matched_random'):
        rows=[r for r in teacher_predictions if r['condition']==name]
        rho=next(s['rho'] for s in cfg['main_specs'] if s['name']==name)
        fractions=[r['diagnostics']['max_delta_to_hidden_norm_fraction'] for r in rows]
        if not all(np.isfinite(v) and 0<=v<=rho+.005 for v in fractions):raise ValueError('Actual norm outside tolerance')
        if rho and any(r['diagnostics']['modified_positions']!=len(r['token_ids']) for r in rows):
            raise ValueError('Live-token modification coverage mismatch')
        for row in rows:
            if len(row['token_ids'])!=row['output_token_count']:raise ValueError('Token count mismatch')
        diagnostics[name]=dict(records=len(rows),maximum_actual_norm_fraction=max(fractions),
            mean_actual_norm_fraction=float(np.mean([r['diagnostics']['mean_delta_to_hidden_norm_fraction'] for r in rows])),
            cap_count=sum(r['hit_max_new_tokens'] for r in rows))
        m=data['teacher_metrics'][name]
        if m['records']!=len(rows) or not np.isclose(m['accuracy'],np.mean([r['is_correct'] for r in rows]),atol=1e-12):
            raise ValueError('Teacher aggregate mismatch')
    tok=AutoTokenizer.from_pretrained(cfg['student']['snapshot_path'],local_files_only=True)
    datasets={};budget_rows=[]
    for rank in cfg['ranks']:
        rows=list(read_jsonl(root/'sft_data'/f'{rank}.jsonl'));datasets[rank]=rows
        if {r['problem_id'] for r in rows}!=set(support):raise ValueError('SFT support mismatch')
        if rank.startswith('equal_examples__') and len(rows)!=len(support):raise ValueError('Equal-example duplicate')
        budget=next(b for b in data['budgets'] if f"{b['regime']}__{b['condition']}"==rank)
        tokens=0
        for row in rows:
            pid=row['problem_id']
            if row['prompt']!=questions[pid]['student_prompt']:raise ValueError('Training prompt mismatch')
            if not verify_answer(extract_final_answer(row['completion']),questions[pid]['answer']):raise ValueError('Incorrect selected SFT trace')
            n=len(tok.encode(row['completion'],add_special_tokens=False));tokens+=n
            if n!=row['completion_token_count']:raise ValueError('Supervision token count mismatch')
        if tokens!=budget['completion_tokens'] or len(rows)!=budget['rows'] or dict(Counter(r['problem_id'] for r in rows))!=budget['question_weights']:
            raise ValueError('Budget/weight audit mismatch')
        budget_rows.append({k:v for k,v in budget.items() if k!='question_weights'})
    fixtures=list(read_jsonl(root/'inputs/evaluation.jsonl'))
    expected_ids=[r['problem_id'] for r in fixtures];eval_gold={r['problem_id']:r['answer'] for r in fixtures}
    if [r['source_index'] for r in fixtures]!=list(range(50,1319)):raise ValueError('Locked evaluation split mismatch')
    summary=read_json(root/'analysis/summary.json');eval_counts={};adapter_count=0
    names=['base']+[f'{r}__seed_{s}' for r in cfg['ranks'] for s in cfg['student_seeds']]
    for name in names:
        directory=root/'student_evaluation'/name
        rows=list(read_jsonl(directory/'predictions.jsonl'))
        if [r['problem_id'] for r in rows]!=expected_ids:raise ValueError('Evaluation missing/duplicate/order mismatch')
        for r in rows:
            gold=eval_gold[r['problem_id']]
            if r['gold_answer']!=gold or r['is_correct']!=verify_answer(extract_final_answer(r['prediction_text']),gold):
                raise ValueError('Evaluation verifier mismatch')
        recomputed=summarize_predictions(rows)
        for key,value in recomputed.items():
            if not np.isclose(value,summary['metrics'][name][key],atol=1e-12):raise ValueError('Evaluation aggregate mismatch')
        eval_counts[name]=len(rows)
        if name=='base':continue
        rank,seed=name.split('__seed_')
        adapter=Path(cfg['checkpoint_root'])/'students'/name
        metrics=read_json(adapter/'training_metrics.json')
        if metrics['records']!=len(datasets[rank]) or metrics['steps']!=math.ceil(len(datasets[rank])/4) or metrics['seed']!=int(seed):
            raise ValueError('Training step/seed/record count mismatch')
        if not any(r.get('learning_rate',0)>0 for r in metrics['log_history']):raise ValueError('No positive logged learning rate')
        if not all(math.isfinite(v) for r in metrics['log_history'] for k,v in r.items() if isinstance(v,(float,int))):
            raise ValueError('Nonfinite training log')
        adapter_count+=1
    logging.info('All %s adapters and %s complete evaluations audited',adapter_count,len(eval_counts))
    full_teacher_effects={}
    for field in ('output_token_count','is_correct'):
        groups={name:defaultdict(list) for name in ('no_steering','selected_target')}
        for row in teacher_predictions:
            if row['condition'] in groups:groups[row['condition']][row['problem_id']].append(float(row[field]))
        means={name:{pid:float(np.mean(values)) for pid,values in records.items()} for name,records in groups.items()}
        full_teacher_effects[field]=paired_question_bootstrap(means['selected_target'],means['no_steering'],
            samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
    repeated_controls=[]
    for seed in cfg['student_seeds']:
        a=f'equal_examples__matched_random__seed_{seed}';b=f'equal_target_tokens__matched_random__seed_{seed}'
        repeated_controls.append(dict(seed=seed,
            adapter_weights_identical=digest(Path(cfg['checkpoint_root'])/'students'/a/'adapter_model.safetensors')==digest(Path(cfg['checkpoint_root'])/'students'/b/'adapter_model.safetensors'),
            predictions_identical=digest(root/'student_evaluation'/a/'predictions.jsonl')==digest(root/'student_evaluation'/b/'predictions.jsonl')))
    report=dict(status='passed',hash_comparisons=comparisons,unique_hashed_files=len(cached),verified_markers=len(visited),
        main_generation_records=len(teacher_predictions),common_support_count=len(support),adapter_count=adapter_count,
        evaluation_runs=len(eval_counts),evaluation_records=sum(eval_counts.values()),records_per_evaluation=eval_counts,
        budgets=budget_rows,full_teacher_diagnostics=diagnostics,full_teacher_effects=full_teacher_effects,
        repeated_random_controls=repeated_controls,formal_claim_allowed=False,
        note='Paired full-teacher effects are descriptive post-run summaries; no gate, tuning or model selection changed.',
        audit_source_sha256=file_sha256(Path(__file__)),verified_hashes=cached,
        completion_marker_sha256=digest(teacher_root/'INTERVENTION_COMPLETE.json'))
    parent.save(output,report)
    return report
