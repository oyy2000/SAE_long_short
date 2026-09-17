"""Separate, parent-bound 3B rerun of the historical 1.5B optimization recipe.

The current question support, prompts, explicit completion labels and grader
remain fixed. This is a recipe sensitivity study, not an old-data replication.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import unittest

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import save, seal, verify
from .factorial import file_sha256
from . import gsm8k_pilot_runtime as runtime
from .gsm8k_student_pilot import freeze_child

CODE = Path(__file__).resolve().parents[2]
ENTRY = '13_64_run_gsm8k_recipe_alignment.py'


def aligned_config(parent, reference, spec, root):
    cfg = copy.deepcopy(parent)
    for key in ('execution_marker', 'execution_tests_marker', 'child_roots', 'pilot_root',
                'directions_execution_config'):
        cfg.pop(key, None)
    cfg.update(result_root=str(root), code_root=str(root/'code'),
               checkpoint_root=str((CODE/spec['checkpoint_root']).resolve()),
               entrypoint=ENTRY, alignment=spec,
               execution_marker=str(root/'protocol/FROZEN.json'),
               execution_config_path=str(root/'protocol/frozen_config.json'))
    cfg['training'] = copy.deepcopy(reference['training'])
    # The frozen current data already carry exact masked completion labels.
    cfg['training'].update(completion_only_loss=None, dataset_kwargs={}, disable_tqdm=True,
                           lr_scheduler_type='linear', max_grad_norm=1.0, weight_decay=0.0)
    cfg['lora'] = copy.deepcopy(reference['student']['lora'])
    cfg['student_seeds'] = list(reference['student_seeds'])
    cfg['registered_student_cells'] = [['base',17]] + [
        [m,s] for s in cfg['student_seeds'] for m in spec['methods']]
    cfg['caps'] = [reference['evaluation']['max_new_tokens']]
    cfg['batch_size'] = reference['evaluation']['batch_size']
    cfg['repetition_penalty'] = spec['evaluation_repetition_penalty']
    cfg['recovery'] = {**cfg.get('recovery',{}),'worker_execution_configs':[cfg['execution_config_path']],
        'controller_prefix':'aligned_recovery','run_luna_locally':False,
        'maximum_retries':3,'maximum_controller_cycles':24,'controller_poll_seconds':60}
    cfg['claim_boundary'] = ('Exploratory GSM8K recipe sensitivity on previously observed evaluation questions; '
        'current 1024-question support and traces retained, historical 1.5B optimizer/LoRA/decoding budget adopted. '
        'Not a strict replication of the historical 878-question experiment; no new independent confirmation.')
    return cfg


def freeze(config_path):
    spec = read_json(config_path)
    parent_path = (CODE/spec['parent_config']).resolve()
    reference_path = (CODE/spec['reference_config']).resolve()
    root = (CODE/spec['result_root']).resolve()
    if root.exists(): raise FileExistsError(root)
    parent = read_json(parent_path); reference = read_json(reference_path)
    if (parent['teacher']['model_name'],parent['teacher']['revision']) != (
            reference['teacher']['model_name'],reference['teacher']['revision']):
        raise ValueError('Teacher identity differs between the two experiments')
    generation_path=Path(reference['student']['snapshot_path'])/'generation_config.json'
    spec = {**spec,'parent_config':str(parent_path),'reference_config':str(reference_path),
            'evaluation_repetition_penalty':read_json(generation_path)['repetition_penalty']}
    cfg = aligned_config(parent,reference,spec,root)
    for name in ('src','scripts','tests'):
        shutil.copytree(CODE/name,root/'code'/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    sources = [p for p in (root/'code').rglob('*') if p.is_file()]
    seal(root/'protocol/SOURCES.json',sources)
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',[root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
         Path(config_path).resolve(),parent_path,reference_path,generation_path],formal_claim_allowed=False)
    job = runtime.submit(cfg,'prepare','prepare-aligned')
    controller = runtime.submit(cfg,'aligned_recovery_0','recover',route='monitor_cpu')
    save(root/'launch/INITIAL_SUBMISSION.json',{'prepare_job':job,'controller_job':controller,'experiment_complete':False})
    return job


def prepare(cfg):
    from .completion_supervision import validate_encoded_record
    from transformers import TrainingArguments
    root = Path(cfg['result_root']); spec = cfg['alignment']
    parent = read_json(CODE/spec['parent_config'])
    # Reference paths are resolved before the source snapshot is frozen.
    source = Path(parent['result_root'])
    reference_path = Path(spec['reference_config'])
    reference = read_json(reference_path)
    historical_source = reference_path.parents[1]/'code/src/length_budget_distill/training.py'
    if not historical_source.is_file(): raise FileNotFoundError(historical_source)
    if 'lr_scheduler_type' not in historical_source.read_text():
        raise ValueError('Historical training wrapper must be inspected for scheduler handling')
    defaults = {k:TrainingArguments.__dataclass_fields__[k].default
                for k in ('lr_scheduler_type','max_grad_norm','weight_decay')}
    if defaults != {'lr_scheduler_type':'linear','max_grad_norm':1.0,'weight_decay':0.0}:
        raise ValueError('Historical optimizer defaults differ from the aligned recipe')
    suite = unittest.TestSuite()
    for pattern in ('test_gsm8k_student_pilot.py','test_gsm8k_recipe_alignment.py',
                    'test_completion_supervision.py','test_pilot_storage.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(CODE/'tests'),pattern=pattern))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful(): raise RuntimeError('Alignment regression checks failed')
    bindings = []
    for rel in ('protocol/FROZEN.json','sft/protocol/FROZEN.json','sft/protocol/SOURCES.json',
                'final_review_v1/COMPLETE.json'):
        marker=source/rel; verify(marker); bindings.append(marker)
    for name in ('model_hashes.json','runtime_versions.json','evaluation.jsonl','development.jsonl'):
        target=root/'inputs'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'inputs'/name,target);bindings.append(target)
    methods=spec['methods']; common=None; audit={}
    for method in methods:
        for kind in ('text','encoded/qwen3b_student'):
            src=source/'sft'/kind/(method+'.jsonl'); dest=root/'sft'/kind/(method+'.jsonl')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            if file_sha256(src)!=file_sha256(dest): raise ValueError('Copied training data changed')
            bindings += [src,dest]
        rows=list(read_jsonl(root/'sft/encoded/qwen3b_student'/f'{method}.jsonl'))
        ids=[r['problem_id'] for r in rows]
        if len(ids)!=spec['expected_questions'] or len(set(ids))!=len(ids):
            raise ValueError('Missing or duplicate training problems')
        if common is not None and common!=ids: raise ValueError('Training support/order differs')
        common=ids
        for row in rows: validate_encoded_record(row,max_length=cfg['training']['max_length'])
        audit[method]={'questions':len(rows),'maximum_sequence_tokens':max(len(r['input_ids']) for r in rows),
                       'expected_steps':len(rows)//4,'supervision_tokens':sum(r['supervision_tokens'] for r in rows)}
    for name in ('model_hashes.json','runtime_versions.json'):
        target=root/'sft/inputs'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'sft/inputs'/name,target)
    save(root/'sft/inputs/alignment_audit.json',{'methods':audit,'same_parent_data_bytes':True,
         'historical_defaults':defaults,'regression_tests':outcome.testsRun,
         'preserved_differences':['1024 rather than 878 questions','current prompts and grader',
             'explicit pretokenized completion masks and native EOS supervision',
             'three selected conditions; equal-example budget only'],
         'historical_reference':str(reference_path),'formal_claim_allowed':False})
    child={**cfg,'actual_common_questions':spec['expected_questions'],
           'primary_stage':{'students':['qwen3b_student'],'methods':methods},'regime':'equal_examples'}
    freeze_child(cfg,child,root/'sft',bindings)
    protocol={'cells':cfg['registered_student_cells'],'max_new_tokens':cfg['caps'][0],
              'tokenskip_ratio':None,'selection_uses_test_outputs':False,'all_cells_prespecified':True,
              'previously_observed_test_cohort':True,'split':'test[50:1319]','count':1269}
    save(root/'evaluation_protocol/protocol.json',protocol)
    seal(root/'evaluation_protocol/COMPLETE.json',[root/'protocol/FROZEN.json',
         root/'inputs/evaluation.jsonl',root/'evaluation_protocol/protocol.json'],formal_claim_allowed=False)
    seal(root/'PREPARE_COMPLETE.json',[root/'sft/protocol/FROZEN.json',root/'evaluation_protocol/COMPLETE.json'],
         formal_claim_allowed=False,regression_tests=outcome.testsRun)
    # A full first cell measures the new batch/LoRA recipe before releasing the matrix.
    job=runtime.submit(cfg,'train_B1_17','train',route='h200',method='B1',seed=17)
    smoke=runtime.submit(cfg,'eval_B1_17_0','evaluate-test',route='h200',parents=[job],method='B1',seed=17,shard=0)
    runtime.submit(cfg,'release_matrix','queue-aligned',parents=[smoke])


def queue(cfg, *, analysis_code_root=None):
    root=Path(cfg['result_root']);verify(root/'PREPARE_COMPLETE.json')
    adapter=Path(cfg['checkpoint_root'])/'qwen3b_student/B1/seed_17'
    verify(adapter/'TRAIN_COMPLETE.json')
    measured=read_json(adapter/'training_metrics.json')
    smoke=root/'evaluation/B1/seed_17/shard_00'
    verify(smoke/'COMPLETE.json');evaluation=read_json(smoke/'summary.json')
    if measured['optimizer_steps']!=256 or measured['actual_epoch']!=1:
        raise ValueError('Aligned first-cell training exposure differs')
    if measured['peak_gpu_allocated_mib']+cfg['alignment']['gpu_memory_safety_margin_mib']>cfg['runtime']['minimum_free_mib']:
        raise ValueError('Measured memory exceeds registered admission budget')
    l40s_eligible=(max(measured['peak_gpu_reserved_mib'],evaluation['peak_gpu_allocated_mib'])+
        cfg['alignment']['gpu_memory_safety_margin_mib'] <= cfg['runtime']['minimum_free_mib'] and
        max(measured['elapsed_seconds_including_batch_audit'],evaluation['elapsed_seconds'])*
        cfg['alignment']['l40s_runtime_multiplier']+600 < 7200)
    evaluation_jobs=[];lanes=[None]*cfg['runtime']['maximum_gpu_lanes']
    for index,(method,seed) in enumerate(cfg['registered_student_cells']):
        lane=index%len(lanes); deps=[lanes[lane]] if lanes[lane] else []
        route='l40s' if l40s_eligible and lane%2 else 'h200'
        if method!='base' and (method,seed)!=('B1',17):
            job=runtime.submit(cfg,f'train_{method}_{seed}','train',route=route,parents=deps,method=method,seed=seed)
            deps=[job]
        for shard in range(cfg['evaluation_shards']):
            if (method,seed,shard)==('B1',17,0):continue
            job=runtime.submit(cfg,f'eval_{method}_{seed}_{shard}','evaluate-test',route=route,
                               parents=deps,method=method,seed=seed,shard=shard)
            deps=[job];evaluation_jobs.append(job)
        lanes[lane]=job
    analysis_cfg=cfg
    if analysis_code_root is not None:
        analysis_cfg={**cfg,'code_root':str(analysis_code_root),'entrypoint':'13_65_finalize_gsm8k_recipe_alignment.py'}
    analysis_job=runtime.submit(analysis_cfg,'analyze','analyze-aligned',parents=evaluation_jobs)
    save(root/'launch/MATRIX_SUBMITTED.json',{'evaluation_jobs':evaluation_jobs,'experiment_complete':False,
         'first_cell_peak_reserved_mib':measured['peak_gpu_reserved_mib'],
         'evaluation_smoke':evaluation,'l40s_eligible':l40s_eligible,'analysis_job':analysis_job})


def analyze(cfg):
    from .ranked_multiseed_analysis import crossed_seed_problem_bootstrap
    from .sae_feature_analysis import holm_adjust
    root=Path(cfg['result_root']);questions=list(read_jsonl(root/'inputs/evaluation.jsonl'))
    predictions={};metrics=[];markers=[root/'PREPARE_COMPLETE.json',Path(cfg['execution_marker'])]; merged=[]
    verify(cfg['execution_marker'])
    analysis_sources=CODE.parent/'protocol/SOURCES.json'
    verify(analysis_sources);markers.append(analysis_sources)
    for method,seed in cfg['registered_student_cells']:
        rows=[]
        if method!='base':
            adapter=Path(cfg['checkpoint_root'])/'qwen3b_student'/method/f'seed_{seed}'
            verify(adapter/'TRAIN_COMPLETE.json');markers.append(adapter/'TRAIN_COMPLETE.json')
            trained=read_json(adapter/'training_metrics.json')
            if trained['optimizer_steps']!=256 or trained['actual_epoch']!=1:
                raise ValueError('Incomplete aligned training exposure')
        for shard in range(cfg['evaluation_shards']):
            part=root/'evaluation'/method/f'seed_{seed}'/f'shard_{shard:02d}'
            marker=part/'COMPLETE.json';verify(marker);markers.append(marker)
            values=list(read_jsonl(part/'predictions.jsonl'))
            expected={q['problem_id'] for q in questions[shard::cfg['evaluation_shards']]}
            if len(values)!=len(expected) or {r['problem_id'] for r in values}!=expected:
                raise ValueError('Missing/duplicate evaluation predictions')
            if any(r['method']!=method or r['seed']!=seed or r['max_new_tokens']!=512 for r in values):
                raise ValueError('Changed evaluation cell/budget')
            rows.extend(values)
        predictions[(method,seed)]={r['problem_id']:int(r['grade']['is_correct']) for r in rows}
        merged.extend(rows)
        metrics.append({'method':method,'seed':seed,'questions':len(rows),
                        'accuracy':statistics.mean(r['grade']['is_correct'] for r in rows),
                        'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rows),
                        'cap_hit_rate':statistics.mean(r['hit_max_new_tokens'] for r in rows)})
    comparisons={}
    for method in cfg['alignment']['methods'][1:]:
        for baseline in ('base','B1'):
            effects={s:{pid:value-predictions[(baseline,17 if baseline=='base' else s)][pid]
                        for pid,value in predictions[(method,s)].items()} for s in cfg['student_seeds']}
            comparisons[f'{method}_minus_{baseline}']=crossed_seed_problem_bootstrap(
                effects,samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
    for row,p in zip(comparisons.values(),holm_adjust([r['bootstrap_p_value'] for r in comparisons.values()])):
        row['holm_bootstrap_p_value']=float(p)
    out=root/'analysis';write_jsonl(out/'metrics.jsonl',metrics);write_jsonl(out/'all_predictions.jsonl',merged)
    save(out/'comparisons.json',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for index,method in enumerate(cfg['alignment']['methods']):
        ys=[100*r['accuracy'] for r in metrics if r['method']==method]
        ax.bar(index,statistics.mean(ys),color='#4c72b0' if method=='B1' else '#c44e52')
        ax.scatter([index]*len(ys),ys,color='black',s=16)
    ax.axhline(100*metrics[0]['accuracy'],color='gray',linestyle='--',label='Base, 512-token cap')
    ax.set_xticks(range(3),cfg['alignment']['methods']);ax.set_ylabel('GSM8K accuracy (%)');ax.legend()
    fig.tight_layout();fig.savefig(out/'student_accuracy.png',dpi=180);plt.close(fig)
    report=['# 3B historical-recipe sensitivity','',cfg['claim_boundary'],'',
            '| Method | Seed | Accuracy | Mean tokens |','| --- | ---: | ---: | ---: |']
    report += [f"| {r['method']} | {r['seed']} | {100*r['accuracy']:.2f}% | {r['mean_output_tokens']:.2f} |" for r in metrics]
    report += ['','![Student accuracy](student_accuracy.png)','',
               'Question and training-seed crossed bootstrap; four planned contrasts adjusted together. '
               'The 512-token base must be used for this comparison; the earlier 86.84% used 1024 tokens.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    seal(out/'COMPLETE.json',[*markers,*sorted(out.glob('*'))],formal_claim_allowed=False)
    seal(root/'EXPERIMENT_COMPLETE.json',[root/'protocol/FROZEN.json',out/'COMPLETE.json'],
         formal_claim_allowed=False,trained_adapters=9,predictions=len(merged))


def dispatch(args):
    if args.stage=='freeze-aligned':
        print(freeze(args.config));return
    cfg=read_json(args.config);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if CODE!=Path(cfg['code_root']): raise ValueError('Use the frozen aligned source snapshot')
    if args.stage=='prepare-aligned':prepare(cfg)
    elif args.stage=='queue-aligned':queue(cfg)
    elif args.stage=='analyze-aligned':analyze(cfg)
    else:runtime.dispatch(args)
