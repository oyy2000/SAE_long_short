"""Conditional current-data teacher generation and original-recipe student SFT.

Reuse the measured controller, the frozen legacy TRL trainer/evaluator, and the
historical repeat-to-token-budget helper without modifying sealed source files.
"""
from __future__ import annotations
import copy
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

from .experiment_io import read_json,write_json_exclusive
from .factorial import file_sha256
from .sae_local_data import ROOT,paths,jsonl,write_jsonl,evidence,verify
from .sae_joint_control import JointRandomController
from .sae_norm_intervention import generate_condition
from . import legacy_replication as legacy
from . import local_short_gain_check as gain
from .utility_analysis import paired_question_bootstrap
from .sae_feature_analysis import holm_adjust


def require_teacher_gate(config):
    root,_=paths(config)
    original=read_json(root/'TEACHER_DECISION.json')
    corrected=read_json(root/'MATCHED_CONTROL_DECISION.json')
    if not original['student_authorized_by_protocol'] or not corrected['matched_control_gate_passed']:
        raise RuntimeError('Teacher gate did not authorize student follow-up')
    if original['selected_condition']!=corrected['selected_condition']:
        raise ValueError('Selected target differs between gates')


def freeze_main(config):
    require_teacher_gate(config)
    root,_=paths(config)
    output=root/'student_followup'
    output.mkdir(exist_ok=False)
    control=read_json(root/'matched_joint_control/protocol.json')
    specs=copy.deepcopy(control['specs'])
    for spec,name in zip(specs,['no_steering','selected_target','matched_random']):spec['name']=name
    corpus=jsonl(root/'corpus.jsonl')
    ids=sorted({r['problem_id'] for r in corpus})
    if len(ids)!=config['student_followup']['question_count']:raise ValueError('Wrong generation cohort')
    protocol={'status':'frozen','question_ids':ids,'question_count':len(ids),
              'candidates_per_question':config['student_followup']['candidates_per_question'],
              'shards':7,'batch_size':8,'specs':specs,'base_seed':61317,
              'control':evidence(root/'matched_joint_control/protocol.json'),
              'parent_config':evidence(root/'protocol/frozen_protocol.json'),
              'teacher_gate':evidence(root/'TEACHER_DECISION.json'),
              'structural_gate':evidence(root/'MATCHED_CONTROL_DECISION.json'),
              'source':evidence(Path(__file__)),'expected_predictions':len(ids)*4*3,
              'correct_selection':'shortest unique correct EOS-completed candidate; tie by candidate index',
              'same_question_support':'intersection of all four SFT conditions',
              'equal_token_rule':'reuse sealed Phase-3 whole-trace repeat-to-maximum-target helper; maximum token gap 512',
              'formal_claim_allowed':False}
    write_json_exclusive(output/'main_protocol.json',protocol)


def generate_main(config,shard):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    require_teacher_gate(config)
    root,_=paths(config);output=root/'student_followup'
    protocol=read_json(output/'main_protocol.json')
    for key in ['source','control','parent_config','teacher_gate','structural_gate']:verify(protocol[key])
    control=read_json(protocol['control']['path']);verify(control['checkpoint']);verify(control['source'])
    directory=output/'generation'/f'shard_{shard}'
    directory.mkdir(parents=True,exist_ok=False)
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl') if r['analysis_length_label']=='short'}
    rows=[corpus[pid] for i,pid in enumerate(protocol['question_ids']) if i%protocol['shards']==shard]
    teacher=config['teacher']['snapshot_path']
    tokenizer=AutoTokenizer.from_pretrained(teacher,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(teacher,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    short=[f['feature_id'] for f in control['features'] if f['direction']=='short']
    long=[f['feature_id'] for f in control['features'] if f['direction']=='long']
    controller=JointRandomController(positive_ids=control['positive_random_ids'],negative_ids=control['negative_random_ids'],
        short_ids=short,long_ids=long,random_ids=control['positive_random_ids'],torch_module=torch,
        checkpoint_path=control['checkpoint']['path'],layer_module=model.model.layers[17],k=64,
        maximum_delta_fraction=.30,device=model.device,dtype=torch.bfloat16)
    filename=directory/'predictions.jsonl';n=0
    with filename.open('x') as handle:
        for start in range(0,len(rows),protocol['batch_size']):
            batch=rows[start:start+protocol['batch_size']]
            for candidate in range(protocol['candidates_per_question']):
                settings=dict(config['generation'],seed=protocol['base_seed']+100000*candidate)
                for spec in protocol['specs']:
                    generated=generate_condition(model,tokenizer,controller,spec,batch,settings)
                    for row in generated:
                        row['candidate_index']=candidate
                        handle.write(json.dumps(row,ensure_ascii=False)+'\n');n+=1
                    handle.flush()
            print(json.dumps({'shard':shard,'questions':min(start+protocol['batch_size'],len(rows)),
                              'total_questions':len(rows),'records':n}),flush=True)
    write_json_exclusive(directory/'COMPLETE.json',{'status':'complete','records':n,
        'problem_ids':[r['problem_id'] for r in rows],'predictions':evidence(filename),
        'protocol':evidence(output/'main_protocol.json')})


def _repeat_to_target(rows,target,condition):
    # The historical entrypoint is sealed; importing its pure helper preserves it.
    filename=ROOT/'scripts/4_0_build_intervention_sft_datasets.py'
    spec=importlib.util.spec_from_file_location('sealed_phase3_sft_data',filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module._repeat_to_target(rows,target,seed=61317,condition=condition)


def build(config):
    from transformers import AutoTokenizer
    from .verifiers import extract_final_answer,verify_answer
    require_teacher_gate(config)
    root,_=paths(config);output=root/'student_followup'
    protocol=read_json(output/'main_protocol.json');verify(protocol['source'])
    predictions=[]
    for shard in range(protocol['shards']):
        marker=read_json(output/'generation'/f'shard_{shard}'/'COMPLETE.json')
        verify(marker['predictions']);verify(marker['protocol'])
        part=jsonl(marker['predictions']['path'])
        if len(part)!=marker['records']:raise ValueError('Generation count mismatch')
        predictions+=part
    expected={(p,s['name'],c) for p in protocol['question_ids'] for s in protocol['specs'] for c in range(4)}
    observed=[(r['problem_id'],r['condition'],r['candidate_index']) for r in predictions]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError('Missing or duplicate full-cohort generation records')
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl') if r['analysis_length_label']=='short'}
    grouped={name:{} for name in ['no_steering','selected_target','matched_random']}
    seeds={}
    for row in predictions:
        gold=extract_final_answer(corpus[row['problem_id']]['gold_answer'])
        if gold!=row['gold_answer'] or row['is_correct']!=verify_answer(extract_final_answer(row['response']),gold):
            raise ValueError('Incorrect full-cohort gold or verifier result')
        key=(row['problem_id'],row['candidate_index'])
        if key in seeds and seeds[key]!=row['seed']:raise ValueError('Unpaired generation RNG')
        seeds[key]=row['seed']
        if row['is_correct'] and not row['hit_max_new_tokens']:
            grouped[row['condition']].setdefault(row['problem_id'],[]).append(row)
    support=sorted(set.intersection(*(set(v) for v in grouped.values())))
    if not support:raise ValueError('No all-condition correct support')
    original={r['metadata']['problem_id']:r for r in jsonl(root/'imported/short.jsonl')}
    e1=read_json(ROOT/'configs/phase1_local_short_gain_check_v1.json')
    tokenizer=AutoTokenizer.from_pretrained(e1['student_snapshot'],local_files_only=True)
    datasets={name:[] for name in config['student_followup']['conditions']}
    for pid in support:
        for condition in datasets:
            if condition=='historical_short':
                row=copy.deepcopy(original[pid]);candidate=-1
            else:
                # Duplicate texts are collapsed before shortest-correct selection.
                unique={}
                for item in sorted(grouped[condition][pid],key=lambda r:r['candidate_index']):
                    unique.setdefault(item['response'].strip(),item)
                chosen=min(unique.values(),key=lambda r:(r['output_token_count'],r['candidate_index']))
                row={'id':f"{pid}:{condition}:candidate_{chosen['candidate_index']}",
                     'prompt':corpus[pid]['student_prompt'],'teacher_prompt':corpus[pid]['prompt'],
                     'completion':chosen['response'],'metadata':{'problem_id':pid,'is_correct':True}}
                candidate=chosen['candidate_index']
            count=len(tokenizer.encode(row['completion'],add_special_tokens=False))
            row.update(problem_id=pid,trace_id=row['id'],occurrence_index=0,
                       completion_token_count=count,source_candidate_index=candidate)
            row['metadata']['solution_token_count']=count
            datasets[condition].append(row)
    target=max(sum(r['completion_token_count'] for r in rows) for rows in datasets.values())
    regimes={'equal_examples':datasets,
             'equal_target_tokens':{name:_repeat_to_target(rows,target,name) for name,rows in datasets.items()}}
    totals={name:sum(r['completion_token_count'] for r in rows) for name,rows in regimes['equal_target_tokens'].items()}
    if max(totals.values())-min(totals.values())>512:raise ValueError('Unequal target-token budget exceeds tolerance')
    sft=output/'sft';sft.mkdir()
    checkpoint=ROOT/config['checkpoint_root']/'students';checkpoint.mkdir()
    (sft/'replication').mkdir();(sft/'replication/checkpoints').symlink_to(checkpoint,target_is_directory=True)
    (sft/'logs').mkdir()
    student_config={k:copy.deepcopy(e1[k]) for k in ['student_revision','student_snapshot','student_model_sha256','versions','seed','training','lora','evaluation','bootstrap_samples','bootstrap_seed']}
    student_config.update(experiment_name=config['experiment_name']+'_student_followup',result_root=str(sft),
        checkpoint_root=str(checkpoint),runtime_root=str(Path(config['runtime_root'])/'students'),formal_claim_allowed=False)
    student_config['arms']=[f'{regime}__{condition}__seed_17' for regime in regimes for condition in datasets]
    config_path=sft/'student_config.json'
    write_json_exclusive(config_path,student_config)
    student_config['_path']=str(config_path)
    files=[]
    source_root=ROOT/'results/phase1_local_short_gain_check_v1'
    source_manifest=read_json(source_root/'IMPORT_COMPLETE.json')
    for item in source_manifest['files']:
        if '/legacy_code/' in item['path']:
            relative=item['path'].split('/legacy_code/')[1]
            files.append(legacy.copy_checked(item['path'],sft/'legacy_code'/relative,item['sha256']))
    fixture=source_root/'imported/locked_evaluation_questions.jsonl'
    files.append(legacy.copy_checked(fixture,sft/'imported/locked_evaluation_questions.jsonl',e1['fixture_sha256']))
    eval_config={'dataset':{'source':'local_jsonl','path':str(sft/'imported/locked_evaluation_questions.jsonl'),
                            'question_field':'question','answer_field':'answer'},'logical_split':'test[50:1319]'}
    write_json_exclusive(sft/'replication/eval_config.json',eval_config)
    template=read_json(source_root/'replication/configs/short__seed_17.json')
    runs=[];budget=[]
    for regime,cells in regimes.items():
        for condition,rows in cells.items():
            name=f'{regime}__{condition}__seed_17'
            filename=sft/'data'/regime/f'{condition}.jsonl';write_jsonl(filename,rows);files.append(evidence(filename))
            run=copy.deepcopy(template);run['experiment_name']=name
            run['data']['train_path']=str(filename)
            run['student']['model_name']=e1['student_snapshot'];run['student']['tokenizer_name']=e1['student_snapshot']
            run['training']['output_dir']=str(checkpoint/name)
            run['replication_evidence'].update(rank=condition,train_sha256=file_sha256(filename),expected_steps=math.ceil(len(rows)/4))
            run_path=sft/'replication/configs'/f'{name}.json';run_path.parent.mkdir(parents=True,exist_ok=True)
            write_json_exclusive(run_path,run)
            runs.append({'name':name,'rank':condition,'seed':17,'config_path':str(run_path),'config_sha256':file_sha256(run_path)})
            budget.append({'name':name,'regime':regime,'condition':condition,'rows':len(rows),'unique_questions':len(support),
                           'completion_tokens':sum(r['completion_token_count'] for r in rows),'steps':math.ceil(len(rows)/4)})
    write_json_exclusive(sft/'IMPORT_COMPLETE.json',{'status':'complete','files':files,'runs':runs,
        'config_sha256':file_sha256(config_path),'eval_config_sha256':file_sha256(sft/'replication/eval_config.json')})
    write_json_exclusive(sft/'RUNTIME_SOURCES.json',{'hashes':{str(p):file_sha256(p) for p in [Path(__file__),Path(legacy.__file__),Path(gain.__file__)]}})
    write_json_exclusive(output/'DATA_COMPLETE.json',{'status':'complete','support':support,'support_count':len(support),
        'all_generation_rows':len(predictions),'correct_support_by_condition':{k:len(v) for k,v in grouped.items()},
        'budget':budget,'repeat_helper':evidence(ROOT/'scripts/4_0_build_intervention_sft_datasets.py'),
        'student_config':evidence(config_path),'formal_claim_allowed':False})


def train_eval(config,arm):
    import torch
    root,_=paths(config);sft=root/'student_followup/sft'
    student=legacy.config_load(sft/'student_config.json')
    manifest=read_json(sft/'IMPORT_COMPLETE.json')
    if arm not in [r['name'] for r in manifest['runs']]:raise ValueError('Unregistered SFT arm')
    runtime=Path(student['runtime_root'])/arm
    cache=runtime/'datasets';cache.mkdir(parents=True,exist_ok=True)
    temp=runtime/'tmp';temp.mkdir(exist_ok=True)
    os.environ['HF_DATASETS_CACHE']=str(cache);os.environ['TMPDIR']=str(temp)
    legacy.train(student,arm,runtime/'training')
    gc.collect();torch.cuda.empty_cache()
    gain.evaluate(student,arm,runtime/'evaluation')


def analyze_student(config):
    import numpy as np
    root,_=paths(config);output=root/'student_followup';sft=output/'sft'
    student=legacy.config_load(sft/'student_config.json')
    manifest=legacy.source_hashes(student)
    metrics=[];predictions={}
    base_root=ROOT/'results/phase1_local_short_gain_check_v1'
    base_config=legacy.config_load(ROOT/'configs/phase1_local_short_gain_check_v1.json')
    base=gain.validate_evaluation(base_config,'base')
    # Fixed model bytes, identical fixture, same evaluator and runtime versions.
    verify({'path':str(Path(student['student_snapshot'])/'model.safetensors'),'sha256':student['student_model_sha256']})
    if file_sha256(sft/'imported/locked_evaluation_questions.jsonl')!=file_sha256(base_root/'imported/locked_evaluation_questions.jsonl'):
        raise ValueError('Cannot reuse a different baseline cohort')
    for entry in manifest['runs']:
        name=entry['name'];training=legacy.validate_training(student,name)
        rows=gain.validate_evaluation(student,name);predictions[name]={r['problem_id']:r for r in rows}
        metrics.append({'name':name,'accuracy':float(np.mean([r['is_correct'] for r in rows])),
                        'correct':sum(r['is_correct'] for r in rows),'n':len(rows),
                        'mean_output_tokens':float(np.mean([r['output_token_count'] for r in rows])),
                        'training_steps':training['optimizer_steps'],'training_rows':training['record_count']})
    contrasts=[]
    for regime in config['student_followup']['budget_regimes']:
        target=f'{regime}__selected_target__seed_17'
        for comparator in ['no_steering','matched_random','historical_short']:
            other=f'{regime}__{comparator}__seed_17'
            effect=paired_question_bootstrap({p:float(r['is_correct']) for p,r in predictions[target].items()},
                {p:float(r['is_correct']) for p,r in predictions[other].items()},samples=10000,seed=61417)
            contrasts.append({'target':target,'comparator':other,**effect})
    for row,p in zip(contrasts,holm_adjust([r['bootstrap_p_value'] for r in contrasts])):row['holm_p']=float(p)
    report={'status':'complete','training_runs':len(metrics),'new_evaluations':len(metrics),'reused_base':{
        'accuracy':float(np.mean([r['is_correct'] for r in base])),'n':len(base),
        'predictions':evidence(base_root/'evaluation/base/predictions.jsonl')},
        'metrics':metrics,'contrasts':contrasts,'seed':17,'training_seed_variability_estimated':False,
        'formal_claim_allowed':False}
    write_json_exclusive(output/'STUDENT_COMPLETE.json',report)
    print(json.dumps(report,indent=2),flush=True)
