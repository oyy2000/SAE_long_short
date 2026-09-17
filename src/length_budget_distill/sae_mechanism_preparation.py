"""Freeze fresh SAE mechanism cohorts and generate unmodified reference traces.

This is the input stage of S1b. It neither rediscovers features nor completes
the later causal controls, same-state readback, or student distillation.
"""
from collections import Counter
from pathlib import Path
from fractions import Fraction
import ast
import difflib
import importlib.metadata
import json
import logging
import os
import random
import re
import shutil
import subprocess

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify, evidence, admission, teacher_bundle
from .baseline_data_preflight import load_source, normalize_question, near_matches
from .baseline_reproduction import generate_text, summarize_generations
from .baseline_method_analysis import audit_cohort
from .ranked_sampling import build_length_agnostic_teacher_prompt
from .gsm8k_grading_v3 import grade_gsm8k_response

CODE = Path(__file__).resolve().parents[2]


def arithmetic_value(expression):
    """Evaluate a bounded numeric annotation without executing dataset code."""
    if len(expression) > 150:
        raise ValueError('Annotation too long')
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError('Annotation too complex')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            return left / right
        raise ValueError('Unsupported arithmetic annotation')
    return visit(tree.body)


def wrong_calculation(raw_body):
    """Change one displayed, exactly checked equation result by one digit.

    Only calculator annotations with an immediately repeated numeric result
    qualify. Unsupported or already inconsistent calculations remain visible
    as unavailable controls, never silently labelled as a verified mutation.
    """
    clean = re.sub(r'<<[^>]*>>', '', raw_body)
    for match in re.finditer(r'<<([^=<>]+)=([^<>]+)>>', raw_body):
        shown = re.match(r'([-+]?\d[\d,]*(?:\.\d+)?)', raw_body[match.end():])
        if shown is None: continue
        value = shown[1]
        try:
            expected = arithmetic_value(match[1])
            if expected != Fraction(match[2].strip().replace(',', '')) or expected != Fraction(value.replace(',', '')):
                continue
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
            continue
        at = len(re.sub(r'<<[^>]*>>', '', raw_body[:match.start()]))
        last = max(i for i, c in enumerate(value) if c.isdigit())
        changed = value[:last] + str((int(value[last]) + 1) % 10) + value[last+1:]
        if Fraction(changed.replace(',', '')) == expected:
            raise AssertionError('Mutation did not change numeric value')
        if clean[at:at+len(value)] != value:
            raise AssertionError('Annotation-to-visible span mismatch')
        return clean[:at]+changed+clean[at+len(value):], {
            'expression': match[1], 'verified_value': str(expected), 'original': value,
            'replacement': changed, 'start': at, 'end': at+len(value),
            'verification': 'Exact Fraction arithmetic; only this equation was checked.'}
    return None, None


def reference_variants(raw_answer):
    if raw_answer.count('####') != 1:
        raise ValueError('Expected exactly one GSM8K reference answer delimiter')
    raw_body, gold = raw_answer.split('####');gold = gold.strip()
    # Keep leading whitespace so annotation offsets remain directly reusable.
    body = re.sub(r'<<[^>]*>>', '', raw_body).rstrip()
    base = body + '\nAnswer: ' + gold
    variants = {
        'reference': base,
        'result_marker': body + '\nResult: ' + gold,
        'no_marker': body + '\n' + gold,
        'neutral_note': 'The label in this note is Entry.\n' + base,
        'answer_note': 'The label in this note is Answer.\n' + base,
    }
    changed, edit = wrong_calculation(raw_body)
    if changed is not None: variants['wrong_calculation'] = changed.rstrip() + '\nAnswer: ' + gold
    rows=[]
    for name,text in variants.items():
        operations = [{'operation': tag, 'base_span': [a,b], 'variant_span': [c,d]}
            for tag,a,b,c,d in difflib.SequenceMatcher(None,base,text,autojunk=False).get_opcodes()]
        rows.append({'variant':name,'text':text,'gold_answer':gold,'base_alignment':operations,
                     'equation_mutation':edit if name=='wrong_calculation' else None,
                     'expected_reasoning_preservation':name!='wrong_calculation',
                     'provenance':'GSM8K official reference, not a teacher-generated trace'})
    return rows


def partition_questions(rows, split_counts, smoke_count, seed):
    keys=[normalize_question(r['question']) for r in rows]
    if len(set(keys))!=len(rows) or len({r['problem_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate question input to split')
    ordered=sorted(rows,key=lambda r:canonical_sha256([seed,r['problem_id']]))
    if len(ordered)<sum(split_counts.values())+smoke_count:
        raise ValueError('Insufficient eligible unique questions')
    smoke=[{**r,'question_split':'smoke'} for r in ordered[:smoke_count]]
    offset=smoke_count;selected=[]
    for split,count in split_counts.items():
        selected.extend({**r,'question_split':split} for r in ordered[offset:offset+count]);offset+=count
    return smoke,selected


def prepare(config_path):
    from transformers import AutoTokenizer
    from safetensors import safe_open
    cfg=read_json(config_path);project=Path(cfg['project_root']);root=project/cfg['result_root']
    if root.exists():raise FileExistsError(root)
    parent=project/cfg['parent_intervention_root']
    parent_marker=parent/'protocol/FROZEN.json';verify(parent_marker)
    parent_cfg=read_json(parent/'protocol/frozen_protocol.json')
    for key in ('model_name','revision','snapshot_path'):
        if cfg['teacher'][key]!=parent_cfg['teacher'][key]:raise ValueError('Teacher mismatch: '+key)
    checkpoint=Path(parent_cfg['sae_checkpoint'])
    with safe_open(str(checkpoint),framework='pt') as handle:
        dictionary_size=handle.get_slice('encoder_bias').get_shape()[0]
    cfg['sae']={**cfg['sae'],'checkpoint_path':str(checkpoint),
                'short_features':parent_cfg['feature_ids']['short'],
                'long_features':parent_cfg['feature_ids']['long'],'dictionary_size':dictionary_size}
    if cfg['sae']['layer_index']!=parent_cfg['intervention']['layer'] or cfg['sae']['top_k']!=parent_cfg['intervention']['k']:
        raise ValueError('SAE layer or TopK mismatch')
    excluded_features=set(cfg['sae']['short_features']+cfg['sae']['long_features'])
    population=[i for i in range(dictionary_size) if i not in excluded_features]
    cfg['sae']['random_feature_sets']=[random.Random(seed).sample(population,len(cfg['sae']['short_features']))
                                      for seed in cfg['sae']['random_seeds']]
    history=[];bindings=[parent_marker,checkpoint,Path(config_path)]
    for path in cfg['history_question_sources']:
        path=project/path;history.extend(read_jsonl(path));bindings.append(path)
    histories={normalize_question(r['question']) for r in history}
    train=load_source({'path':cfg['gsm8k']['train_arrow']});test=load_source({'path':cfg['gsm8k']['test_arrow']})
    if len(train)!=7473 or len(test)!=1319:raise ValueError('GSM8K source count mismatch')
    blocked=histories | {normalize_question(r['question']) for r in test}
    candidates=[];seen=set();exclusions=[]
    for i,row in enumerate(train):
        pid=f'hf-{i:06d}';key=normalize_question(row['question'])
        reason='historical_or_evaluation_exact' if key in blocked else 'training_duplicate' if key in seen else None
        seen.add(key)
        if reason:exclusions.append({'problem_id':pid,'reason':reason});continue
        candidates.append({'problem_id':pid,'source_index':i,'question':row['question'],
            'raw_answer':row['answer'],'answer':row['answer'].split('####')[-1].strip(),
            'prompt':build_length_agnostic_teacher_prompt(row['question']),
            'source_revision':cfg['gsm8k']['revision']})
    refs={normalize_question(r['question']):{'problem_id':'history-'+str(i),'question':r['question']} for i,r in enumerate(history)}
    references=list(refs.values())+[{'problem_id':f'gsm-test-{i:05d}','question':r['question']} for i,r in enumerate(test)]
    near=near_matches(candidates,references,n=cfg['decontamination']['shingle_size'],threshold=cfg['decontamination']['jaccard_threshold'])
    near_ids={r['query_id'] for r in near}
    exclusions.extend({'problem_id':pid,'reason':'conservative_near_history_or_evaluation'} for pid in sorted(near_ids))
    eligible=[r for r in candidates if r['problem_id'] not in near_ids]
    smoke,selected=partition_questions(eligible,cfg['split_counts'],cfg['smoke_questions'],cfg['split_seed'])
    tok=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    variants=[]
    for question in smoke+selected:
        for variant in reference_variants(question['raw_answer']):
            encoded=tok(variant['text'],add_special_tokens=False,return_offsets_mapping=True)
            variants.append({**variant,'problem_id':question['problem_id'],'question_split':question['question_split'],
                             'token_ids':encoded['input_ids'],'token_offsets':encoded['offset_mapping']})
    inputs=root/'inputs';write_jsonl(inputs/'questions.jsonl',selected);write_jsonl(inputs/'smoke_questions.jsonl',smoke)
    write_jsonl(inputs/'reference_variants.jsonl',variants);write_jsonl(inputs/'exclusions.jsonl',exclusions)
    write_jsonl(inputs/'near_matches.jsonl',near)
    model_paths=sorted(Path(cfg['teacher']['snapshot_path']).glob('*.safetensors'))+sorted(Path(cfg['teacher']['snapshot_path']).glob('*.json'))
    save(inputs/'model_hashes.json',evidence(model_paths))
    summary={'source_train_questions':len(train),'historical_unique_questions':len(histories),
        'eligible_questions':len(eligible),'mechanism_questions':len(selected),'smoke_questions':len(smoke),
        'split_counts':dict(Counter(r['question_split'] for r in selected)),
        'exclusion_reasons':dict(Counter(r['reason'] for r in exclusions)),
        'reference_variants':dict(Counter(r['variant'] for r in variants)),
        'newness_scope':'Independent of listed historical SAE discovery/tuning questions; not a pretraining contamination guarantee. Author-data TokenSkip student SFT does not change the frozen SAE teacher.',
        'student_training_data':False,'full_S1b_complete':False}
    save(inputs/'audit.json',summary)
    save(inputs/'versions.json',{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','math-verify','sympy')})
    cfg.update(result_root=str(root),code_root=str(root/'code'),parent_intervention_root=str(parent))
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',bindings+[root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
         Path(cfg['gsm8k']['train_arrow']),Path(cfg['gsm8k']['test_arrow']),*sorted(inputs.glob('*'))],
         formal_claim_allowed=False,stage='sae_new_question_input_preparation')
    logging.info('Prepared new SAE mechanism cohort: %s',summary)


def load_frozen(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if CODE.resolve()!=Path(cfg['code_root']).resolve():raise ValueError('Use frozen source snapshot')
    return cfg


def generate(cfg, *, smoke=False, shard=0):
    import torch
    root=Path(cfg['result_root']);out=root/'baseline'/('smoke' if smoke else f'shard_{shard:02d}')
    if not 0<=shard<cfg['generation']['shards']:raise ValueError('Invalid shard')
    if not smoke:verify(root/'baseline/smoke/COMPLETE.json')
    out.mkdir(parents=True,exist_ok=False)
    admission(cfg)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv'],check=True,text=True,capture_output=True)
    save(out/'hardware.json',{'slurm_job_id':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'inventory_csv':hardware.stdout,'torch_device_name':torch.cuda.get_device_name(0),
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))})
    model,tok=teacher_bundle(cfg);torch.cuda.reset_peak_memory_stats()
    source=list(read_jsonl(root/'inputs'/('smoke_questions.jsonl' if smoke else 'questions.jsonl')))
    if not smoke:source=[r for i,r in enumerate(source) if i%cfg['generation']['shards']==shard]
    rows=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for question in source:
            seed=int(canonical_sha256([cfg['generation']['seed'],question['problem_id']])[:8],16)
            result=generate_text(model,tok,[{'role':'user','content':question['prompt']}],cfg['generation'],seed)
            row={**question,**result,**grade_gsm8k_response(result['solution'],question['answer'],timeout_seconds=cfg['grading_timeout_seconds']),
                 'candidate_index':0,'trace_id':question['problem_id']+'__unmodified_0','method':'unmodified'}
            rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
            logging.info('SAE input trace %s split=%s tokens=%s correct=%s',question['problem_id'],question['question_split'],row['generated_tokens'],row['is_correct'])
    audit_cohort(rows,[r['problem_id'] for r in source])
    summary={**summarize_generations(rows),'smoke_only':smoke,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
             'full_S1b_complete':False,'role':'One unmodified candidate per question for mechanism measurements; not the four-candidate distillation pool.'}
    save(out/'summary.json',summary)
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',out/'predictions.jsonl',out/'summary.json',out/'hardware.json'],
         stage='sae_new_question_baseline_generation',formal_claim_allowed=False,smoke_only=smoke)


def merge(cfg):
    root=Path(cfg['result_root']);out=root/'baseline/merged';out.mkdir(exist_ok=False)
    source=list(read_jsonl(root/'inputs/questions.jsonl'));rows=[];markers=[]
    for shard in range(cfg['generation']['shards']):
        directory=root/'baseline'/f'shard_{shard:02d}';verify(directory/'COMPLETE.json');markers.append(directory/'COMPLETE.json')
        subset=list(read_jsonl(directory/'predictions.jsonl'))
        audit_cohort(subset,[r['problem_id'] for i,r in enumerate(source) if i%cfg['generation']['shards']==shard]);rows.extend(subset)
    by_id=audit_cohort(rows,[r['problem_id'] for r in source]);rows=[by_id[r['problem_id']] for r in source]
    if any(r['candidate_index']!=0 or r['method']!='unmodified' for r in rows):raise ValueError('Unexpected trace condition')
    write_jsonl(out/'predictions.jsonl',rows)
    summary={**summarize_generations(rows),'by_split':{split:summarize_generations([r for r in rows if r['question_split']==split]) for split in cfg['split_counts']},
             'unique_questions':len(by_id),'full_S1b_complete':False,'student_distillation_pool':False}
    save(out/'summary.json',summary)
    seal(out/'COMPLETE.json',markers+[root/'protocol/FROZEN.json',out/'predictions.jsonl',out/'summary.json'],formal_claim_allowed=False,
         stage='sae_new_question_baseline_audited_merge')
