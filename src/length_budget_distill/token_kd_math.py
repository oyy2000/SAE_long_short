"""Prepare a reviewed common-support MATH matrix for matched 3B SFT/KD."""
from pathlib import Path
import json
import shutil

from .records import read_jsonl,write_jsonl
from .experiment_io import read_json
from .ncsu_reproduction import save,seal,verify
from .factorial import canonical_sha256,file_sha256
from .completion_supervision import validate_encoded_record
from .token_kd_baselines import paired_ids


def prepare(cfg):
    from transformers import AutoTokenizer
    root=Path(cfg['result_root']);data=Path(cfg['source_student_data_root'])
    cohort=Path(cfg['reviewed_cohort_root']);pilot=Path(cfg['source_pilot_root'])
    for marker in [root/'protocol/FROZEN.json',root/'protocol/SOURCES.json',
                   data/'protocol/FROZEN.json',data/'protocol/SOURCES.json',cohort/'COMPLETE.json',
                   pilot/'sft/protocol/FROZEN.json']:
        verify(marker)
    audit=read_json(data/'inputs/common_support.json')
    ids=audit['common_ids']
    if len(ids)!=cfg['train_questions'] or len(ids)!=len(set(ids)) or len(ids)<cfg['minimum_common_questions']:
        raise ValueError('Changed or insufficient eight-method MATH common support')
    bindings=[data/'protocol/FROZEN.json',cohort/'COMPLETE.json',data/'inputs/common_support.json']
    methods={};targets={}
    for method in cfg['baseline_matrix']:
        path=data/'encoded/qwen3b_student'/f'{method}.jsonl';text_path=data/'text'/f'{method}.jsonl'
        rows=list(read_jsonl(path));text=list(read_jsonl(text_path))
        paired_ids(rows,ids);paired_ids(text,ids)
        for row in rows:validate_encoded_record(row,max_length=cfg['training']['max_length'])
        write_jsonl(root/'sft/encoded/qwen3b_student'/f'{method}.jsonl',rows)
        write_jsonl(root/'sft/text'/f'{method}.jsonl',text)
        methods[method]={'questions':len(rows),'maximum_input_tokens':max(len(r['input_ids']) for r in rows),
                         'target_tokens_per_epoch':sum(sum(t!=-100 for t in r['labels'][1:]) for r in rows),
                         'input_sha256':file_sha256(path),'text_sha256':file_sha256(text_path)}
        targets[method]=methods[method]['target_tokens_per_epoch'];bindings.extend([path,text_path])
    if len(methods)!=8:raise ValueError('Eight reviewed sources required')
    if cfg['smoke_method']!=max(methods,key=lambda m:methods[m]['maximum_input_tokens']):
        raise ValueError('Longest input baseline changed; revise smoke method before launch')
    teacher=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    student=AutoTokenizer.from_pretrained(cfg['students']['qwen3b_student']['snapshot_path'],local_files_only=True)
    vocab=student.get_vocab()
    if teacher.get_vocab()!=vocab or set(vocab.values())!=set(range(len(vocab))) or teacher.backend_tokenizer.to_str()!=student.backend_tokenizer.to_str():
        raise ValueError('Teacher/student token IDs or tokenizer rules differ')
    save(root/'inputs/vocabulary.json',{'valid_vocabulary_size':len(vocab),'mapping_sha256':canonical_sha256(vocab),
         'projection':'Renormalize teacher and student over identical valid token IDs'})
    shutil.copyfile(pilot/'inputs/model_hashes.json',root/'inputs/model_hashes.json')
    shutil.copyfile(data/'inputs/runtime_versions.json',root/'inputs/runtime_versions.json')
    roles={}
    for name in cfg['evaluation_cohorts']:
        path=cohort/'evaluation'/f'{name}.jsonl';rows=list(read_jsonl(path));paired_ids(rows)
        if len(rows)!=cfg['evaluation_questions'][name] or any(r['question_role']!='locked_evaluation' for r in rows):
            raise ValueError('Changed locked evaluation cohort: '+name)
        if set(r['problem_id'] for r in rows)&set(ids):raise ValueError('MATH train/evaluation overlap')
        write_jsonl(root/'inputs'/f'{name}.jsonl',rows)
        roles[name]=len(rows);bindings.append(path)
    dev=list(read_jsonl(cohort/'cohorts/development.jsonl'));paired_ids(dev)
    if len(dev)!=cfg['development_questions'] or set(r['problem_id'] for r in dev)&set(ids):
        raise ValueError('MATH development role changed')
    write_jsonl(root/'inputs/development.jsonl',dev);bindings.append(cohort/'cohorts/development.jsonl')
    save(root/'inputs/data_audit.json',{'train_questions':len(ids),'common_support_sha256':canonical_sha256(ids),
        'training_methods':methods,'target_tokens_per_epoch':targets,'evaluation_questions':roles,
        'B7':'historical Answer-associated SAE teacher traces','equal_examples_and_order':True,
        'equal_target_tokens_across_methods':False,'previously_observed_cohorts':True,
        'Math500_not_independent_OOD_relative_to_MATH_training':True})
    seal(root/'inputs/COMPLETE.json',[root/'protocol/FROZEN.json',*bindings,
        *sorted((root/'inputs').glob('*.json*')),*sorted((root/'sft/text').glob('*.jsonl')),
        *sorted((root/'sft/encoded/qwen3b_student').glob('*.jsonl'))],formal_claim_allowed=False)
