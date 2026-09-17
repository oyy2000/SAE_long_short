"""TokenSkip Qwen-3B SFT reproduction from the pinned author dataset.

Reuses the project TRL trainer, completion masking, verifiers, and NCSU admission.
This original-setting reproduction is separate from unified-teacher distillation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import gc
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import time

from .experiment_io import read_json, publish_files_hash_verified
from .factorial import canonical_sha256, file_sha256
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import resolve, save, seal, verify, evidence, admission
from .compression_baselines import TOKEN_SKIP_INSTRUCTION, tokenskip_prompt
from .gsm8k_grading import grade_gsm8k_response
from .verifiers import extract_final_answer

CODE = Path(__file__).resolve().parents[2]
RATIO_SUFFIX = re.compile(r'<\|eot_id\|>(0\.[5-9])<\|eot_id\|>$')


def unpack_author_input(row):
    """Keep author text exact while separating question identity from the ratio."""
    if row['instruction'] != TOKEN_SKIP_INSTRUCTION:
        raise ValueError('Unexpected author instruction')
    match = RATIO_SUFFIX.search(row['input'])
    ratio = float(match.group(1)) if match else 1.0
    question = row['input'][:match.start()] if match else row['input']
    if '<|eot_id|>' in question:
        raise ValueError('Unexpected or malformed ratio suffix')
    prompt = row['instruction'] + '\n' + row['input']
    if tokenskip_prompt(question, ratio) != prompt:
        raise ValueError('Training/inference ratio formatting mismatch')
    return question, ratio, prompt


def evaluation_cells(cfg, model):
    if model not in ('base','author','replica'):
        raise ValueError(model)
    ratios = [1.0] if model == 'base' else cfg['ratios']
    cells=[]
    for ratio in ratios:
        for policy in cfg['evaluation']['cap_policies']:
            if ratio == 1.0 and policy != 'fixed':
                continue  # Same cap and prompt, not an independent repeated run.
            cap=cfg['evaluation']['max_new_tokens']
            if policy == 'scaled':cap=int(cap*ratio)
            cells.append({'ratio':ratio,'cap_policy':policy,'max_new_tokens':cap,
                          'name':f"ratio_{ratio:.1f}__{policy}"})
    return cells


def expected_training_steps(records, cfg):
    """Pinned Transformers 4.48.3 sets the stopping step from floor updates/epoch.

    Its final partial accumulation batch can advance the epoch fraction; record
    the actual epoch separately instead of claiming exactly three full exposures.
    """
    spec=cfg['training']
    batches=math.ceil(records/spec['per_device_train_batch_size'])
    return math.ceil(spec['num_train_epochs']*max(1,batches//spec['gradient_accumulation_steps']))


def prepare(config_path):
    from datasets import Dataset
    from transformers import AutoTokenizer
    import numpy as np
    cfg=read_json(config_path);root=resolve(cfg['result_root'])
    if root.exists():raise FileExistsError(root)
    source=resolve(cfg['author_training_data']);metadata=read_json(resolve(cfg['author_source_metadata']))
    if file_sha256(source)!=metadata['sha256']:raise ValueError('Author training data changed')
    author_rows=json.loads(source.read_text())
    if not isinstance(author_rows,list):raise ValueError('Author training data must be a JSON array')
    if len(author_rows)!=cfg['expected_author_records']:raise ValueError('Author record count changed')
    train=list(Dataset.from_file(cfg['gsm8k']['train_arrow']))
    test=list(Dataset.from_file(cfg['gsm8k']['test_arrow']))
    if len(train)!=7473 or len(test)!=1319:raise ValueError('GSM8K cohort count changed')
    def norm(text):return re.sub(r'\s+',' ',text).strip()
    lookup=defaultdict(list)
    for i,row in enumerate(train):lookup[norm(row['question'])].append(i)
    test_questions={norm(r['question']) for r in test}
    tok=AutoTokenizer.from_pretrained(cfg['student']['snapshot_path'],local_files_only=True)
    records=[];wrong=[];max_length=0
    for index,row in enumerate(author_rows):
        question,ratio,prompt=unpack_author_input(row)
        candidates=lookup.get(norm(question),[])
        if len(candidates)!=1 or norm(question) in test_questions:
            raise ValueError(f'Author question not uniquely in GSM8K train: row {index}: {candidates}')
        qi=candidates[0];gold=extract_final_answer(train[qi]['answer'])
        grade=grade_gsm8k_response(row['output'],gold,timeout_seconds=cfg['grading_timeout_seconds'])
        chat=[{'role':'user','content':prompt},{'role':'assistant','content':row['output']}]
        length=len(tok.apply_chat_template(chat,tokenize=True))
        if length>cfg['training']['max_length']:raise ValueError(f'SFT would truncate author record {index}')
        max_length=max(max_length,length)
        record={'problem_id':f'gsm8k-train-{qi:05d}','author_index':index,'source_index':qi,
                'question':question,'answer':gold,'ratio':ratio,'prompt':prompt,'completion':row['output'],
                'source_output_sha256':canonical_sha256(row['output']),'chat_tokens':length,
                'completion_tokens':len(tok.encode(row['output'],add_special_tokens=False)),
                'source_answer_grade':grade}
        records.append(record)
        if not grade['is_correct']:wrong.append(record)
    if len({r['problem_id'] for r in records})!=len(records):raise ValueError('Duplicate author training questions')
    # Matches the default datasets train_test_split permutation and ceil test-size.
    permutation=np.random.default_rng(cfg['split_seed']).permutation(len(records)).tolist()
    n_dev=math.ceil(cfg['validation_fraction']*len(records));dev_indices=set(permutation[:n_dev])
    train_rows=[records[i] for i in permutation[n_dev:]]
    dev_rows=[records[i] for i in permutation[:n_dev]]
    evaluation=[]
    for i in range(50,1319):
        evaluation.append({'problem_id':f'gsm8k-test-{i:05d}','source_index':i,
            'question':test[i]['question'],'answer':extract_final_answer(test[i]['answer'])})
    inputs=root/'inputs'
    for name,rows in [('train',train_rows),('validation',dev_rows),('evaluation',evaluation),('source_answer_audit',wrong)]:
        write_jsonl(inputs/(name+'.jsonl'),rows)
    smoke=[{'problem_id':f'gsm8k-smoke-{i:05d}','source_index':i,'question':test[i]['question'],
            'answer':extract_final_answer(test[i]['answer'])} for i in range(cfg['smoke']['evaluation_questions'])]
    write_jsonl(inputs/'smoke_evaluation.jsonl',smoke)
    write_jsonl(inputs/'smoke_train.jsonl',train_rows[:cfg['smoke']['training_records']])
    save(inputs/'data_audit.json',{'author_records':len(records),'unique_questions':len(records),
        'train_records':len(train_rows),'validation_records':len(dev_rows),'evaluation_records':len(evaluation),
        'train_validation_overlap':0,'train_test_overlap':0,'ratio_counts':dict(Counter(str(r['ratio']) for r in records)),
        'source_answer_disagreements':len(wrong),'preserve_exact_author_outputs':True,
        'max_chat_tokens':max_length,'token_truncations':0,
        'expected_optimizer_steps':expected_training_steps(len(train_rows),cfg),
        'author_rows_sha256':file_sha256(source)})
    save(inputs/'source_metadata.json',metadata)
    model_hashes={}
    for role in ('student','author_adapter'):
        snapshot=Path(cfg[role]['snapshot_path'])
        paths=sorted(snapshot.glob('*.safetensors'))+sorted(snapshot.glob('*.json'))
        model_hashes[role]=evidence(paths)
    save(inputs/'model_hashes.json',model_hashes)
    state=read_json(Path(cfg['author_adapter']['snapshot_path'])/'trainer_state.json')
    save(inputs/'author_checkpoint_state.json',{k:state.get(k) for k in
         ('global_step','max_steps','epoch','num_train_epochs','best_metric','best_model_checkpoint')})
    save(inputs/'versions.json',{name:importlib.metadata.version(name) for name in
         ('torch','transformers','trl','peft','datasets','math-verify')})
    code=root/'code'
    for name in ('src','scripts','configs'):
        shutil.copytree(CODE/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    cfg.update(result_root=str(root),checkpoint_root=str(resolve(cfg['checkpoint_root'])),code_root=str(code))
    sources=[p for name in ('src','scripts','configs') for p in (code/name).rglob('*') if p.is_file() and not p.is_symlink()]
    seal(root/'protocol/SOURCES.json',sources)
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',[root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
        *sorted(inputs.glob('*')),source,resolve(cfg['author_source_metadata']),
        Path(cfg['gsm8k']['train_arrow']),Path(cfg['gsm8k']['test_arrow'])],formal_claim_allowed=False)
    logging.info('TokenSkip inputs frozen: %d train, %d validation, %d evaluation',len(train_rows),len(dev_rows),len(evaluation))


def load_frozen(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if CODE.resolve()!=Path(cfg['code_root']).resolve():raise ValueError('Use the frozen code snapshot')
    return cfg


def train_student(cfg, *, smoke=False):
    import torch
    from transformers import set_seed
    from .training import run_trl_sft
    from safetensors.torch import load_file
    root=Path(cfg['result_root']);name='smoke' if smoke else 'replica'
    out=root/'training'/name;adapter=Path(cfg['checkpoint_root'])/name
    if (adapter/'TRAIN_COMPLETE.json').exists():verify(adapter/'TRAIN_COMPLETE.json');return
    if adapter.exists() or out.exists():raise FileExistsError('Incomplete training attempt needs explicit recovery')
    admission(cfg);out.mkdir(parents=True)
    train_path=root/'inputs'/('smoke_train.jsonl' if smoke else 'train.jsonl')
    rows=list(read_jsonl(train_path));local=Path(os.environ['TMPDIR'])/'tokenskip'/name
    spec={k:v for k,v in cfg['student'].items() if k!='snapshot_path'}
    spec.update(model_name=cfg['student']['snapshot_path'],tokenizer_kwargs={'local_files_only':True})
    settings={**cfg['training'],'seed':cfg['seed'],'data_seed':cfg['seed'],
        'output_dir':str(adapter),'model_init_kwargs':{'local_files_only':True,'attn_implementation':'sdpa'}}
    if smoke:
        settings.update(max_steps=cfg['smoke']['training_steps'],warmup_ratio=0.,warmup_steps=0,save_strategy='no')
    run={'student':spec,'training':settings,'data':{'train_path':str(train_path),'text_format':'prompt_completion'}}
    save(out/'run_config.json',run)
    os.environ['LBD_RUNTIME_OUTPUT_DIR']=str(local);set_seed(cfg['seed'])
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    trainer=run_trl_sft(run)
    expected=cfg['smoke']['training_steps'] if smoke else expected_training_steps(len(rows),cfg)
    if trainer.state.global_step!=expected:raise ValueError(f'Incomplete SFT: {trainer.state.global_step} != {expected}')
    weights=load_file(str(local/'adapter_model.safetensors'))
    b_values=[v for k,v in weights.items() if 'lora_B' in k]
    if not b_values or not all(torch.isfinite(v).all() for v in weights.values()) or not any(torch.count_nonzero(v) for v in b_values):
        raise ValueError('Missing, nonfinite, or unchanged LoRA adapter')
    metrics={'status':'complete','records':len(rows),'optimizer_steps':trainer.state.global_step,
        'actual_epoch':trainer.state.epoch,'configured_epochs':settings['num_train_epochs'],
        'elapsed_seconds':time.monotonic()-start,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'seed':cfg['seed'],'nonzero_lora_b_tensors':sum(bool(torch.count_nonzero(v)) for v in b_values),
        'log_history':trainer.state.log_history,'smoke_only':smoke}
    save(local/'training_metrics.json',metrics)
    publish_files_hash_verified(local,adapter,('adapter_model.safetensors','adapter_config.json','training_metrics.json'))
    seal(adapter/'TRAIN_COMPLETE.json',[train_path,out/'run_config.json',root/'protocol/FROZEN.json',
        CODE/'src/length_budget_distill/training.py',Path(__file__),CODE/'scripts/slurm/13_2_tokenskip_reproduction.sh',
        adapter/'adapter_model.safetensors',adapter/'adapter_config.json',adapter/'training_metrics.json'],formal_claim_allowed=False)
    seal(out/'COMPLETE.json',[adapter/'TRAIN_COMPLETE.json',adapter/'training_metrics.json'],stage='tokenskip_sft',smoke_only=smoke)
    logging.info('TokenSkip SFT complete: %s steps=%d actual_epoch=%s',name,expected,trainer.state.epoch)


def eval_bundle(cfg, model):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(cfg['student']['snapshot_path'],local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    base=AutoModelForCausalLM.from_pretrained(cfg['student']['snapshot_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if model!='base':
        if model=='author':adapter=Path(cfg['author_adapter']['snapshot_path'])
        else:
            adapter=Path(cfg['checkpoint_root'])/model
            verify(adapter/'TRAIN_COMPLETE.json')
        base=PeftModel.from_pretrained(base,str(adapter)).merge_and_unload().eval()
    return base,tokenizer


def generate_batch(cfg,model,tok,questions,cell):
    import torch
    from .student_evaluation import generate_native_greedy_batch
    messages=[[{'role':'system','content':cfg['evaluation']['system_prompt']},
               {'role':'user','content':tokenskip_prompt(row['question'],cell['ratio'])}] for row in questions]
    generated,elapsed=generate_native_greedy_batch({'model':model,'tokenizer':tok,'torch':torch},messages,
                                                  max_new_tokens=cell['max_new_tokens'])
    rows=[]
    for question,result in zip(questions,generated):
        tokens=result['token_ids'];text=result['prediction_text']
        grade=grade_gsm8k_response(text,question['answer'],timeout_seconds=cfg['grading_timeout_seconds'])
        rows.append({**question,**grade,'prediction_text':text,'token_ids':tokens,'output_tokens':len(tokens),
            'hit_max_new_tokens':result['hit_max_new_tokens'],
            'batch_elapsed_seconds':elapsed,'batch_size':len(questions),'cell':cell})
    return rows,elapsed


def evaluate_student(cfg,model_name, *, smoke=False):
    import torch
    from transformers import set_seed
    root=Path(cfg['result_root']);label=model_name+('_smoke' if smoke else '')
    out=root/'evaluation'/label
    if (out/'COMPLETE.json').exists():verify(out/'COMPLETE.json');return
    admission(cfg);set_seed(cfg['seed'])
    model,tok=eval_bundle(cfg,'smoke' if smoke else model_name)
    out.mkdir(parents=True,exist_ok=True)
    questions=list(read_jsonl(root/'inputs'/('smoke_evaluation.jsonl' if smoke else 'evaluation.jsonl')))
    cells=evaluation_cells(cfg,'replica' if smoke else model_name)
    if smoke:cells=[c for c in cells if c['cap_policy']=='fixed' and c['ratio'] in (.5,1.)]
    complete=[]
    for cell in cells:
        dest=out/cell['name'];marker=dest/'COMPLETE.json'
        if marker.exists():verify(marker);complete.append(marker);continue
        if dest.exists():raise FileExistsError('Partial evaluation cell needs explicit recovery')
        dest.mkdir(parents=True)
        predictions=[];batch_times=[]
        with (dest/'predictions.jsonl').open('x') as handle:
            for start in range(0,len(questions),cfg['evaluation']['batch_size']):
                rows,elapsed=generate_batch(cfg,model,tok,questions[start:start+cfg['evaluation']['batch_size']],cell)
                batch_times.append(elapsed);predictions.extend(rows)
                for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                handle.flush()
                logging.info('%s %s predictions=%d/%d',model_name,cell['name'],len(predictions),len(questions))
        if len(predictions)!=len(questions) or {r['problem_id'] for r in predictions}!={r['problem_id'] for r in questions}:
            raise ValueError('Prediction duplicate/missing audit failed')
        metrics={'n':len(predictions),'accuracy':sum(r['is_correct'] for r in predictions)/len(predictions),
            'mean_output_tokens':sum(r['output_tokens'] for r in predictions)/len(predictions),
            'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in predictions)/len(predictions),
            'generation_seconds':sum(batch_times),'generation_seconds_per_problem':sum(batch_times)/len(predictions),
            'cell':cell,'model':model_name,'smoke_only':smoke,'gpu_name':torch.cuda.get_device_name(0)}
        save(dest/'metrics.json',metrics)
        bindings=[dest/'predictions.jsonl',dest/'metrics.json',root/'protocol/FROZEN.json']
        if model_name=='replica' or smoke:
            bindings.append(Path(cfg['checkpoint_root'])/('smoke' if smoke else 'replica')/'TRAIN_COMPLETE.json')
        seal(marker,bindings,stage='tokenskip_evaluation_cell',formal_claim_allowed=False)
        complete.append(marker)
    seal(out/'COMPLETE.json',complete,stage='tokenskip_evaluation',model=model_name,cells=len(cells),smoke_only=smoke)
    del model;gc.collect();torch.cuda.empty_cache()
