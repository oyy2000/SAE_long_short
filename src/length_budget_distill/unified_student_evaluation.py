"""Shared multi-benchmark student decoding, with a synthetic adapter smoke."""
from pathlib import Path
import gc
import importlib.metadata
import logging

from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .records import write_jsonl
from .ncsu_reproduction import admission, save, seal, verify
from .baseline_reproduction import record_hardware
from .student_evaluation import load_student_for_evaluation, generate_native_greedy_batch
from .student_prompts import build_unified_evaluation_prompt
from .unified_math_candidates import grade_candidate

CODE=Path(__file__).resolve().parents[2]


def grade_student_prediction(text, source, cfg):
    """Dispatch by the frozen dataset policy; never infer a gold from output."""
    method=cfg['grading_methods'][source['dataset']]
    return grade_candidate({'grading_method':method,'grading':cfg['grading']},text,source)


def generate_student_batch(bundle, questions, cfg, *, ratio, max_new_tokens):
    """Use the same prompt, token accounting and grading for every student arm."""
    if not questions or len({q['problem_id'] for q in questions})!=len(questions):
        raise ValueError('Student batch is empty or has duplicate problem IDs')
    messages=[[{'role':'user','content':build_unified_evaluation_prompt(q,cfg,ratio=ratio)}] for q in questions]
    raw,seconds=generate_native_greedy_batch(bundle,messages,max_new_tokens=max_new_tokens,
                                            repetition_penalty=cfg['repetition_penalty'])
    rows=[]
    for source,result,prompt in zip(questions,raw,messages):
        rows.append({**result,'problem_id':source['problem_id'],'dataset':source['dataset'],
                     'question_role':source['question_role'],'source_record_sha256':canonical_sha256(source),
                     'ratio':ratio,'max_new_tokens':max_new_tokens,'messages_sha256':canonical_sha256(prompt),
                     'grade':grade_student_prediction(result['prediction_text'],source,cfg),
                     'grading_method':cfg['grading_methods'][source['dataset']]})
    return rows,seconds


def audit_student_prediction(row, source, cfg, tokenizer):
    """Reconstruct exactly what was prompted, generated, stopped and scored."""
    if row['problem_id']!=source['problem_id'] or row['source_record_sha256']!=canonical_sha256(source):
        raise ValueError('Student prediction source identity changed')
    messages=[{'role':'user','content':build_unified_evaluation_prompt(source,cfg,ratio=row['ratio'])}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    if row['messages_sha256']!=canonical_sha256(messages) or row['native_prompt_token_ids']!=ids or row['prompt_tokens']!=len(ids):
        raise ValueError('Student native prompt changed')
    sampled=row['sampled_token_ids'];eos=tokenizer.eos_token_id
    ended=bool(sampled and sampled[-1]==eos)
    if not sampled or eos in sampled[:-1] or len(sampled)>row['max_new_tokens']:
        raise ValueError('Invalid student sampled-token boundary')
    body=sampled[:-1] if ended else sampled
    if row['token_ids']!=body or row['output_tokens']!=len(body) or row['hit_max_new_tokens']==ended:
        raise ValueError('Student EOS/token accounting changed')
    if not ended and len(sampled)!=row['max_new_tokens']:
        raise ValueError('Student stopped early without its EOS')
    if row['prediction_text']!=tokenizer.decode(body,skip_special_tokens=True).strip():
        raise ValueError('Student text differs from sampled tokens')
    if row['grading_method']!=cfg['grading_methods'][source['dataset']] or row['grade']!=grade_student_prediction(row['prediction_text'],source,cfg):
        raise ValueError('Student grading changed')


def smoke(config_path, name):
    import torch
    cfg=read_json(config_path);root=Path(cfg['result_root']);launch=Path(cfg['launch_root'])
    verify(launch/'FROZEN.json')
    if CODE!=Path(cfg['code_root']) or cfg['evidence_class']!='synthetic_evaluation_interface':
        raise ValueError('Use the frozen synthetic evaluation interface')
    if name not in cfg['students']:raise ValueError('Unregistered student')
    questions=cfg['questions']
    if any(q['question_role']!='synthetic_interface_only' for q in questions):
        raise ValueError('This smoke must not inspect real development or evaluation questions')
    interface=Path(cfg['sft_interface_root'])
    markers=[launch/'FROZEN.json',interface/'protocol/FROZEN.json',interface/'protocol/SOURCES.json',
             interface/'training'/name/'COMPLETE.json']
    for marker in markers:verify(marker)
    parent=read_json(interface/'protocol/frozen_config.json')
    if cfg['students'][name]!=parent['students'][name]:raise ValueError('Student differs from the trained adapter base')
    adapter=Path(parent['checkpoint_root'])/name
    marker=adapter/'TRAIN_COMPLETE.json';doc=verify(marker);markers.append(marker)
    if not doc.get('synthetic_only') or doc.get('formal_training_complete'):
        raise ValueError('Smoke requires its synthetic adapter, not a formal model')
    for path,digest in read_json(interface/'inputs/model_hashes.json')[name].items():
        if file_sha256(path)!=digest:raise ValueError('Student model input changed')
    for filename in ('adapter_model.safetensors','adapter_config.json'):
        if str((adapter/filename).resolve()) not in doc['hashes']:raise ValueError('Unbound LoRA file')
    out=root/name;out.mkdir(parents=True,exist_ok=False);admission(cfg);record_hardware(out)
    save(out/'versions.json',{k:importlib.metadata.version(k) for k in ('torch','transformers','peft','math-verify')})
    all_rows=[];batches=[];loaded=[]
    for variant in ('base','synthetic_lora'):
        spec=cfg['students'][name]
        bundle=load_student_for_evaluation({'model_name':spec['snapshot_path'],'torch_dtype':cfg['torch_dtype'],
                                            'attn_implementation':cfg['attention_implementation']},
                                           adapter_path=str(adapter) if variant=='synthetic_lora' else None)
        model=bundle['model'];tok=bundle['tokenizer'];torch.cuda.reset_peak_memory_stats()
        active=bool(getattr(model,'peft_config',None))
        if active!=(variant=='synthetic_lora'):raise ValueError('Adapter attachment differs from registered variant')
        loaded.append({'variant':variant,'model_class':type(model).__name__,'adapter_attached':active,
                       'attention_implementation':model.config._attn_implementation,
                       'model_default_repetition_penalty':model.generation_config.repetition_penalty,
                       'effective_repetition_penalty':cfg['repetition_penalty']})
        for ratio in cfg['smoke_ratios']:
            for start in range(0,len(questions),cfg['batch_size']):
                sources=questions[start:start+cfg['batch_size']]
                rows,seconds=generate_student_batch(bundle,sources,cfg,ratio=ratio,max_new_tokens=cfg['max_new_tokens'])
                batch_id=f'{variant}/{ratio}/{start}'
                for row,source in zip(rows,sources):
                    audit_student_prediction(row,source,cfg,tok)
                    row.update(student=name,variant=variant,batch_id=batch_id)
                all_rows.extend(rows);batches.append({'batch_id':batch_id,'seconds':seconds,'questions':len(sources)})
                logging.info('Student evaluation smoke %s %s ratio=%s records=%d',name,variant,ratio,len(all_rows))
        loaded[-1]['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        del model,bundle;gc.collect();torch.cuda.empty_cache()
    expected={(variant,ratio,q['problem_id']) for variant in ('base','synthetic_lora')
              for ratio in cfg['smoke_ratios'] for q in questions}
    if len(all_rows)!=len(expected) or {(r['variant'],r['ratio'],r['problem_id']) for r in all_rows}!=expected:
        raise ValueError('Incomplete student/ratio smoke grid')
    write_jsonl(out/'predictions.jsonl',all_rows);write_jsonl(out/'batches.jsonl',batches)
    save(out/'summary.json',{'student':name,'synthetic_questions':len(questions),'predictions':len(all_rows),
                            'loaded_models':loaded,'generation_seconds':sum(b['seconds'] for b in batches),
                            'cap_hits':sum(r['hit_max_new_tokens'] for r in all_rows),
                            'real_benchmark_predictions':0,'formal_student_evaluation_complete':False,
                            'scope':'Adapter loading, common decoding controls, ratio prompts, five grading paths and exact native tokens. Synthetic answer scores are not student-quality evidence.'})
    seal(out/'COMPLETE.json',[Path(config_path),*markers,*sorted(out.glob('*'))],
         stage='synthetic_student_evaluation_interface',formal_student_evaluation_complete=False)
