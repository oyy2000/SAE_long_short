"""Native prompt/context checks without generating or inspecting test answers."""
from pathlib import Path
import json
import logging

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify
from .student_prompts import build_unified_evaluation_prompt, is_choice_question

CODE=Path(__file__).resolve().parents[2]


def compare_prompt_records(previous, current):
    """Reject changed cohorts and distinguish approved task-format corrections."""
    def index(rows):
        result={(r['problem_id'],r['ratio']):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate evaluation prompt key')
        return result
    before,after=index(previous),index(current)
    if set(before)!=set(after):raise ValueError('Evaluation prompt cohort or ratio grid changed')
    return [{'problem_id':pid,'ratio':ratio,'previous':before[(pid,ratio)],'current':after[(pid,ratio)]}
            for pid,ratio in after if before[(pid,ratio)]!=after[(pid,ratio)]]


def run(config_path):
    from transformers import AutoConfig, AutoTokenizer
    cfg=read_json(config_path);out=Path(cfg['result_root']);cohorts=Path(cfg['cohort_root'])
    if out.exists():raise FileExistsError(out)
    marker=Path(cfg['launch_root'])/'FROZEN.json';verify(marker);verify(cohorts/'COMPLETE.json')
    questions={name:list(read_jsonl(cohorts/'evaluation'/(name+'.jsonl'))) for name in cfg['expected_counts']}
    for name,rows in questions.items():
        if len(rows)!=cfg['expected_counts'][name] or len({r['problem_id'] for r in rows})!=len(rows):
            raise ValueError('Evaluation cohort count/uniqueness differs')
    if sorted(r['source_index'] for r in questions['gsm8k'])!=list(range(50,1319)):
        raise ValueError('Locked GSM8K evaluation changed')
    if any(not r.get('parent_problem_ids') for r in questions['gsm8k_hard']):
        raise ValueError('GSM8K-Hard parent grouping is absent')
    out.mkdir(parents=True);summaries={};bindings=[Path(config_path),marker,cohorts/'COMPLETE.json']
    changes=[];parent=Path(cfg['prompt_revision_parent']) if cfg.get('prompt_revision_parent') else None
    if parent is not None:
        verify(parent/'COMPLETE.json');bindings.append(parent/'COMPLETE.json')
    interface=Path(cfg['sft_interface_root']);verify(interface/'protocol/FROZEN.json');bindings.append(interface/'protocol/FROZEN.json')
    models=read_json(interface/'protocol/frozen_config.json')['students']
    hashes=read_json(interface/'inputs/model_hashes.json')
    for name,spec in models.items():
        for path,digest in hashes[name].items():
            if Path(path).suffix not in ('.safetensors','.bin') and file_sha256(path)!=digest:
                raise ValueError('Changed tokenizer/model configuration')
        tok=AutoTokenizer.from_pretrained(spec['snapshot_path'],local_files_only=True)
        context=AutoConfig.from_pretrained(spec['snapshot_path'],local_files_only=True).max_position_embeddings
        summaries[name]={}
        for dataset,rows in questions.items():
            lengths=[];records=[]
            for question in rows:
                for ratio in [None,*cfg['tokenskip_ratios']]:
                    prompt=build_unified_evaluation_prompt(question,cfg,ratio=ratio)
                    rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
                    ids=tok.encode(rendered,add_special_tokens=False);lengths.append(len(ids))
                    if len(ids)+max(cfg['proposed_output_caps'])>context:
                        raise ValueError('Evaluation input and proposed output budget exceed context')
                    records.append({'problem_id':question['problem_id'],'ratio':ratio,'prompt_tokens':len(ids),
                        'native_prompt_token_sha256':canonical_sha256(ids),'prompt_text_sha256':canonical_sha256(prompt),
                        'choice_letter_instruction':is_choice_question(question)})
            if parent is not None:
                previous=list(read_jsonl(parent/(name+'__'+dataset+'.jsonl')))
                changes.extend(dict(row,student=name,dataset=dataset) for row in compare_prompt_records(previous,records))
            write_jsonl(out/(name+'__'+dataset+'.jsonl'),records)
            summaries[name][dataset]={'questions':len(rows),'prompt_variants':len(records),'maximum_prompt_tokens':max(lengths),
                'minimum_prompt_tokens':min(lengths),'model_context_tokens':context,'largest_proposed_output_cap':max(cfg['proposed_output_caps']),
                'context_overflows':0,'choice_questions':sum(is_choice_question(r) for r in rows)}
            logging.info('Evaluation input preflight %s %s: %s',name,dataset,summaries[name][dataset])
    bindings.extend(cohorts/'evaluation'/(name+'.jsonl') for name in questions)
    if parent is not None:
        write_jsonl(out/'changed_prompts.jsonl',changes)
        if set(r['problem_id'] for r in changes)!=set(cfg['expected_changed_prompt_ids']):
            raise ValueError('Unexpected evaluation prompt changes')
        if len(changes)!=cfg['expected_changed_prompt_records']:
            raise ValueError('Unexpected changed prompt-variant count')
    save(out/'summary.json',{'by_student_dataset':summaries,'test_generations':0,'evaluation_protocol_frozen':False,
        'changed_prompt_records':len(changes),'changed_prompt_ids':sorted({r['problem_id'] for r in changes}),
        'scope':'Prompt format and context capacity only. No test answers are generated or scored; proposed output caps still require the registered development quality checks.'})
    seal(out/'COMPLETE.json',bindings+sorted(out.glob('*')),evaluation_complete=False,stage='student_evaluation_input_preflight')
