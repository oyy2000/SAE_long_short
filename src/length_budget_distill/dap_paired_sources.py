"""Recover released DAP long/short correspondences and verified MATH inputs.

Original upstream row identities, complete assistant traces and failed matching
attempts are preserved. No released solution-only field replaces a full trace.
"""
from collections import Counter,defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re

from .experiment_io import read_json
from .records import read_jsonl,write_jsonl
from .factorial import canonical_sha256,file_sha256
from .ncsu_reproduction import save,seal,verify
from .baseline_data_preflight import normalize_question

CODE=Path(__file__).resolve().parents[2]


def question_key(text):
    # Only the explicit upstream response-format instruction is removed.
    text=re.sub(r'^\s*Return your final response within \\+boxed\{\}\.\s*','',text,flags=re.I)
    return normalize_question(text)


def conversation_pair(row,kind):
    key,role,text=('messages','role','content') if kind=='smallthoughts' else ('conversations','from','value')
    messages=row[key]
    if len(messages)!=2 or messages[0][role]!='user' or messages[1][role]!='assistant':
        raise ValueError('Unexpected conversation structure')
    if kind=='smallthoughts' and messages[0][text]!=row['problem']:
        raise ValueError('Problem differs from stored user message')
    return messages[0][text],messages[1][text]


def prepare(config_path):
    cfg=read_json(config_path);out=Path(cfg['result_root'])
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen source-audit snapshot')
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    verify(Path(cfg['math_review_root'])/'COMPLETE.json')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    if not cfg.get('staged_root'):
        os.environ['HF_HUB_OFFLINE']='0';os.environ['HF_DATASETS_OFFLINE']='0'
        from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    metadata={name:read_json(spec['metadata_path']) for name,spec in cfg['datasets'].items()}
    targets=[]
    for name,info in metadata.items():
        spec=cfg['datasets'][name]
        if info['sha']!=spec['revision']:raise ValueError('Metadata revision mismatch')
        for item in info['siblings']:
            if item['rfilename'].startswith('data/') and item['rfilename'].endswith('.parquet'):
                targets.append((name,spec,item))
    def download(target):
        name,spec,item=target
        if cfg.get('staged_root'):
            path=Path(cfg['staged_root'])/name/item['rfilename']
        else:
            path=Path(hf_hub_download(repo_id=spec['repo_id'],repo_type='dataset',revision=spec['revision'],
                filename=item['rfilename'],cache_dir=cfg['cache_root']))
        if file_sha256(path)!=item['lfs']['sha256']:raise ValueError('Upstream LFS hash mismatch')
        logging.info('Verified source shard %s %s',name,item['rfilename'])
        return {'dataset':name,'path':str(path),'filename':item['rfilename'],'sha256':item['lfs']['sha256'],
                'revision':spec['revision'],'rows':pq.ParquetFile(path).metadata.num_rows}
    with ThreadPoolExecutor(max_workers=cfg['download_workers']) as pool:files=list(pool.map(download,targets))
    save(out/'source_files.json',{'files':files})
    short_index=defaultdict(list);counts=Counter();structure_failures=[]
    for spec in files:
        if spec['dataset']!='smallthoughts':continue
        offset=0
        for batch in pq.ParquetFile(spec['path']).iter_batches(batch_size=512):
            for local,row in enumerate(batch.to_pylist()):
                try:question,answer=conversation_pair(row,'smallthoughts')
                except ValueError as error:
                    structure_failures.append({'dataset':'smallthoughts','file':spec['filename'],'row_index':offset+local,'reason':str(error)});continue
                split='test' if '/test-' in spec['filename'] else 'train';counts['short_'+split]+=1
                short_index[question_key(question)].append({'file':spec['filename'],'row_index':offset+local,'split':split,
                    'question':question,'response':answer,'response_sha256':canonical_sha256(answer),
                    'solution_field_characters':len(row['solution']),'system_prompt':row['system_prompt']})
            offset+=batch.num_rows
    math_rows=list(read_jsonl(Path(cfg['math_review_root'])/'math_train_eligible_for_split.jsonl'))
    math_index=defaultdict(list)
    for row in math_rows:math_index[question_key(row['question'])].append(row)
    candidates=[];pair_counts=Counter();seen_long=Counter();paired_short=set();pairs=[]
    for spec in files:
        if spec['dataset']!='openthoughts':continue
        offset=0
        for batch in pq.ParquetFile(spec['path']).iter_batches(batch_size=256):
            for local,row in enumerate(batch.to_pylist()):
                try:question,long=conversation_pair(row,'openthoughts')
                except ValueError as error:
                    structure_failures.append({'dataset':'openthoughts','file':spec['filename'],'row_index':offset+local,'reason':str(error)});continue
                counts['long_train']+=1;key=question_key(question);seen_long[key]+=1
                for short in short_index.get(key,()):
                    sid=(short['file'],short['row_index']);paired_short.add(sid);pair_counts[short['split']]+=1
                    record={'long_file':spec['filename'],'long_row_index':offset+local,'short_file':short['file'],
                        'short_row_index':short['row_index'],'short_split':short['split'],'question_key_sha256':canonical_sha256(key),
                        'long_characters':len(long),'short_full_response_characters':len(short['response']),
                        'short_solution_field_characters':short['solution_field_characters'],
                        'long_response_sha256':canonical_sha256(long),'short_response_sha256':short['response_sha256']}
                    pairs.append(record)
                    if short['split']=='train' and len(math_index.get(key,()))==1:
                        candidates.append({**record,'math_record':math_index[key][0],'question':question,
                            'long_response':long,'author_short_response':short['response'],
                            'long_system_prompt':row['system'],'short_system_prompt':short['system_prompt']})
            offset+=batch.num_rows
    # Require unique upstream question support for the small executable check.
    multiplicity=Counter(r['math_record']['problem_id'] for r in candidates)
    unique=[r for r in candidates if multiplicity[r['math_record']['problem_id']]==1]
    unique.sort(key=lambda r:canonical_sha256([cfg['selection_seed'],r['math_record']['problem_id']]))
    from transformers import AutoTokenizer
    from .typed_math_grading import grade_typed_response
    grading=read_json(cfg['grading_config'])
    tok=AutoTokenizer.from_pretrained(cfg['tokenizer_snapshot'],local_files_only=True)
    selected=[];attempts=[]
    for candidate in unique[:cfg['maximum_candidate_checks']]:
        math=candidate['math_record'];long=candidate['long_response'];short=candidate['author_short_response']
        long_tokens=len(tok.encode(long,add_special_tokens=False));short_tokens=len(tok.encode(short,add_special_tokens=False))
        long_grade=grade_typed_response(long,math,grading);short_grade=grade_typed_response(short,math,grading)
        eligible=bool(long_grade['is_correct'] and cfg['minimum_long_tokens']<=long_tokens<=cfg['maximum_long_tokens'])
        record={**candidate,'problem_id':math['problem_id'],'long_tokens':long_tokens,'author_short_tokens':short_tokens,
            'long_grade':long_grade,'author_short_grade':short_grade,'eligible_long_input':eligible,
            'role':'Reserved development source for long-trace DAP method checks; exclude from later student SFT pools.'}
        attempts.append(record)
        if eligible:selected.append(record)
        if len(selected)==cfg['target_questions']:break
    write_jsonl(out/'paired_row_manifest.jsonl',pairs);write_jsonl(out/'matching_structure_failures.jsonl',structure_failures)
    write_jsonl(out/'candidate_checks.jsonl',attempts);write_jsonl(out/'selected_sources.jsonl',selected)
    write_jsonl(out/'reserved_math_development_ids.jsonl',[{'problem_id':r['problem_id'],'role':r['role']} for r in selected])
    summary={'status':'paired_source_audit_complete','source_counts':dict(counts),'paired_rows_by_short_split':dict(pair_counts),
        'matched_unique_short_rows':len(paired_short),'unmatched_short_rows':sum(len(x) for x in short_index.values())-len(paired_short),
        'duplicate_long_question_keys':sum(v>1 for v in seen_long.values()),'math_exact_match_candidates':len(candidates),
        'unique_math_candidates':len(unique),'checked_candidates':len(attempts),'selected_questions':len(selected),
        'target_questions':cfg['target_questions'],'ready_for_rewriting':len(selected)==cfg['target_questions'],
        'known_paper_25k_sample_ids_recovered':False,'structure_failures':len(structure_failures),
        'claim_boundary':cfg['claim_boundary'],'formal_claim_allowed':False}
    save(out/'summary.json',summary)
    report=['# DAP released-data correspondence audit','',json.dumps(summary,ensure_ascii=False,indent=2),
        '', 'The complete assistant message is the short trace. The separate solution field is not substituted for it.',
        'Only the documented response-format prefix and case/whitespace/Unicode normalization are removed for question matching. No approximate match is accepted.',
        'MATH references are independently reviewed source answers. Short correctness is measured but is not an eligibility condition. All selected MATH IDs are reserved as development data.',
        'Recovering corresponding rows in current public releases does not recover the paper’s original 25K IDs or establish a student result.','']
    (out/'report.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json',[Path(config_path),Path(cfg['launch_root'])/'FROZEN.json',Path(cfg['math_review_root'])/'COMPLETE.json',
         Path(cfg['grading_config'])]+[Path(s['metadata_path']) for s in cfg['datasets'].values()]+
         [Path(s['path']) for s in files]+[p for p in out.rglob('*') if p.is_file()],stage='dap_released_pair_source_audit',formal_claim_allowed=False)
    logging.info('DAP source audit: %s',summary)
