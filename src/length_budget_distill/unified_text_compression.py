"""DAP and ratio-conditioned TokenSkip on the actual common-teacher pool."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import time

from .experiment_io import read_json
from .records import read_jsonl,write_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import verify,save,seal,evidence,admission,teacher_bundle
from .baseline_reproduction import record_hardware
from .ncsu_multi_answer import unique_correct
from .unified_math_candidates import audit_candidate_grid, grade_candidate, require_same_grading_method
from .sae_norm_intervention import generate_condition_raw
from .compression_baselines import (balanced_ratio_assignment,dap_messages,dap_structure,
                                   compress_trace,tokenskip_prompt,tokenskip_completion)

CODE=Path(__file__).resolve().parents[2]


def stage_tokenskip_assets(cfg,scratch):
    """Copy pinned tokenizer data into the designated job scratch before offline use."""
    spec=cfg['tokenskip']['offline_assets'];cache=Path(cfg['runtime']['auxiliary_cache_root'])
    tiktoken=spec['tiktoken'];key=hashlib.sha1(tiktoken['url'].encode()).hexdigest()
    sources=[(cache/'tiktoken'/key,scratch/'tiktoken'/key,tiktoken['sha256'])]
    for name,digest in spec['nltk_punkt_english_sha256'].items():
        sources.append((cache/'nltk/tokenizers/punkt_tab/english'/name,
                        scratch/'nltk/tokenizers/punkt_tab/english'/name,digest))
    for source,dest,digest in sources:
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('Pinned offline tokenizer asset missing or changed: '+str(source))
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
            raise ValueError('Staged tokenizer asset hash differs: '+str(dest))
    return {'tiktoken_cache_key':key,'tiktoken_sha256':tiktoken['sha256'],
            'nltk_punkt_english_sha256':spec['nltk_punkt_english_sha256'],
            'scratch_root':str(scratch),'source_cache_root':str(cache)}


def build_sources(questions,rows,ratios,seed):
    """Freeze ratios before correct-source filtering; preserve candidate identity."""
    assignment=balanced_ratio_assignment([q['problem_id'] for q in questions],ratios,seed)
    groups=defaultdict(list)
    for row in rows:
        if row['condition']=='B1':groups[row['problem_id']].append(row)
    dap,skip,missing=[],[],[]
    for question in questions:
        pid=question['problem_id'];eligible=unique_correct(groups[pid])
        if not eligible:missing.append(pid);continue
        def item(row):
            return {'problem_id':pid,'source_candidate_index':row['candidate_index'],
                'source_trace_key':f'{pid}::candidate_{row["candidate_index"]}',
                'question_record':question,'source':row,'source_response_sha256':canonical_sha256(row['response'])}
        dap.extend(item(row) for row in sorted(eligible,key=lambda r:r['candidate_index']))
        skip.append({**item(eligible[0]),'assigned_ratio':assignment[pid]})
    return dap,skip,assignment,missing


def prepare(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root']);parent=Path(cfg['candidate_root'])
    if root.exists():raise FileExistsError(root)
    if cfg['compression_mode'] not in ('all_ratios','assigned_only'):raise ValueError('Unknown TokenSkip compression mode')
    if cfg['dap_generation']['repetition_penalty']!=1. or cfg['dap_generation']['add_special_tokens']:
        raise ValueError('DAP decoder requires unit repetition penalty and an already rendered chat template')
    markers=[parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json',parent/'selection'/cfg['source_cohort']/'COMPLETE.json']
    for marker in markers:verify(marker)
    pcfg=read_json(parent/'protocol/frozen_config.json')
    if 'grading_method' in cfg:require_same_grading_method(cfg,pcfg)
    if cfg['source_cohort'] not in pcfg['allowed_generation_cohorts']:raise ValueError('Source cohort was not authorized')
    questions=list(read_jsonl(parent/'inputs'/f'{cfg["source_cohort"]}.jsonl'))
    rows=list(read_jsonl(parent/'selection'/cfg['source_cohort']/'predictions.jsonl'))
    if len(questions)!=cfg['expected_questions']:raise ValueError('Source cohort changed')
    audit_candidate_grid(rows,questions,pcfg)
    dap,skip,assignment,missing=build_sources(questions,rows,cfg['tokenskip']['ratios'],cfg['ratio_assignment_seed'])
    if len(skip)<cfg['smoke_questions']:raise ValueError('Insufficient verified sources for compression smoke')
    smoke=sorted([r['problem_id'] for r in skip],key=lambda pid:canonical_sha256([cfg['smoke_seed'],pid]))[:cfg['smoke_questions']]
    cfg.update(teacher=pcfg['teacher'],grading=pcfg['grading'],
               grading_method=pcfg.get('grading_method','typed_math_v2'),code_root=str(root/'code'))
    prompt=read_json(cfg['dap_prompt_config']);save(root/'inputs/dap_prompt.json',prompt)
    cfg['dap_prompt_config']=str(root/'inputs/dap_prompt.json')
    write_jsonl(root/'inputs/questions.jsonl',questions);write_jsonl(root/'inputs/dap_sources.jsonl',dap)
    write_jsonl(root/'inputs/tokenskip_sources.jsonl',skip)
    write_jsonl(root/'inputs/ratio_assignments.jsonl',[{'problem_id':pid,'ratio':ratio} for pid,ratio in assignment.items()])
    save(root/'inputs/source_audit.json',{'questions':len(questions),'eligible_questions':len(skip),'dap_sources':len(dap),
        'zero_raw_support_ids':missing,'smoke_ids':smoke,'ratio_counts_before_filtering':dict(Counter(assignment.values())),
        'ratio_counts_after_filtering':dict(Counter(r['assigned_ratio'] for r in skip)),
        'source_selection':{'DAP':'all unique correct noncapped B1 candidates, at most four per question',
                            'TokenSkip':'B1 shortest unique correct noncapped source; assigned ratio fixed on full cohort'},
        'parent_costs':read_json(parent/'selection'/cfg['source_cohort']/'summary.json')['generation_batch_seconds_by_method'],
        'development_records_must_not_train_students':cfg['source_cohort']!='student_pool'})
    snapshot=Path(cfg['tokenskip']['snapshot_path']);paths=sorted(snapshot.glob('*.safetensors'))+sorted(snapshot.glob('*.json'))
    if not any(p.suffix=='.safetensors' for p in paths):raise ValueError('Missing pinned LLMLingua weights')
    save(root/'inputs/compressor_model_hashes.json',evidence(paths))
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/FROZEN.json',[Path(config_path),*markers,root/'protocol/SOURCES.json',root/'protocol/frozen_config.json',
         *sorted((root/'inputs').glob('*'))],stage='unified_text_compression_inputs',formal_training_ready=False)


def load(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen text-compression implementation')
    return cfg


def sources_for(cfg,method,cohort,shard):
    if cohort not in ('smoke',cfg['source_cohort']):raise ValueError('Unknown compression cohort')
    if not 0<=shard<cfg['shards'] or (cohort=='smoke' and shard!=0):raise ValueError('Invalid compression shard')
    root=Path(cfg['result_root']);audit=read_json(root/'inputs/source_audit.json')
    if cohort=='smoke':ids=audit['smoke_ids']
    else:
        verify(root/method/'smoke/shard_00/COMPLETE.json')
        ids=[r['problem_id'] for i,r in enumerate(read_jsonl(root/'inputs/questions.jsonl')) if i%cfg['shards']==shard]
    sources=[r for r in read_jsonl(root/'inputs'/f'{method}_sources.jsonl') if r['problem_id'] in set(ids)]
    if not sources:raise ValueError('No eligible compression source in this shard')
    return ids,sources


def run_dap(cfg,cohort,shard):
    import torch
    root=Path(cfg['result_root']);out=root/'dap'/cohort/f'shard_{shard:02d}'
    ids,sources=sources_for(cfg,'dap',cohort,shard);out.mkdir(parents=True,exist_ok=False)
    admission(cfg);record_hardware(out);model,tok=teacher_bundle(cfg);torch.cuda.reset_peak_memory_stats()
    prompt=read_json(cfg['dap_prompt_config']);rows=[];batches=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for candidate in range(cfg['maximum_source_candidates']):
            selected=[r for r in sources if r['source_candidate_index']==candidate]
            for offset in range(0,len(selected),cfg['dap_generation']['batch_size']):
                batch=selected[offset:offset+cfg['dap_generation']['batch_size']]
                inputs=[{**r['question_record'],'messages':dap_messages(r['question_record']['question'],r['source']['response'],prompt)} for r in batch]
                settings={**cfg['dap_generation'],'seed':cfg['dap_generation']['seed']+candidate*cfg['candidate_seed_stride']}
                torch.cuda.synchronize();start=time.monotonic()
                generated=generate_condition_raw(model,tok,None,{'name':'B5'},inputs,settings)
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                batch_id=f'{candidate}_{offset:04d}';batches.append({'batch_id':batch_id,'condition':'B5','questions':len(batch),'generation_wall_seconds':elapsed})
                for source,result in zip(batch,generated):
                    q=source['question_record'];grade=grade_candidate(cfg,result['response'],q)
                    row={**result,**grade,**dap_structure(result['response']),'question':q['question'],'gold_answer':q['answer'],
                        'source_trace_key':source['source_trace_key'],'source_response_sha256':source['source_response_sha256'],
                        'candidate_index':source['source_candidate_index'],'source_generated_tokens':source['source']['generated_tokens'],
                        'batch_id':batch_id,'amortized_generation_wall_seconds':elapsed/len(batch),'full_thought_and_solution_retained':True}
                    rows.append(row);handle.write(json.dumps(row)+'\n')
                handle.flush();logging.info('Unified DAP %s shard=%d records=%d/%d',cohort,shard,len(rows),len(sources))
    if Counter(r['source_trace_key'] for r in rows)!=Counter(r['source_trace_key'] for r in sources):raise ValueError('Incomplete DAP source coverage')
    write_jsonl(out/'batches.jsonl',batches)
    save(out/'summary.json',{'pool_questions':len(ids),'eligible_source_questions':len({r['problem_id'] for r in sources}),
        'rewrite_records':len(rows),'correct_rewrites':sum(r['is_correct'] for r in rows),'cap_hits':sum(r['hit_max_new_tokens'] for r in rows),
        'rewrite_batch_seconds':sum(b['generation_wall_seconds'] for b in batches),'rewrite_input_tokens':sum(r['prompt_tokens'] for r in rows),
        'rewrite_output_tokens':sum(r['generated_tokens'] for r in rows),'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'formal_training_ready':False,'claim_boundary':cfg['claim_boundary']})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*sorted(out.glob('*'))],stage='unified_DAP_rewriting',formal_training_ready=False)


def run_tokenskip(cfg,cohort,shard):
    import torch
    from transformers import AutoTokenizer
    root=Path(cfg['result_root']);out=root/'tokenskip'/cohort/f'shard_{shard:02d}'
    ids,sources=sources_for(cfg,'tokenskip',cohort,shard);out.mkdir(parents=True,exist_ok=False)
    spec=cfg['tokenskip']
    if importlib.metadata.version('llmlingua')!=spec['package_version']:raise ValueError('LLMLingua version differs')
    from .pilot_storage import validate_scratch
    scratch=validate_scratch(cfg['runtime']['storage_policy'],os.environ)
    save(out/'offline_assets.json',stage_tokenskip_assets(cfg,scratch))
    os.environ['TIKTOKEN_CACHE_DIR']=str(scratch/'tiktoken')
    os.environ['NLTK_DATA']=str(scratch/'nltk')
    import nltk
    if os.environ['NLTK_DATA'] not in nltk.data.path:nltk.data.path.insert(0,os.environ['NLTK_DATA'])
    from llmlingua import PromptCompressor
    admission(cfg);record_hardware(out)
    compressor=PromptCompressor(model_name=spec['snapshot_path'],use_llmlingua2=True,device_map='cuda')
    tok=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    torch.cuda.reset_peak_memory_stats();rows=[];assigned=[]
    with (out/'compression.jsonl').open('x') as handle:
        for source in sources:
            ratios=spec['ratios'] if cfg['compression_mode']=='all_ratios' else [source['assigned_ratio']]
            q=source['question_record'];original=source['source']['response'];source_count=len(tok.encode(original,add_special_tokens=False))
            for ratio in ratios:
                torch.cuda.synchronize();start=time.monotonic()
                compressed=compress_trace(compressor,original,ratio,family=spec['family'])
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                completion=tokenskip_completion(compressed['compressed_prompt'],source['source']['predicted_answer'])
                grade=grade_candidate(cfg,completion,q)
                if not grade['is_correct']:raise ValueError('Appending the verified source answer changed its correctness')
                row={'problem_id':q['problem_id'],'question':q['question'],'question_role':q['question_role'],
                    'ratio':ratio,'assigned_ratio':source['assigned_ratio'],'assigned_for_sft':ratio==source['assigned_ratio'],
                    'prompt':tokenskip_prompt(q['question'],ratio),'completion':completion,'compression':compressed,
                    'source_trace_key':source['source_trace_key'],'source_response_sha256':source['source_response_sha256'],
                    'source_tokens':source_count,'compressed_body_tokens':len(tok.encode(compressed['compressed_prompt'],add_special_tokens=False)),
                    'completion_tokens':len(tok.encode(completion,add_special_tokens=False)),'compression_seconds':elapsed,
                    'final_answer_grade':grade,'answer_appended_from_verified_source':True,'compressed_reasoning_semantically_verified':False,
                    'development_records_must_not_train_students':cfg['source_cohort']!='student_pool'}
                rows.append(row);handle.write(json.dumps(row)+'\n')
                if row['assigned_for_sft']:assigned.append(row)
            handle.flush();logging.info('Unified TokenSkip %s shard=%d question=%s records=%d',cohort,shard,q['problem_id'],len(rows))
    if len(assigned)!=len(sources) or len({r['problem_id'] for r in assigned})!=len(sources):raise ValueError('Assigned-ratio coverage differs')
    write_jsonl(out/'assigned_ratio_examples.jsonl',assigned)
    save(out/'summary.json',{'pool_questions':len(ids),'source_questions':len(sources),'compression_records':len(rows),
        'assigned_records':len(assigned),'all_compression_seconds':sum(r['compression_seconds'] for r in rows),
        'assigned_ratio_measured_compression_seconds':sum(r['compression_seconds'] for r in assigned),
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'formal_training_ready':False,
        'appended_answer_score_is_not_reasoning_quality':True,'claim_boundary':cfg['claim_boundary']})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*sorted(out.glob('*'))],stage='unified_TokenSkip_compression',formal_training_ready=False)


def merge(cfg,cohort):
    import math
    from transformers import AutoTokenizer
    from .sae_generation_analysis import audit_batches
    root=Path(cfg['result_root']);out=root/'merged'/cohort
    if out.exists():raise FileExistsError(out)
    parts=1 if cohort=='smoke' else cfg['shards'];dap=[];skip=[];markers=[];costs=defaultdict(float)
    tok=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    for method,filename in [('dap','predictions.jsonl'),('tokenskip','compression.jsonl')]:
        for shard in range(parts):
            path=root/method/cohort/f'shard_{shard:02d}';marker=path/'COMPLETE.json';verify(marker);markers.append(marker)
            rows=list(read_jsonl(path/filename));_,sources=sources_for(cfg,method,cohort,shard)
            source_by_key={r['source_trace_key']:r for r in sources}
            for row in rows:
                source=source_by_key[row['source_trace_key']];q=source['question_record']
                if row['problem_id']!=q['problem_id'] or row['question']!=q['question'] or row['source_response_sha256']!=canonical_sha256(source['source']['response']):
                    raise ValueError('Compression source identity changed')
                if method=='dap':
                    if row['candidate_index']!=source['source_candidate_index'] or row['gold_answer']!=q['answer'] or not row['full_thought_and_solution_retained']:
                        raise ValueError('DAP source candidate/reference/output retention changed')
                    tokens=row['token_ids'];ended=bool(tokens and tokens[-1]==tok.eos_token_id)
                    if not tokens or tok.eos_token_id in tokens[:-1] or len(tokens)>cfg['dap_generation']['max_new_tokens']:
                        raise ValueError('Invalid DAP token/EOS boundary')
                    if row['generated_tokens']!=len(tokens)-int(ended) or row['hit_max_new_tokens']==ended or (not ended and len(tokens)!=cfg['dap_generation']['max_new_tokens']):
                        raise ValueError('Invalid DAP token/cap accounting')
                    if row['response']!=tok.decode(tokens,skip_special_tokens=True):raise ValueError('DAP text differs from sampled tokens')
                    grade=grade_candidate(cfg,row['response'],q)
                    if any(row[k]!=v for k,v in grade.items()):raise ValueError('DAP grading differs')
                else:
                    expected_completion=tokenskip_completion(row['compression']['compressed_prompt'],source['source']['predicted_answer'])
                    if row['completion']!=expected_completion or row['prompt']!=tokenskip_prompt(q['question'],row['ratio']):
                        raise ValueError('TokenSkip prompt/completion format differs')
                    if row['assigned_ratio']!=source['assigned_ratio'] or row['assigned_for_sft']!=(row['ratio']==source['assigned_ratio']):
                        raise ValueError('TokenSkip ratio assignment changed after filtering')
                    for field,text in [('source_tokens',source['source']['response']),('compressed_body_tokens',row['compression']['compressed_prompt']),('completion_tokens',row['completion'])]:
                        if row[field]!=len(tok.encode(text,add_special_tokens=False)):raise ValueError('TokenSkip token accounting differs')
                    if row['final_answer_grade']!=grade_candidate(cfg,row['completion'],q):raise ValueError('TokenSkip appended-answer grading differs')
            if method=='dap':
                expected=Counter(r['source_trace_key'] for r in sources)
                if Counter(r['source_trace_key'] for r in rows)!=expected:raise ValueError('DAP merge source coverage differs')
                measured=audit_batches(rows,list(read_jsonl(path/'batches.jsonl')))['B5']
                if not math.isclose(measured,read_json(path/'summary.json')['rewrite_batch_seconds'],rel_tol=1e-9):raise ValueError('DAP batch cost differs')
                dap.extend(rows);costs['dap_rewrite_seconds']+=measured
            else:
                expected={(r['problem_id'],ratio) for r in sources for ratio in (cfg['tokenskip']['ratios'] if cfg['compression_mode']=='all_ratios' else [r['assigned_ratio']])}
                if len(rows)!=len(expected) or {(r['problem_id'],r['ratio']) for r in rows}!=expected:raise ValueError('TokenSkip merge ratio grid differs')
                skip.extend(rows);costs['tokenskip_compression_seconds']+=sum(r['compression_seconds'] for r in rows)
    questions=list(read_jsonl(root/'inputs/questions.jsonl'));audit=read_json(root/'inputs/source_audit.json')
    if cohort=='smoke':questions=[q for q in questions if q['problem_id'] in audit['smoke_ids']]
    chosen=[]
    for question in questions:
        eligible=unique_correct([r for r in dap if r['problem_id']==question['problem_id']])
        if eligible:chosen.append(eligible[0])
    assigned=[r for r in skip if r['assigned_for_sft']];common=sorted({r['problem_id'] for r in chosen}&{r['problem_id'] for r in assigned})
    out.mkdir(parents=True);write_jsonl(out/'dap_all_rewrites.jsonl',dap);write_jsonl(out/'B5_selected_full_support.jsonl',chosen)
    write_jsonl(out/'tokenskip_all_compressions.jsonl',skip);write_jsonl(out/'B6_assigned_examples.jsonl',assigned)
    save(out/'summary.json',{'cohort':cohort,'questions':len(questions),'dap_rewrites':len(dap),'B5_correct_support':len(chosen),
        'tokenskip_records':len(skip),'B6_assigned_support':len(assigned),'B5_B6_common_ids':common,
        'compression_costs':dict(costs),'parent_generation_costs_full_source_pool':audit['parent_costs'],
        'source_cohort':cfg['source_cohort'],'compression_mode':cfg['compression_mode'],
        'cost_note':'B1 raw generation is shared. DAP cost includes each actual rewrite. TokenSkip cost covers only actually executed compressions in the registered mode; answer-appended correctness is not a reasoning metric.',
        'all_eight_method_support_ready':False,'student_training_complete':False,'claim_boundary':cfg['claim_boundary']})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*markers,*sorted(out.glob('*'))],stage='unified_text_compression_merge',formal_training_ready=False)
