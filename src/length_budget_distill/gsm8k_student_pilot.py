"""GSM8K student-utility pilot using the existing audited generation/SFT engines.

The independent protocol admits fourteen operating points, not a replacement
for the immutable eight-method MATH protocol. Every downstream stage binds
complete parent artifacts; no partial shard can supply student supervision.
"""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import importlib.metadata
import logging
import re
import shutil

from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import save, seal, verify, evidence
from .baseline_data_preflight import normalize_question, near_matches, load_source
from .unified_math_candidates import select_candidates, grade_candidate
from .unified_text_compression import build_sources
from .ncsu_multi_answer import unique_correct
from .compression_baselines import tokenskip_completion, tokenskip_prompt

CODE = Path(__file__).resolve().parents[2]
GRADER = 'gsm8k_explicit_answer_quantity_v3'
ANSWER_REGION_PATTERN = r'(?i)(?:\b(?:final\s+)?answer\s*:|\\boxed\s*\{)'


def condition_ids(cfg):
    return ['B0', 'B1', 'B2', *[f'{m}__{s:g}' for m in ('B3', 'B4') for s in cfg['strengths'][m]],
            'B5', 'B6', *[f'B7__{s:g}' for s in cfg['strengths']['B7']]]


def absolute_config(cfg, project):
    cfg = copy.deepcopy(cfg)
    for name in ('result_root', 'checkpoint_root', 'mechanism_root', 'generic_text_root', 'sft_interface_root', 'dap_prompt_config'):
        cfg[name] = str((project / cfg[name]).resolve())
    cfg['history_question_sources'] = [str((project / p).resolve()) for p in cfg['history_question_sources']]
    return cfg


def freeze_child(main, cfg, root, bindings=()):
    root = Path(root)
    cfg = {**copy.deepcopy(cfg), 'result_root': str(root), 'code_root': main['code_root'],
           'pilot_root': main.get('pilot_root', main['result_root'])}
    save(root/'protocol/frozen_config.json', cfg)
    # Share the immutable execution snapshot rather than copy it per condition.
    seal(root/'protocol/SOURCES.json', [Path(main['code_root']).parent/'protocol/SOURCES.json'])
    execution_bindings = [Path(main['execution_marker'])] if main.get('execution_marker') else []
    seal(root/'protocol/FROZEN.json', [Path(cfg['pilot_root'])/'protocol/FROZEN.json',*execution_bindings,
         root/'protocol/frozen_config.json', root/'protocol/SOURCES.json', *bindings,
         *sorted((root/'inputs').glob('*'))], formal_claim_allowed=False)
    return cfg


def load_child(main, relative):
    root = Path(main.get('child_roots',{}).get(relative,Path(main['result_root'])/relative))
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    return read_json(root/'protocol/frozen_config.json')


def ordered_questions(rows, seed):
    return sorted(rows, key=lambda r: (canonical_sha256([seed, r['problem_id']]), r['problem_id']))


def isolate_sources(train, references, cfg):
    """Exclude exact/near history or test matches and keep one near-group member."""
    blocked = {normalize_question(r['question']) for r in references}
    unique = {}; excluded = []
    for row in train:
        key = normalize_question(row['question'])
        if key in blocked or key in unique:
            excluded.append({'problem_id': row['problem_id'], 'reason': 'history_test_or_duplicate_exact'})
        else:
            unique[key] = row
    rows = list(unique.values()); policy = cfg['decontamination']
    matches = near_matches(rows, references, n=policy['shingle_size'], threshold=policy['jaccard_threshold'])
    bad = {r['query_id'] for r in matches}
    excluded += [{'problem_id': k, 'reason': 'history_or_test_near'} for k in sorted(bad)]
    rows = [r for r in rows if r['problem_id'] not in bad]
    pairs = near_matches(rows, rows, n=policy['shingle_size'], threshold=policy['jaccard_threshold'])
    parent = {r['problem_id']: r['problem_id'] for r in rows}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for pair in pairs:
        a, b = find(pair['query_id']), find(pair['reference_id'])
        parent[max(a,b)] = min(a,b)
    kept = []; seen = set()
    for row in ordered_questions(rows, cfg['seed']):
        group = find(row['problem_id'])
        if group in seen:
            excluded.append({'problem_id': row['problem_id'], 'reason': 'training_near_group_duplicate'})
        else:
            seen.add(group); kept.append({**row, 'near_component_id': group})
    return kept, excluded, matches


def question_record(row, index, revision, role, template, *, test=False):
    answer = row['answer'].split('####')[-1].strip()
    return {'problem_id': f'{"gsm-test" if test else "hf"}-{index:06d}', 'source_index': index,
            'question': row['question'], 'raw_answer': row['answer'], 'answer': answer,
            'dataset': 'gsm8k', 'source_revision': revision, 'question_role': role,
            'near_component_id': f'{"test" if test else "train"}-{index}',
            'answer_spec': {'kind': 'symbolic'}, 'prompt': template.format(question=row['question'])}


def prepare(config_path):
    """CPU-only source audit and immutable pilot snapshot; no GPU or submission."""
    from transformers import AutoTokenizer
    cfg = absolute_config(read_json(config_path), CODE); root = Path(cfg['result_root'])
    if root.exists(): raise FileExistsError(root)
    mech = Path(cfg['mechanism_root']); marker = mech/'protocol/FROZEN.json'; verify(marker)
    parent = read_json(mech/'protocol/frozen_config.json')
    if cfg['teacher'] != parent['teacher']: raise ValueError('Teacher differs from the fixed SAE parent')
    cfg['sae'] = parent['sae']; bindings = [Path(config_path), marker, Path(cfg['sae']['checkpoint_path'])]
    history = []
    for path in [*cfg['history_question_sources'], str(mech/'inputs/questions.jsonl'), str(mech/'inputs/smoke_questions.jsonl')]:
        history.extend(read_jsonl(path)); bindings.append(Path(path))
    train = load_source({'path': cfg['gsm8k']['train_arrow']}); test = load_source({'path': cfg['gsm8k']['test_arrow']})
    if (len(train), len(test)) != (7473, 1319): raise ValueError('Pinned GSM8K source counts changed')
    revision = cfg['gsm8k']['revision']; template = cfg['question_template']
    full = [question_record(row,i,revision,'student_pool',template) for i,row in enumerate(train)]
    evaluation = [question_record(row,i,revision,'locked_evaluation',template,test=True) for i,row in enumerate(test)]
    refs = [{'problem_id': 'excluded-'+str(i), 'question': r['question']} for i,r in enumerate(history)] + evaluation
    eligible, exclusions, near = isolate_sources(full, refs, cfg)
    if len(eligible) < cfg['maximum_source_questions']: raise ValueError('Insufficient independent GSM8K source pool')
    mechanism = list(read_jsonl(mech/'inputs/questions.jsonl'))
    qmap = {normalize_question(q['question']): q for q in full}
    def cohort(split, role):
        return [{**qmap[normalize_question(q['question'])], 'question_role':role,
                 'previously_observed':True} for q in mechanism if q['question_split']==split]
    development = cohort('dev','development'); calibration = cohort('discovery','calibration')
    if len(development)!=cfg['development_questions'] or len(calibration)!=400: raise ValueError('Mechanism role counts changed')
    root.mkdir(parents=True)
    for name,rows in [('source_order',eligible[:cfg['maximum_source_questions']]), ('development',development),
                      ('calibration',calibration), ('evaluation',evaluation[50:]),
                      ('smoke',[{**r,'question_role':'smoke'} for r in evaluation[:cfg['smoke_questions']]])]:
        write_jsonl(root/'inputs'/f'{name}.jsonl',rows)
    write_jsonl(root/'inputs/exclusions.jsonl',exclusions); write_jsonl(root/'inputs/near_matches.jsonl',near)
    save(root/'inputs/audit.json',{'eligible_questions':len(eligible),'source_questions':cfg['source_pool_questions'],
         'reserved_source_questions':cfg['maximum_source_questions'],'development':len(development),
         'evaluation':1269,'conditions':condition_ids(cfg),'historical_student_results_are_not_new_results':True,
         'common_support_not_yet_established':True})
    inventories = {}
    for name, model in {'teacher':cfg['teacher'], **cfg['students'], 'compressor':cfg['tokenskip']}.items():
        snapshot = Path(model['snapshot_path']); files = sorted(snapshot.glob('*.safetensors'))+sorted(snapshot.glob('*.json'))
        if not any(p.suffix=='.safetensors' for p in files): raise ValueError('Missing model weights: '+name)
        inventories[name] = evidence(files)
    save(root/'inputs/model_hashes.json',inventories)
    interface = Path(cfg['sft_interface_root']); verify(interface/'protocol/FROZEN.json')
    prototype = read_json(interface/'protocol/frozen_config.json')
    if cfg['training']!=prototype['training'] or cfg['lora']!=prototype['lora'] or cfg['students']['qwen3b_student']!=prototype['students']['qwen3b_student']:
        raise ValueError('Pilot student recipe differs from tested interface')
    verify(interface/'training/qwen3b_student/COMPLETE.json')
    versions = read_json(interface/'inputs/versions.json')
    if any(importlib.metadata.version(k)!=v for k,v in versions.items()): raise ValueError('SFT overlay changed')
    save(root/'inputs/runtime_versions.json',versions)
    tok=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    cfg['teacher_eos_token_id']=tok.eos_token_id
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    cfg['code_root']=str(root/'code'); cfg['pilot_root']=str(root)
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'protocol/FROZEN.json',[*bindings,Path(cfg['gsm8k']['train_arrow']),Path(cfg['gsm8k']['test_arrow']),
         interface/'protocol/FROZEN.json',interface/'training/qwen3b_student/COMPLETE.json',
         root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',*sorted((root/'inputs').glob('*'))],
         formal_claim_allowed=False,student_training_complete=False)


def prepare_calibration(main):
    root=Path(main['result_root']); out=root/'calibration'; generic=Path(main['generic_text_root'])
    verify(generic/'protocol/FROZEN.json')
    questions=list(read_jsonl(root/'inputs/calibration.jsonl')); pairs=[]
    for q in ordered_questions(questions,main['seed']):
        body=re.sub(r'<<.*?>>','',q['raw_answer'].split('####')[0]).strip()
        concise=body+'\n\\boxed{'+q['answer']+'}'
        if grade_candidate(main,concise,q)['is_correct']:
            pairs.append({**q,'concise':concise,'source_trace_id':'official-'+q['problem_id']})
    if len(pairs)<main['asc']['pairs']: raise ValueError('Insufficient valid concise calibration references')
    write_jsonl(out/'inputs/pairs.jsonl',pairs)
    for filename in ('generic_text.jsonl','generic_text_holdout.jsonl'):
        write_jsonl(out/'inputs'/filename,list(read_jsonl(generic/'inputs'/filename)))
    cfg={**main,'grading':{'method':GRADER,'config':main['grading']}}
    freeze_child(main,cfg,out,[generic/'protocol/FROZEN.json'])


def prepare_directions(main):
    from .math_steering_directions import match_calibration_pairs
    root=Path(main['result_root']); parent=root/'calibration'; verify(parent/'asc_calibration/COMPLETE.json')
    cfg={**main,'pairs':main['asc']['pairs'],'layer_index':17,'maximum_sequence_tokens':2048,
         'direction_stage':'GSM8K_dense_and_SAE_directions'}
    pairs=match_calibration_pairs(list(read_jsonl(parent/'asc_calibration/pairs.jsonl')),
                                 list(read_jsonl(parent/'asc_calibration/attempts.jsonl')),cfg)
    out=root/'directions'; write_jsonl(out/'inputs/pairs.jsonl',pairs)
    save(out/'inputs/sae_hashes.json',evidence([Path(main['sae']['checkpoint_path'])]))
    save(out/'inputs/teacher_hashes.json',read_json(root/'inputs/model_hashes.json')['teacher'])
    freeze_child(main,cfg,out,[parent/'asc_calibration/COMPLETE.json'])


def candidate_config(main, group, round_number):
    cfg=copy.deepcopy(main); root=Path(main['result_root'])
    cfg['allowed_generation_cohorts']=['smoke','student_pool']; cfg['registered_gpu_route']=main['runtime']['generation_route']
    methods={}; bindings=[]
    if group=='raw':
        for m,key in [('B1','question_template'),('B2','concise_question_template')]:
            methods[m]={'kind':'unmodified','question_template':main[key]}
    elif group=='steered':
        for relative in ('calibration/asc/COMPLETE.json','directions/directions/COMPLETE.json'):
            verify(root/relative); bindings.append(root/relative)
        for m in ('B3','B4','B7'):
            for strength in cfg['strengths'][m]:
                methods[f'{m}__{strength:g}']={'kind':'absolute_vector' if m=='B4' else 'relative_vector',
                    'question_template':main['question_template'],'layer_index':16 if m=='B4' else 17,'strength':strength,
                    'vector_file':str(root/('calibration/asc/vector.safetensors' if m=='B4' else 'directions/directions/directions.safetensors')),
                    'vector_key':{'B3':'dense_reference_minus_generated','B4':'vector','B7':'sae_short_8'}[m]}
    else: raise ValueError('Unknown candidate group')
    cfg['methods']=methods
    out=root/f'round_{round_number}'/group
    order=list(read_jsonl(root/'inputs/source_order.jsonl'))
    questions=order[:cfg['source_pool_questions']] if round_number==1 else order[cfg['source_pool_questions']:cfg['maximum_source_questions']]
    if round_number not in (1,2): raise ValueError('Unregistered source round')
    write_jsonl(out/'inputs/student_pool.jsonl',questions)
    write_jsonl(out/'inputs/smoke.jsonl',list(read_jsonl(root/'inputs/smoke.jsonl')))
    return freeze_child(main,cfg,out,bindings)


def prepare_compression(main, round_number):
    root=Path(main['result_root']); raw=load_child(main,f'round_{round_number}/raw')
    parent=Path(raw['result_root']); marker=parent/'selection/student_pool/COMPLETE.json'; verify(marker)
    questions=list(read_jsonl(parent/'inputs/student_pool.jsonl')); rows=list(read_jsonl(parent/'selection/student_pool/predictions.jsonl'))
    dap,skip,ratios,missing=build_sources(questions,rows,main['tokenskip_ratios'],main['ratio_assignment_seed'])
    if len(skip)<4: raise ValueError('Insufficient compression sources')
    out=root/f'round_{round_number}'/'text'
    for name,values in [('questions',questions),('dap_sources',dap),('tokenskip_sources',skip)]:
        write_jsonl(out/'inputs'/f'{name}.jsonl',values)
    save(out/'inputs/source_audit.json',{'smoke_ids':sorted(r['problem_id'] for r in skip)[:4],
         'parent_costs':read_json(parent/'selection/student_pool/summary.json'),'zero_raw_support_ids':missing,
         'ratio_counts_before_filtering':dict(Counter(ratios.values()))})
    write_jsonl(out/'inputs/ratio_assignments.jsonl',[{'problem_id':k,'ratio':v} for k,v in sorted(ratios.items())])
    save(out/'inputs/dap_prompt.json',read_json(main['dap_prompt_config']))
    cfg={**main,'source_cohort':'student_pool','compression_mode':'assigned_only','maximum_source_candidates':4,
         'dap_generation':{**main['generation'],'seed':main['generation']['seed']+7},
         'dap_prompt_config':str(out/'inputs/dap_prompt.json')}
    return freeze_child(main,cfg,out,[marker])


def collect_pilot_sources(main, rounds):
    root=Path(main['result_root']); selected={m:{} for m in condition_ids(main)}; questions=[]; bindings=[]; costs={}
    for number in rounds:
        local_questions=None
        for group in ('raw','steered'):
            cfg=load_child(main,f'round_{number}/{group}'); source=Path(cfg['result_root'])
            marker=source/'selection/student_pool/COMPLETE.json'; verify(marker); bindings.append(marker)
            qs=list(read_jsonl(source/'inputs/student_pool.jsonl'))
            if local_questions is not None and local_questions!=qs: raise ValueError('Candidate groups differ in source pool')
            local_questions=qs
            values,_,_=select_candidates(list(read_jsonl(source/'selection/student_pool/predictions.jsonl')),qs,cfg)
            for m,lookup in values.items():
                if set(selected[m])&set(lookup): raise ValueError('Repeated source question across rounds')
                selected[m].update(lookup)
            costs[f'{number}_{group}']=read_json(source/'selection/student_pool/summary.json')
        questions.extend(local_questions)
        text=root/f'round_{number}/text'; marker=text/'merged/student_pool/COMPLETE.json';verify(marker);bindings.append(marker)
        if list(read_jsonl(text/'inputs/questions.jsonl'))!=local_questions: raise ValueError('Text compression pool differs')
        grouped=defaultdict(list)
        for row in read_jsonl(text/'merged/student_pool/dap_all_rewrites.jsonl'): grouped[row['problem_id']].append(row)
        selected['B5'].update({pid:values[0] for pid,rows in grouped.items() if (values:=unique_correct(rows))})
        for row in read_jsonl(text/'merged/student_pool/B6_assigned_examples.jsonl'):
            pid=row['problem_id']; source=selected['B1'][pid]
            if row['source_response_sha256']!=canonical_sha256(source['response']) or row['completion']!=tokenskip_completion(row['compression']['compressed_prompt'],source['predicted_answer']):
                raise ValueError('TokenSkip no longer binds the shortest correct source')
            selected['B6'][pid]=row
        costs[f'{number}_text']=read_json(text/'merged/student_pool/summary.json')
    if len({q['problem_id'] for q in questions})!=len(questions): raise ValueError('Repeated pilot source question')
    return questions,selected,bindings,costs


def common_training_rows(questions, selected, cfg, *, allow_reduced=False):
    if set(selected)!=set(condition_ids(cfg)): raise ValueError('Incomplete fourteen-condition matrix')
    qmap={q['problem_id']:q for q in questions}
    if len(qmap)!=len(questions) or any(q['question_role']!='student_pool' for q in questions): raise ValueError('Invalid training pool')
    for method, lookup in selected.items():
        if not set(lookup)<=set(qmap): raise ValueError('Foreign supervision')
        for pid,row in lookup.items():
            if row['question']!=qmap[pid]['question']: raise ValueError('Changed question')
            if method=='B6':
                if not row['assigned_for_sft'] or not row['final_answer_grade']['is_correct'] or row['ratio'] not in cfg['tokenskip_ratios']:
                    raise ValueError('Invalid ratio-conditioned supervision')
                if row['prompt']!=tokenskip_prompt(qmap[pid]['question'],row['ratio']): raise ValueError('Changed ratio prompt')
            elif not row['is_correct'] or row['hit_max_new_tokens'] or row['gold_answer']!=qmap[pid]['answer']:
                raise ValueError('Wrong, capped, or changed supervision')
    common=set.intersection(*(set(v) for v in selected.values()))
    n=min(cfg['target_train_questions'],len(common)//8*8)
    audit={'source_questions':len(questions),'all_condition_common_questions':len(common),'training_questions':n,
           'method_support':{m:len(v) for m,v in selected.items()},
           'exclusions':[{'problem_id':pid,'missing_conditions':[m for m,v in selected.items() if pid not in v]} for pid in qmap if pid not in common]}
    if n<cfg['target_train_questions'] and not allow_reduced: return None,audit
    if n<cfg['minimum_train_questions']: raise ValueError('Common support below registered minimum')
    chosen=[q for q in ordered_questions(questions,cfg['selection_seed']) if q['problem_id'] in common][:n]
    datasets={}
    for method,lookup in selected.items():
        values=[]
        for q in chosen:
            source=lookup[q['problem_id']]
            values.append({**{k:q[k] for k in ('problem_id','question_role','dataset','question','near_component_id')},
                'baseline':method,'prompt':source['prompt'] if method=='B6' else cfg['question_template'].format(question=q['question']),
                'completion':source['completion'] if method=='B6' else source['response'],
                'compression_ratio':source['ratio'] if method=='B6' else None,
                'selected_source_sha256':canonical_sha256(source),'teacher_candidate_index':source.get('candidate_index'),
                'source_trace_key':source.get('source_trace_key'),'verified_source_answer_appended':method=='B6'})
        if any(not r['completion'].strip() for r in values): raise ValueError('Empty supervision')
        datasets[method]=values
    audit['common_ids']=[q['problem_id'] for q in chosen]
    return datasets,audit


def prepare_sft(main, rounds):
    from transformers import AutoTokenizer
    from .completion_supervision import encode_completion
    questions,selected,bindings,costs=collect_pilot_sources(main,rounds)
    datasets,audit=common_training_rows(questions,selected,main,allow_reduced=len(rounds)==2)
    root=Path(main['result_root']); decision=root/'support'/f'round_{max(rounds)}'
    if (decision/'COMPLETE.json').exists():
        previous=verify(decision/'COMPLETE.json')
        if read_json(decision/'audit.json') != audit or previous['requires_extension'] != (datasets is None):
            raise ValueError('Resumed common-support audit differs from completed audit')
        if any(previous['hashes'].get(str(Path(p).resolve())) != file_sha256(p) for p in bindings):
            raise ValueError('Resumed common-support inputs differ from completed audit')
    else:
        save(decision/'audit.json',audit)
        seal(decision/'COMPLETE.json',[*bindings,decision/'audit.json'],requires_extension=datasets is None)
    if datasets is None: return False
    out=root/'sft'; token_audit={}; model=main['students']['qwen3b_student']
    tok=AutoTokenizer.from_pretrained(model['snapshot_path'],local_files_only=True)
    teacher_tok=AutoTokenizer.from_pretrained(main['teacher']['snapshot_path'],local_files_only=True)
    from .sae_generation_controls import generated_regions
    for method,rows in datasets.items():
        encoded=[encode_completion(tok,r,max_length=main['training']['max_length']) for r in rows]
        write_jsonl(out/'text'/f'{method}.jsonl',rows);write_jsonl(out/'encoded/qwen3b_student'/f'{method}.jsonl',encoded)
        token_audit[method]={'questions':len(rows),'input_tokens_one_exposure':sum(len(r['input_ids']) for r in encoded),
            'supervision_tokens_one_exposure':sum(r['supervision_tokens'] for r in encoded),
            'mean_response_tokens':sum(len(tok.encode(r['completion'],add_special_tokens=False)) for r in rows)/len(rows),
            'maximum_sequence_tokens':max(len(r['input_ids']) for r in encoded)}
        lengths=[len(teacher_tok.encode(r['completion'],add_special_tokens=False)) for r in rows]
        regions=[generated_regions(teacher_tok,r['completion'],main.get('answer_heading_pattern',ANSWER_REGION_PATTERN)) for r in rows]
        token_audit[method].update(mean_teacher_tokens=sum(lengths)/len(lengths),
            median_teacher_tokens=__import__('statistics').median(lengths),
            mean_teacher_body_tokens=sum(r['region_token_counts']['reasoning_body'] for r in regions)/len(regions))
    for filename,value in [('common_support.json',audit),('source_costs.json',costs),('token_audit.json',token_audit),
                           ('model_hashes.json',{'qwen3b_student':read_json(root/'inputs/model_hashes.json')['qwen3b_student']}),
                           ('runtime_versions.json',read_json(root/'inputs/runtime_versions.json'))]: save(out/'inputs'/filename,value)
    cfg={**main,'actual_common_questions':audit['training_questions'],
         'primary_stage':{'students':['qwen3b_student'],'methods':condition_ids(main)},'regime':'equal_examples'}
    freeze_child(main,cfg,out,[*bindings,*sorted((out/'text').glob('*')),*sorted((out/'encoded/qwen3b_student').glob('*'))])
    return True
