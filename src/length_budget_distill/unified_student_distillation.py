"""Audited eight-method common support and equal-example mathematical SFT.

Teacher selection reuses the actual candidate selector, token deletion retains
TokenSkip's ratio prompt, and optimization reuses the tested TRL wrapper.
"""
from collections import Counter, defaultdict
from pathlib import Path
import logging
import importlib.metadata
import os
import shutil
import statistics
import time

from .experiment_io import read_json, publish_files_hash_verified
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify, admission
from .baseline_reproduction import record_hardware
from .unified_math_candidates import select_candidates, require_same_grading_method
from .reviewed_math_candidate_regrading import generation_provenance
from .ncsu_multi_answer import unique_correct
from .compression_baselines import tokenskip_prompt, tokenskip_completion
from .completion_supervision import encode_completion, TrainingExposureAudit
from .tokenskip_reproduction import expected_training_steps

CODE = Path(__file__).resolve().parents[2]
METHODS = tuple('B'+str(i) for i in range(8))


def unique_by_id(rows):
    result = {r['problem_id']:r for r in rows}
    if len(result) != len(rows): raise ValueError('Duplicate selected problem ID')
    return result


def common_student_rows(questions, selected, question_template, ratios):
    """Use all eight supports; never replace missing methods with easier ones."""
    qmap = unique_by_id(questions)
    if set(selected) != set(METHODS) or not qmap:
        raise ValueError('Eight complete method supports and a nonempty pool are required')
    if any(q['question_role'] != 'student_pool' for q in questions):
        raise ValueError('Development/calibration records cannot train the student')
    for method, values in selected.items():
        if not set(values) <= set(qmap): raise ValueError('Selected supervision is outside the student pool')
        for pid, row in values.items():
            if row['problem_id'] != pid or row['question'] != qmap[pid]['question']:
                raise ValueError('Selected problem/question identity differs')
            if method == 'B6':
                if not row['final_answer_grade']['is_correct'] or row['ratio'] not in ratios or not row['assigned_for_sft']:
                    raise ValueError('TokenSkip lacks a valid assigned-ratio source')
                if row['prompt'] != tokenskip_prompt(qmap[pid]['question'], row['ratio']):
                    raise ValueError('TokenSkip ratio prompt differs')
            elif not row['is_correct'] or row['hit_max_new_tokens'] or row['gold_answer'] != qmap[pid]['answer']:
                raise ValueError('Direct or rewritten supervision is ineligible')
    common = sorted(set.intersection(*(set(v) for v in selected.values())))
    if not common: raise ValueError('Eight-method common correct support is empty')
    datasets = {}
    for method in METHODS:
        values = []
        for pid in common:
            source = selected[method][pid]; question = qmap[pid]
            prompt = source['prompt'] if method == 'B6' else question_template.format(question=question['question'])
            completion = source['completion'] if method == 'B6' else source['response']
            if not completion.strip(): raise ValueError('Empty selected completion')
            values.append({'problem_id':pid,'question_role':'student_pool','dataset':question['dataset'],
                'question':question['question'],'near_component_id':question['near_component_id'],
                'baseline':method,'prompt':prompt,'completion':completion,
                'compression_ratio':source['ratio'] if method == 'B6' else None,
                'selected_source_sha256':canonical_sha256(source),
                'teacher_candidate_index':source.get('candidate_index'),
                'source_trace_key':source.get('source_trace_key'),
                'verified_source_answer_appended':method == 'B6'})
        datasets[method] = values
    exclusions = [{'problem_id':pid,'missing_methods':[m for m in METHODS if pid not in selected[m]]}
                  for pid in sorted(qmap) if pid not in common]
    return datasets, {'source_questions':len(questions),'common_questions':len(common),'common_ids':common,
                      'method_support':{m:len(selected[m]) for m in METHODS},'exclusions':exclusions,
                      'unique_questions_are_not_repeated_to_meet_a_target':True}


def collect_sources(cfg):
    roots = {key:Path(cfg[key]) for key in ('raw_candidate_root','steered_candidate_root','text_compression_root')}
    markers = []; configs = {}; selected = {}
    for name, root in roots.items():
        terminal = 'merged/student_pool/COMPLETE.json' if name == 'text_compression_root' else 'selection/student_pool/COMPLETE.json'
        for relative in ('protocol/FROZEN.json','protocol/SOURCES.json',terminal):
            marker = root/relative; verify(marker); markers.append(marker)
        configs[name] = read_json(root/'protocol/frozen_config.json')
    require_same_grading_method(*configs.values())
    for key in ('teacher','grading'):
        if configs['text_compression_root'][key] != configs['raw_candidate_root'][key]:
            raise ValueError('Text and raw supervision differ: '+key)
    raw, steering, text = (roots[n] for n in ('raw_candidate_root','steered_candidate_root','text_compression_root'))
    questions = list(read_jsonl(raw/'inputs/student_pool.jsonl'))
    if len(questions) != cfg['expected_source_questions']:
        raise ValueError('Student source-pool count changed')
    if list(read_jsonl(steering/'inputs/student_pool.jsonl')) != questions or list(read_jsonl(text/'inputs/questions.jsonl')) != questions:
        raise ValueError('The methods do not share the exact student question pool/order')
    if configs['text_compression_root']['source_cohort'] != 'student_pool' or Path(configs['text_compression_root']['candidate_root']) != raw:
        raise ValueError('Text compression uses a different source or a held-out cohort')
    if configs['text_compression_root']['compression_mode'] != 'assigned_only':
        raise ValueError('Primary SFT requires exactly one preassigned TokenSkip ratio per source')
    if cfg['tokenskip_ratios'] != configs['text_compression_root']['tokenskip']['ratios']:
        raise ValueError('The registered TokenSkip ratio set changed')
    for root, name in ((raw,'raw_candidate_root'),(steering,'steered_candidate_root')):
        if 'student_pool' not in configs[name]['allowed_generation_cohorts']:
            raise ValueError('Source generation was not registered for students')
        generation_root, generation_markers = generation_provenance(configs[name],root,'student_pool')
        markers.extend(generation_markers)
        for shard in range(configs[name]['shards']):
            hardware = read_json(generation_root/'generation/student_pool'/f'shard_{shard:02d}'/'hardware.json')
            if 'NVIDIA H100' not in hardware['inventory_csv']:
                raise ValueError('Primary direct generation must use the registered H100 hardware')
        values, _, _ = select_candidates(list(read_jsonl(root/'selection/student_pool/predictions.jsonl')), questions, configs[name])
        for method, lookup in values.items():
            stored = unique_by_id(list(read_jsonl(root/'selection/student_pool'/f'{method}_full_support.jsonl')))
            if {pid:{k:v for k,v in r.items() if k!='selected_baseline'} for pid,r in stored.items()} != lookup:
                raise ValueError('Saved selected supervision differs from actual candidate selection')
            if method in selected: raise ValueError('Repeated method from different candidate roots')
            selected[method] = lookup
    for key in ('teacher','grading','generation','candidates_per_question','candidate_seed_stride','selection_seed','shards'):
        if configs['raw_candidate_root'][key] != configs['steered_candidate_root'][key]:
            raise ValueError('Raw and steered student generation differ: '+key)
    if configs['steered_candidate_root']['registered_gpu_route'] != 'h100': raise ValueError('Unregistered steering hardware')
    grouped = defaultdict(list)
    for row in read_jsonl(text/'merged/student_pool/dap_all_rewrites.jsonl'): grouped[row['problem_id']].append(row)
    dap = {pid:eligible[0] for pid,rows in grouped.items() if (eligible:=unique_correct(rows))}
    if unique_by_id(list(read_jsonl(text/'merged/student_pool/B5_selected_full_support.jsonl'))) != dap:
        raise ValueError('DAP shortest-correct selection differs')
    selected['B5'] = dap
    skip = unique_by_id(list(read_jsonl(text/'merged/student_pool/B6_assigned_examples.jsonl')))
    for pid, row in skip.items():
        raw_source = selected['B1'].get(pid)
        if raw_source is None or row['source_response_sha256'] != canonical_sha256(raw_source['response']):
            raise ValueError('TokenSkip source differs from B1 shortest-correct')
        if row['completion'] != tokenskip_completion(row['compression']['compressed_prompt'], raw_source['predicted_answer']):
            raise ValueError('TokenSkip completion does not retain the verified source answer')
    selected['B6'] = skip
    if cfg['question_template'] != configs['raw_candidate_root']['methods']['B1']['question_template']:
        raise ValueError('Common student prompt differs from the registered mathematics prompt')
    costs = {key:read_json(path/('merged/student_pool/summary.json' if key=='text_compression_root' else 'selection/student_pool/summary.json'))
             for key,path in roots.items()}
    return questions, selected, markers, costs


def prepare(config_path):
    from transformers import AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    if root.exists(): raise FileExistsError(root)
    questions, selected, bindings, costs = collect_sources(cfg)
    datasets, audit = common_student_rows(questions,selected,cfg['question_template'],cfg['tokenskip_ratios'])
    smoke = Path(cfg['sft_interface_root']); verify(smoke/'protocol/FROZEN.json'); bindings.append(smoke/'protocol/FROZEN.json')
    scfg = read_json(smoke/'protocol/frozen_config.json')
    if cfg['students'] != scfg['students'] or cfg['lora'] != scfg['lora'] or cfg['training'] != scfg['training']:
        raise ValueError('Unified recipe/model changed after the actual synthetic capacity checks')
    for name in cfg['students']:
        marker = smoke/'training'/name/'COMPLETE.json'; verify(marker); bindings.append(marker)
    versions = read_json(smoke/'inputs/versions.json')
    if any(importlib.metadata.version(k)!=v for k,v in versions.items()):
        raise ValueError('SFT runtime versions differ from the actual capacity checks')
    inventories = read_json(smoke/'inputs/model_hashes.json')
    for inventory in inventories.values():
        for path,digest in inventory.items():
            if file_sha256(path) != digest: raise ValueError('Student model/tokenizer changed: '+path)
    root.mkdir(parents=True)
    save(root/'inputs/common_support.json',audit)
    save(root/'inputs/source_costs.json',{'components':costs,'scope':'Measured source generation and text compression. One-time SAE/ASC calibration/fit and downstream SFT/evaluation must be added separately; B0/B1 raw generation is shared.'})
    save(root/'inputs/model_hashes.json',inventories)
    save(root/'inputs/runtime_versions.json',read_json(smoke/'inputs/versions.json'))
    for method, rows in datasets.items(): write_jsonl(root/'text'/(method+'.jsonl'),rows)
    token_audit = {}
    for name, model in cfg['students'].items():
        tokenizer = AutoTokenizer.from_pretrained(model['snapshot_path'],local_files_only=True)
        token_audit[name] = {}
        for method, rows in datasets.items():
            encoded = [encode_completion(tokenizer,row,max_length=cfg['training']['max_length']) for row in rows]
            write_jsonl(root/'encoded'/name/(method+'.jsonl'),encoded)
            sizes = [len(r['input_ids']) for r in encoded]; targets = [r['supervision_tokens'] for r in encoded]
            token_audit[name][method] = {'records':len(encoded),'unique_problems':len(encoded),
                'input_tokens_one_exposure':sum(sizes),'supervision_tokens_one_exposure':sum(targets),
                'mean_supervision_tokens':statistics.mean(targets),'maximum_sequence_tokens':max(sizes),
                'truncated_records':0,'ratio_counts':dict(Counter(str(r['compression_ratio']) for r in encoded)) if method=='B6' else {}}
        logging.info('Eight-method SFT encoding complete: %s questions=%d',name,len(audit['common_ids']))
    save(root/'inputs/token_audit.json',token_audit)
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    cfg['code_root'] = str(root/'code'); cfg['actual_common_questions'] = audit['common_questions']
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'protocol/FROZEN.json',[Path(config_path),*bindings,root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
         *sorted((root/'inputs').glob('*')),*sorted((root/'text').glob('*')),*sorted((root/'encoded').glob('*/*.jsonl'))],
         training_data_ready=True,student_training_complete=False)


def train(config_path, student_name, method, seed):
    import torch
    from transformers import set_seed
    from safetensors.torch import load_file
    from .training import run_trl_sft
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']): raise ValueError('Use the frozen unified SFT source')
    if any(importlib.metadata.version(k)!=v for k,v in read_json(root/'inputs/runtime_versions.json').items()):
        raise ValueError('Frozen SFT dependencies changed')
    if student_name not in cfg['primary_stage']['students'] or method not in cfg['primary_stage']['methods'] or seed not in cfg['student_seeds']:
        raise ValueError('SFT cell is outside the registered first stage')
    name = f'{student_name}/{method}/seed_{seed}'
    out = root/'training'/name; adapter = Path(cfg['checkpoint_root'])/name
    if out.exists() or adapter.exists(): raise FileExistsError('Preserve the existing SFT attempt')
    path = root/'encoded'/student_name/(method+'.jsonl'); rows = list(read_jsonl(path))
    if len(rows) != cfg['actual_common_questions'] or any(r['question_role']!='student_pool' or r['baseline']!=method for r in rows):
        raise ValueError('SFT data no longer matches the registered common support')
    for path_value,digest in read_json(root/'inputs/model_hashes.json')[student_name].items():
        if file_sha256(path_value) != digest: raise ValueError('Changed student input: '+path_value)
    scratch_free=shutil.disk_usage(os.environ['TMPDIR']).free
    if cfg['runtime'].get('storage_policy'):
        from .pilot_storage import estimate
        scratch_required=estimate(cfg,'train',method)['temporary_required_bytes']
    else:
        scratch_required=max(cfg['runtime'].get('minimum_free_scratch_bytes',2*1024**3),4*path.stat().st_size+1024**3)
    if scratch_free<scratch_required:
        raise RuntimeError(f'Insufficient SFT job-local scratch: {scratch_free} < {scratch_required}')
    admission(cfg); out.mkdir(parents=True); record_hardware(out)
    save(out/'storage_preflight.json',{'free_bytes':scratch_free,'required_bytes':scratch_required,
        'encoded_input_bytes':path.stat().st_size,
        'policy':cfg['runtime'].get('storage_policy','Legacy minimum or four encoded-input copies plus 1 GiB; no periodic Trainer checkpoints.')})
    model = cfg['students'][student_name]
    student = {'model_name':model['snapshot_path'],'torch_dtype':'bfloat16','use_lora':True,'lora':cfg['lora'],
               'tokenizer_kwargs':{'local_files_only':True,'padding_side':'right'}}
    local = Path(os.environ['TMPDIR'])/'unified_sft'/student_name/method/str(seed)
    if cfg['runtime'].get('storage_policy',{}).get('scratch_root'):
        from .pilot_storage import validate_scratch
        validate_scratch(cfg['runtime']['storage_policy'],os.environ)
    training = {**cfg['training'],'seed':seed,'data_seed':seed,'output_dir':str(local),
                'save_strategy':'no','model_init_kwargs':{'local_files_only':True,'attn_implementation':'sdpa'}}
    run = {'student':student,'training':training,'data':{'train_path':str(path),'text_format':'pretokenized_completion'}}
    save(out/'run_config.json',run)
    meter = TrainingExposureAudit(rows,max_length=training['max_length'])
    os.environ['LBD_RUNTIME_OUTPUT_DIR'] = str(local); set_seed(seed)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
    trainer = run_trl_sft(run,before_train=meter.install)
    torch.cuda.synchronize(); seconds = time.monotonic()-start
    if trainer.state.global_step != expected_training_steps(len(rows),cfg): raise ValueError('Incomplete SFT update count')
    weights = load_file(str(local/'adapter_model.safetensors')); bs = [v for k,v in weights.items() if 'lora_B' in k]
    if not bs or not all(torch.isfinite(v).all() for v in weights.values()) or not any(torch.count_nonzero(v) for v in bs):
        raise ValueError('Invalid or unchanged LoRA weights')
    metrics = {**meter.summary(),'student':student_name,'method':method,'seed':seed,
        'optimizer_steps':trainer.state.global_step,'actual_epoch':trainer.state.epoch,'configured_epochs':training['num_train_epochs'],
        'elapsed_seconds_including_batch_audit':seconds,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'log_history':trainer.state.log_history,
        'training_complete':True,'student_evaluation_complete':False,'regime':'equal_examples',
        'nonzero_lora_b_tensors':sum(bool(torch.count_nonzero(v)) for v in bs)}
    save(local/'training_metrics.json',metrics)
    publish_files_hash_verified(local,adapter,('adapter_model.safetensors','adapter_config.json','training_metrics.json'))
    seal(adapter/'TRAIN_COMPLETE.json',[root/'protocol/FROZEN.json',root/'protocol/SOURCES.json',path,out/'run_config.json',
        out/'hardware.json',out/'storage_preflight.json',*sorted(adapter.glob('*'))],training_complete=True,evaluation_complete=False)
    seal(out/'COMPLETE.json',[adapter/'TRAIN_COMPLETE.json',adapter/'training_metrics.json'],student_evaluation_complete=False)
    logging.info('Unified equal-example SFT complete: %s steps=%d epoch=%s',name,trainer.state.global_step,trainer.state.epoch)
