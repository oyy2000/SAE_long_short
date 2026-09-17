"""Audited NCSU stages for ASC-CES, DAP, and TokenSkip method checks.

These stages establish executable baselines; the small observed development
cohort is explicitly not the expanded student-distillation experiment.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from pathlib import Path
import importlib.metadata
import json
import logging
import os
import random
import re
import shutil
import time

from .factorial import file_sha256, canonical_sha256
from .records import read_jsonl, write_jsonl
from .experiment_io import read_json
from .ncsu_reproduction import resolve, save, seal, verify, evidence, admission, teacher_bundle
from .verifiers import extract_final_answer, verify_answer
from .compression_baselines import (dap_messages, dap_structure, compress_trace,
    ratio_for_problem, tokenskip_prompt, tokenskip_completion)

CODE = Path(__file__).resolve().parents[2]


def ordered(rows, seed):
    return sorted(rows, key=lambda row: canonical_sha256([seed, row['problem_id']]))


def prepare(config_path):
    """Freeze disjoint calibration/development cohorts, generic text, and code."""
    if read_json(config_path).get('math_cohorts'):
        from .math_baseline_calibration import prepare as prepare_math
        return prepare_math(config_path)
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    cfg = read_json(config_path)
    root = resolve(cfg['result_root'])
    if root.exists():
        raise FileExistsError(root)
    audit_marker = resolve(cfg['source_audit'])
    audit_doc = read_json(audit_marker)
    audit_path = audit_marker.parent/'source_audit.json'
    if audit_doc['status'] != 'source_audit_complete' or file_sha256(audit_path) != audit_doc['source_audit_sha256']:
        raise ValueError('Source audit changed or is incomplete')
    audit = read_json(audit_path)
    for name, digest in audit['paper_hashes'].items():
        if file_sha256(audit_marker.parent/name) != digest:
            raise ValueError(f'Changed source paper: {name}')
    for method, entry in audit['methods'].items():
        for name, item in entry['files'].items():
            if file_sha256(audit_marker.parent/method/name) != item['sha256']:
                raise ValueError(f'Changed upstream source: {method}/{name}')
    tok = AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'], local_files_only=True)
    corpus = list(read_jsonl(resolve(cfg['corpus'])))
    groups = defaultdict(list)
    for row in corpus:
        if row['is_correct'] and not row['hit_max_new_tokens'] and row['candidate_index'] < cfg['validation']['source_candidates']:
            groups[row['problem_id']].append(row)
    pairs, sources = [], []
    for pid, rows in groups.items():
        rows.sort(key=lambda r: r['candidate_index'])
        first = rows[0]
        if first['question_split'] == cfg['asc']['pair_split']:
            reference = re.sub(r'<<[^>]*>>', '', first['raw_answer']).replace('####', 'Answer:')
            short_count = len(tok.encode(reference, add_special_tokens=False))
            longer = [r for r in rows if r['solution_token_count'] > short_count]
            if longer:
                row = longer[0]
                if not verify_answer(extract_final_answer(reference), row['answer']):
                    raise ValueError('Reference normalization changed the final answer')
                pairs.append({'problem_id':pid, 'prompt':row['prompt'], 'question':row['question'],
                    'concise':reference, 'verbose':row['solution'], 'answer':row['answer'],
                    'source_trace_id':row['trace_id'], 'source_split':row['question_split'],
                    'concise_tokens':short_count, 'verbose_tokens':row['solution_token_count']})
        if first['question_split'] == cfg['validation']['split']:
            sources.append(min(rows, key=lambda r: (r['solution_token_count'], r['trace_id'])))
    requested_pairs = cfg.get('calibration_generation', {}).get('pool_size', cfg['asc']['pairs'])
    pairs = ordered(pairs, cfg['seed'])[:requested_pairs]
    sources = ordered(sources, cfg['seed'])[:cfg['validation']['count']]
    if len(pairs) != requested_pairs or len(sources) != cfg['validation']['count']:
        raise ValueError('Insufficient eligible calibration/development questions')
    pair_ids, dev_ids = {r['problem_id'] for r in pairs}, {r['problem_id'] for r in sources}
    if pair_ids & dev_ids or len(pair_ids) != len(pairs) or len(dev_ids) != len(sources):
        raise ValueError('Duplicate or overlapping question cohorts')
    if cfg.get('prompt_policy'):
        policy=cfg['prompt_policy']
        if policy['rendering']!='native_chat' or policy['concise_answer_format']!='boxed':
            raise ValueError('Unsupported ASC calibration prompt policy')
        for row in pairs:
            row['original_project_prompt']=row['prompt']
            row['prompt']=policy['question_template'].format(question=row['question'])
            body,marker,_=row['concise'].rpartition('Answer:')
            if not marker:raise ValueError('Reference is missing its final-answer boundary')
            row['concise']=body.rstrip()+'\n'+r'\boxed{'+str(row['answer'])+'}'
            row['concise_tokens']=len(tok.encode(row['concise'],add_special_tokens=False))
            if not verify_answer(extract_final_answer(row['concise']),row['answer']):
                raise ValueError('Boxed reference normalization changed the answer')
        sources=[dict(row,original_project_prompt=row['prompt'],
            prompt=policy['question_template'].format(question=row['question'])) for row in sources]
    texts = pq.read_table(cfg['asc']['general_parquet'], columns=['text']).column('text').to_pylist()
    # The paper names WikiText but leaves variant and sample/window policy open.
    eligible = [{'source_row':i, 'text':t} for i,t in enumerate(texts) if t and len(t.split()) >= 64]
    random.Random(cfg['seed']).shuffle(eligible)
    generic = eligible[:cfg['asc']['general_samples']]
    generic_holdout = eligible[cfg['asc']['general_samples']:][:cfg['asc']['general_holdout_samples']]
    if len(generic) != cfg['asc']['general_samples']:
        raise ValueError('Insufficient generic-text samples')
    if len(generic_holdout) != cfg['asc']['general_holdout_samples']:
        raise ValueError('Insufficient generic-text holdout')
    inputs = root/'inputs'
    write_jsonl(inputs/'pairs.jsonl', pairs)
    write_jsonl(inputs/'sources.jsonl', sources)
    write_jsonl(inputs/'generic_text.jsonl', generic)
    write_jsonl(inputs/'generic_text_holdout.jsonl', generic_holdout)
    save(inputs/'audit.json', {'pairs':len(pairs), 'development_questions':len(sources),
        'generic_samples':len(generic), 'duplicate_ids':0, 'calibration_development_overlap':0,
        'observed_development_only':True, 'student_training_authorized_from_these_dev_records':False,
        'prompt_policy':cfg.get('prompt_policy','historical_project_prompt'),
        'source_sha256':file_sha256(resolve(cfg['corpus'])),
        'generic_parquet_sha256':file_sha256(cfg['asc']['general_parquet'])})
    freeze_protocol(config_path, cfg, root)


def freeze_protocol(config_path, cfg, root, *, extra_bindings=()):
    """Seal prepared inputs and the shared method implementation for execution."""
    inputs = root/'inputs'
    cfg['result_root'] = str(root)
    prompt_path = resolve(cfg['dap']['prompt_config'])
    save(inputs/'dap_prompt.json', read_json(prompt_path))
    cfg['dap']['prompt_config'] = str(inputs/'dap_prompt.json')
    code = root/'code'
    for name in ('src','scripts','configs'):
        shutil.copytree(CODE/name, code/name,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'), symlinks=True)
    cfg['code_root'] = str(code)
    model_hashes = {}
    for role in ('teacher','tokenskip'):
        snapshot = Path(cfg[role]['snapshot_path'])
        paths = sorted(snapshot.glob('*.safetensors')) + sorted(snapshot.glob('*.json'))
        if not any(p.suffix == '.safetensors' for p in paths):
            raise ValueError(f'Missing weights: {role}')
        model_hashes[role] = evidence(paths)
    save(inputs/'model_hashes.json', model_hashes)
    packages = {n:importlib.metadata.version(n) for n in
                ('torch','transformers','tokenizers','llmlingua','nltk','tiktoken','numpy')}
    save(inputs/'runtime_versions.json', packages)
    source_files = [p for name in ('src','scripts','configs') for p in (code/name).rglob('*')
                    if p.is_file() and not p.is_symlink()]
    seal(root/'protocol/SOURCES.json', source_files)
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/FROZEN.json', [root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
        *sorted(inputs.glob('*')), *extra_bindings], source_config=str(config_path), source_config_sha256=file_sha256(config_path))
    logging.info('Frozen baseline method check: %s', root/'protocol/frozen_config.json')


def load_frozen(config_path):
    cfg = read_json(config_path)
    root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json')
    verify(root/'protocol/SOURCES.json')
    if Path(cfg['code_root']).resolve() != CODE.resolve():
        raise ValueError('Run from the frozen code snapshot')
    return cfg


def finish(cfg, out, files, **extra):
    if (out/'hardware.json').exists(): files = [*files, out/'hardware.json']
    seal(out/'COMPLETE.json', [*files, Path(cfg['result_root'])/'protocol/FROZEN.json'],
         evidence_level=cfg['evidence_level'], formal_claim_allowed=False, **extra)


def record_hardware(out):
    """Record allocated-device identity using the existing NCSU probe fields."""
    import subprocess
    import torch
    result = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv'],
                            capture_output=True, text=True, check=True)
    save(out/'hardware.json', {'job_id':os.environ['SLURM_JOB_ID'], 'inventory_csv':result.stdout,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))})


def train_asc(cfg, *, smoke=False):
    import torch
    from safetensors.torch import save_file
    from .asc_ces import ResidualAddition, pair_energies, contrastive_energy, text_kl, trust_region_penalty
    root, spec = Path(cfg['result_root']), cfg['asc']
    out = root/('asc_smoke' if smoke else 'asc')
    if (out/'COMPLETE.json').exists():
        verify(out/'COMPLETE.json'); return
    out.mkdir(parents=True, exist_ok=False)
    admission(cfg)
    record_hardware(out)
    model, tok = teacher_bundle(cfg)
    model.requires_grad_(False)
    vector = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device=model.device, dtype=torch.float32))
    controller = ResidualAddition(model.model.layers[spec['layer_index']], vector)
    optimizer = torch.optim.Adam([vector], lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    pair_path = root/'inputs/pairs.jsonl'
    pair_bindings = []
    if 'calibration_generation' in cfg:
        marker = root/'asc_calibration/COMPLETE.json'
        verify(marker)
        pair_path = root/'asc_calibration/pairs.jsonl'
        pair_bindings = [marker, pair_path]
    pairs = list(read_jsonl(pair_path))
    if len(pairs) != spec['pairs']:
        raise ValueError('ASC calibration pair count does not match protocol')
    generic = list(read_jsonl(root/'inputs/generic_text.jsonl'))
    rng = random.Random(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    steps = spec['smoke_steps'] if smoke else spec['steps']
    start = time.monotonic()
    history = []
    torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x') as log:
        for step in range(1,steps+1):
            optimizer.zero_grad(set_to_none=True)
            ces_value, short_value, long_value = 0.,0.,0.
            for row in rng.sample(pairs, spec['batch_size']):
                short, long = pair_energies(model, tok, controller, row,
                                           max_length=spec['max_sequence_length'])
                loss = contrastive_energy(short, long) / spec['batch_size']
                loss.backward()
                ces_value += float(loss.detach())
                short_value += float(short.detach()) / spec['batch_size']
                long_value += float(long.detach()) / spec['batch_size']
                del short, long, loss
            kl = torch.stack([text_kl(model, tok, controller, row['text'],
                         max_length=spec['general_max_length'])
                         for row in rng.sample(generic, spec['general_batch_size'])]).mean()
            penalty = trust_region_penalty(kl, epsilon=spec['epsilon'], coefficient=spec['kl_coefficient'])
            penalty.backward()
            if vector.grad is None or not torch.isfinite(vector.grad).all():
                raise RuntimeError('Missing or nonfinite steering gradient')
            grad_norm = float(vector.grad.norm())
            if step == 1 and grad_norm == 0:
                raise RuntimeError('Steering vector does not affect the objective')
            optimizer.step()
            if any(p.grad is not None for p in model.parameters()) or not torch.isfinite(vector).all():
                raise RuntimeError('Frozen base received gradients or vector became nonfinite')
            record = dict(step=step,ces=ces_value,short_energy=short_value,long_energy=long_value,
                kl=float(kl.detach()),penalty=float(penalty.detach()),gradient_norm=grad_norm,
                vector_norm=float(vector.detach().norm()),elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(record)+'\n');log.flush()
            history.append(record)
            if step == 1 or step % spec['log_every'] == 0 or step == steps:
                logging.info('ASC %s', record)
            del kl, penalty
    save_file({'vector':vector.detach().cpu().contiguous()}, str(out/'vector.safetensors'))
    holdout = list(read_jsonl(root/'inputs/generic_text_holdout.jsonl'))
    if smoke: holdout = holdout[:2]
    with torch.no_grad():
        final_holdout_kl = [float(text_kl(model, tok, controller, r['text'],
                                max_length=spec['general_max_length'])) for r in holdout]
    save(out/'summary.json', {'steps':steps,'requested_steps':steps,'pairs':len(pairs),
        'generic_text_samples':len(generic),'only_trainable_parameters':vector.numel(),
        'base_gradients_absent':all(p.grad is None for p in model.parameters()),
        'initial_vector_norm':0., 'final':history[-1],
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'smoke_only':smoke,'full_fit_complete':not smoke,
        'heldout_generic_kl_mean':sum(final_holdout_kl)/len(final_holdout_kl),
        'heldout_generic_kl_samples':len(final_holdout_kl),
        'kl_hinge_is_soft_penalty_not_guaranteed_constraint':True})
    finish(cfg,out,[out/'vector.safetensors',out/'summary.json',out/'training.jsonl',*pair_bindings],
           stage='asc_smoke' if smoke else 'asc_fit', steps=steps)


def generate_text(model, tok, messages, spec, seed, controller=None):
    import torch
    from contextlib import nullcontext
    from transformers import set_seed
    set_seed(seed)
    rendering = spec.get('prompt_rendering', 'native_chat')
    if rendering == 'native_chat':
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif rendering == 'raw_single_user':
        if len(messages) != 1 or messages[0]['role'] != 'user':
            raise ValueError('Raw rendering requires exactly one user message')
        prompt = messages[0]['content']
    else:
        raise ValueError('Unknown prompt rendering: '+rendering)
    # Native chat templates already contain their BOS. A raw question follows
    # the upstream tokenizer default, which adds BOS for R1-Distill-Qwen.
    add_special = bool(spec.get('raw_add_special_tokens', True)) if rendering == 'raw_single_user' else False
    inputs = tok(prompt, return_tensors='pt', add_special_tokens=add_special).to(model.device)
    if inputs['input_ids'].shape[1] + spec['max_new_tokens'] > model.config.max_position_embeddings:
        raise ValueError('Full source and output budget exceed model context')
    torch.cuda.synchronize(); start=time.monotonic()
    sampling = {'do_sample':spec.get('do_sample', True)}
    if sampling['do_sample']:
        sampling.update(temperature=spec['temperature'], top_p=spec['top_p'])
        if 'top_k' in spec:
            sampling['top_k'] = spec['top_k']
    else:
        sampling.update(temperature=None, top_p=None, top_k=None)
    with torch.inference_mode(), (controller.applied() if controller is not None else nullcontext()):
        generated=model.generate(**inputs, **sampling, repetition_penalty=spec.get('repetition_penalty',1.),
            max_new_tokens=spec['max_new_tokens'], use_cache=True,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); elapsed=time.monotonic()-start
    suffix=generated[0,inputs['input_ids'].shape[1]:].tolist()
    eos=suffix.index(tok.eos_token_id) if tok.eos_token_id in suffix else len(suffix)
    text=tok.decode(suffix[:eos],skip_special_tokens=True)
    if not spec.get('preserve_sampled_text', False):
        text=text.strip()
    result = {'solution':text,'predicted_answer':extract_final_answer(text),
        'generated_tokens':eos,'solution_token_count':len(tok.encode(text,add_special_tokens=False)),
        'prompt_tokens':inputs['input_ids'].shape[1],'latency_seconds':elapsed,
        'hit_max_new_tokens':eos==len(suffix) and len(suffix)>=spec['max_new_tokens'],'seed':seed,
        'prompt_rendering':rendering,'tokenizer_add_special_tokens':add_special}
    if spec.get('retain_token_ids', False): result['sampled_token_ids'] = suffix
    return result


def summarize_generations(rows):
    if not rows: raise ValueError('Cannot summarize empty generations')
    return {'n':len(rows),'accuracy':sum(r['is_correct'] for r in rows)/len(rows),
        'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/len(rows),
        'mean_latency_seconds':sum(r['latency_seconds'] for r in rows)/len(rows),
        'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rows)/len(rows)}


def grade_prediction(cfg, text, gold, *, source=None):
    if cfg.get('grading', {}).get('method') == 'gsm8k_explicit_answer_quantity_v3':
        from .gsm8k_grading_v3 import grade_gsm8k_response
        options = cfg['grading']
        timeout = options.get('timeout_seconds', options.get('config', {}).get('timeout_seconds', 5))
        return grade_gsm8k_response(text, gold, timeout_seconds=timeout)
    if cfg.get('grading', {}).get('method') == 'reviewed_math_v1':
        from .reviewed_math_grading import grade_reviewed_response
        if source is None or source.get('answer') != gold or 'reviewed_answer' not in source:
            raise ValueError('Reviewed mathematics grading requires matching explicit source context')
        return grade_reviewed_response(text, source, cfg['grading']['config'])
    if cfg.get('grading', {}).get('method') == 'typed_math_v2':
        from .typed_math_grading import grade_typed_response
        if source is None or source.get('answer') != gold or not source.get('dataset'):
            raise ValueError('Typed mathematics grading requires matching source context')
        return grade_typed_response(text, source, cfg['grading']['config'])
    if cfg.get('grading', {}).get('method') == 'gsm8k_math_verify_v2':
        from .gsm8k_grading import grade_gsm8k_response
        return grade_gsm8k_response(text,gold,timeout_seconds=cfg['grading']['timeout_seconds'])
    prediction=extract_final_answer(text)
    return {'predicted_answer':prediction,'is_correct':verify_answer(prediction,gold)}


def generate_calibration_pairs(cfg):
    """Obtain verbose traces from the actual ASC target model, not another teacher.

    Stop after the registered number of eligible unique questions. Preserve all
    attempts, including wrong/capped/too-short outputs and unused pool members.
    A partial directory is evidence of an interrupted attempt, never a fit input.
    """
    root=Path(cfg['result_root']);out=root/'asc_calibration'
    if (out/'COMPLETE.json').exists():verify(out/'COMPLETE.json');return
    spec=cfg['calibration_generation']
    out.mkdir(parents=True,exist_ok=False)
    import torch
    admission(cfg);record_hardware(out);model,tok=teacher_bundle(cfg)
    torch.cuda.reset_peak_memory_stats()
    pool=list(read_jsonl(root/'inputs/pairs.jsonl'))
    accepted,attempts,visited=[],[],[]
    with (out/'attempts.jsonl').open('x') as handle:
        for source in pool:
            visited.append(source['problem_id'])
            concise_tokens=len(tok.encode(source['concise'],add_special_tokens=False))
            for candidate in range(spec['max_candidates_per_question']):
                seed=int(canonical_sha256([cfg['seed'],'calibration',source['problem_id'],candidate])[:8],16)
                result=generate_text(model,tok,[{'role':'user','content':source['prompt']}],spec,seed)
                assessment=grade_prediction(cfg,result['solution'],source['answer'],source=source)
                correct=assessment['is_correct']
                verbose_tokens = result['generated_tokens'] if spec.get('preserve_sampled_text') else result['solution_token_count']
                eligible=(correct and not result['hit_max_new_tokens'] and verbose_tokens>concise_tokens
                    and result['prompt_tokens']+result['generated_tokens']+1<=cfg['asc']['max_sequence_length'])
                row={'problem_id':source['problem_id'],'candidate_index':candidate,**result,**assessment,
                     'gold_answer':source['answer'],'eligible_pair':eligible}
                attempts.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
                logging.info('ASC calibration question=%s candidate=%d tokens=%d eligible=%s pairs=%d/%d',
                    source['problem_id'],candidate,result['generated_tokens'],eligible,len(accepted)+int(eligible),cfg['asc']['pairs'])
                if eligible:
                    accepted.append({**source,'verbose':result['solution'],
                        'verbose_tokens':verbose_tokens,
                        'source_trace_id':f"target-{source['problem_id']}-{candidate}",
                        'original_teacher_source_trace_id':source['source_trace_id'],
                        'verbose_teacher':cfg['teacher']['model_name'],'verbose_teacher_revision':cfg['teacher']['revision']})
                    break
            if len(accepted)==cfg['asc']['pairs']:break
    summary={**summarize_generations(attempts),'pool_questions':len(pool),'visited_questions':len(visited),
             'accepted_pairs':len(accepted),'required_pairs':cfg['asc']['pairs'],
             'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
             'unused_pool_ids':[r['problem_id'] for r in pool if r['problem_id'] not in set(visited)],
             'selection':'first eligible candidate per question, stable pool order, stop after required unique pairs'}
    save(out/'summary.json',summary)
    if len(accepted)!=cfg['asc']['pairs'] or len({r['problem_id'] for r in accepted})!=len(accepted):
        raise ValueError('Insufficient or duplicate target-model calibration pairs; do not fit ASC')
    write_jsonl(out/'pairs.jsonl',accepted)
    finish(cfg,out,[out/'attempts.jsonl',out/'pairs.jsonl',out/'summary.json'],stage='asc_target_calibration')


def rewrite_dap(cfg):
    root=Path(cfg['result_root']);out=root/'dap'
    if (out/'COMPLETE.json').exists():verify(out/'COMPLETE.json');return
    out.mkdir(parents=True,exist_ok=False)
    admission(cfg);model,tok=teacher_bundle(cfg)
    prompt=read_json(cfg['dap']['prompt_config'])
    rows=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for source in read_jsonl(root/'inputs/sources.jsonl'):
            seed=int(canonical_sha256([cfg['seed'],source['problem_id']])[:8],16)
            result=generate_text(model,tok,dap_messages(source['question'],source['solution'],prompt),cfg['dap'],seed)
            row={'problem_id':source['problem_id'],'source_trace_id':source['trace_id'],
                 'source_tokens':source['solution_token_count'],'gold_answer':source['answer'],**result,
                 **grade_prediction(cfg,result['solution'],source['answer'],source=source),
                 **dap_structure(result['solution'])}
            rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
            logging.info('DAP question=%s tokens=%d correct=%s difficulty=%s',
                         row['problem_id'],row['generated_tokens'],row['is_correct'],row['difficulty'])
    summary=summarize_generations(rows)
    summary.update(difficulty_counts=dict(Counter(r['difficulty'] for r in rows)),
        source_mean_tokens=sum(r['source_tokens'] for r in rows)/len(rows),
        full_thought_and_solution_retained=True, observed_development_only=True)
    if len(rows)!=cfg['validation']['count'] or len({r['problem_id'] for r in rows})!=len(rows):
        raise ValueError('DAP incomplete or duplicate questions')
    save(out/'summary.json',summary)
    finish(cfg,out,[out/'predictions.jsonl',out/'summary.json'],stage='dap_rewrite')


def compress_tokenskip(cfg):
    from llmlingua import PromptCompressor
    from transformers import AutoTokenizer
    root=Path(cfg['result_root']);out=root/'tokenskip'
    if (out/'COMPLETE.json').exists():verify(out/'COMPLETE.json');return
    out.mkdir(parents=True,exist_ok=False)
    admission(cfg)
    spec=cfg['tokenskip']
    if importlib.metadata.version('llmlingua')!=spec['package_version']:
        raise ValueError('LLMLingua version changed')
    compressor=PromptCompressor(model_name=spec['snapshot_path'],use_llmlingua2=True,device_map='cuda')
    tok=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    rows,assigned=[],[]
    with (out/'compression.jsonl').open('x') as handle:
        for source in read_jsonl(root/'inputs/sources.jsonl'):
            chosen=ratio_for_problem(source['problem_id'],spec['ratios'],spec['ratio_assignment_seed'])
            for ratio in spec['ratios']:
                start=time.monotonic()
                compressed=compress_trace(compressor,source['solution'],ratio,family=spec['family'])
                completion=tokenskip_completion(compressed['compressed_prompt'],source['predicted_answer'])
                row={'problem_id':source['problem_id'],'source_trace_id':source['trace_id'],
                    'ratio':ratio,'prompt':tokenskip_prompt(source['question'],ratio),
                    'completion':completion,'compression':compressed,
                    'source_tokens':source['solution_token_count'],
                    'compressed_body_tokens':len(tok.encode(compressed['compressed_prompt'],add_special_tokens=False)),
                    'completion_tokens':len(tok.encode(completion,add_special_tokens=False)),
                    'compression_seconds':time.monotonic()-start,
                    'answer_appended_from_verified_source':True,'assigned_for_ratio_conditioning_check':ratio==chosen}
                rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
                if ratio==chosen:assigned.append(row)
            logging.info('TokenSkip compressed %s at all %d ratios',source['problem_id'],len(spec['ratios']))
    expected=cfg['validation']['count']
    if len(assigned)!=expected or len({r['problem_id'] for r in assigned})!=expected or len(rows)!=expected*len(spec['ratios']):
        raise ValueError('TokenSkip ratio/question coverage audit failed')
    write_jsonl(out/'assigned_ratio_examples_dev_only.jsonl',assigned)
    summary={'questions':expected,'ratio_records':len(rows),'assigned_records':len(assigned),
        'student_training_complete':False,'dev_examples_must_not_train_student':True,
        'appended_answer_accuracy_is_not_reasoning_quality':True,
        'ratios':{str(r):{'mean_body_retention':sum(x['compressed_body_tokens']/x['source_tokens'] for x in rows if x['ratio']==r)/expected,
                         'mean_completion_tokens':sum(x['completion_tokens'] for x in rows if x['ratio']==r)/expected}
                  for r in spec['ratios']}}
    save(out/'summary.json',summary)
    finish(cfg,out,[out/'compression.jsonl',out/'assigned_ratio_examples_dev_only.jsonl',out/'summary.json'],stage='tokenskip_compression')


def evaluate_asc(cfg, *, smoke=False):
    from safetensors.torch import load_file
    from .asc_ces import ResidualAddition
    root=Path(cfg['result_root']);stage='asc_smoke' if smoke else 'asc'
    verify(root/stage/'COMPLETE.json')
    out=root/(stage+'_generation')
    if (out/'COMPLETE.json').exists():verify(out/'COMPLETE.json');return
    out.mkdir(parents=True,exist_ok=False)
    admission(cfg);record_hardware(out);model,tok=teacher_bundle(cfg)
    vector=load_file(str(root/stage/'vector.safetensors'))['vector'].to(model.device)
    sources=list(read_jsonl(root/'inputs/sources.jsonl'))
    if smoke:sources=sources[:2]
    rows=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for source in sources:
            for scale in cfg['validation']['asc_scales']:
                controller=ResidualAddition(model.model.layers[cfg['asc']['layer_index']],vector*scale) if scale else None
                seed=int(canonical_sha256([cfg['seed'],source['problem_id']])[:8],16)
                result=generate_text(model,tok,[{'role':'user','content':source['prompt']}],cfg['validation'],seed,controller)
                row={'problem_id':source['problem_id'],'scale':scale,'gold_answer':source['answer'],**result,
                     **grade_prediction(cfg,result['solution'],source['answer'],source=source)}
                rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
                logging.info('ASC generation question=%s scale=%s tokens=%d correct=%s',
                             row['problem_id'],scale,row['generated_tokens'],row['is_correct'])
    if len(rows)!=len(sources)*len(cfg['validation']['asc_scales']) or len({(r['problem_id'],r['scale']) for r in rows})!=len(rows):
        raise ValueError('ASC generation coverage audit failed')
    save(out/'summary.json',{'smoke_only':smoke,'by_scale':{str(s):summarize_generations([r for r in rows if r['scale']==s]) for s in cfg['validation']['asc_scales']}})
    finish(cfg,out,[out/'predictions.jsonl',out/'summary.json',root/stage/'COMPLETE.json'],stage=stage+'_generation')
