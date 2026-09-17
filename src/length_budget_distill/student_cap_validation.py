"""Validate development budgets with the production student decoder."""
from pathlib import Path
import importlib.metadata
import json
import logging
import shutil
import time

from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import save, seal, verify, admission
from .baseline_reproduction import record_hardware
from .math_cap_calibration import cap_view
from .reviewed_math_cohort import ADDED_FIELDS, unique_index
from .student_evaluation import load_student_for_evaluation
from .student_evaluation_protocol import (bound_student_model, evaluation_cell,
                                          evaluation_ratios, verify_model_files)
from .student_prompts import build_unified_evaluation_prompt
from .unified_student_evaluation import (generate_student_batch, grade_student_prediction,
                                         audit_student_prediction)

CODE = Path(__file__).resolve().parents[2]


def audit_cap_grid(rows, questions, ratios, caps):
    ids = [q['problem_id'] for q in questions]
    if not ids or len(ids) != len(set(ids)) or any(q['question_role'] != 'development' for q in questions):
        raise ValueError('Only a unique development cohort can select student budgets')
    expected = {(pid, ratio, cap) for pid in ids for ratio in ratios for cap in caps}
    actual = [(r['problem_id'], r['ratio'], r['max_new_tokens']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete, duplicated or unexpected development cap grid')
    if any(r['question_role'] != 'development' for r in rows):
        raise ValueError('Held-out predictions cannot select the development budget')


def summarize_cap_grid(rows, questions, ratios, cfg):
    audit_cap_grid(rows, questions, ratios, cfg['caps'])
    summaries = {}
    for ratio in ratios:
        by_cap = {}
        for cap in cfg['caps']:
            values = [r for r in rows if r['ratio'] == ratio and r['max_new_tokens'] == cap]
            n = len(values)
            by_cap[str(cap)] = {'questions': n, 'accuracy': sum(r['grade']['is_correct'] for r in values)/n,
                                'mean_output_tokens': sum(r['output_tokens'] for r in values)/n,
                                'cap_hit_rate': sum(r['hit_max_new_tokens'] for r in values)/n}
        summaries[str(ratio)] = by_cap
    eligible = [cap for cap in cfg['caps'] if all(
        summary[str(cap)]['cap_hit_rate'] <= cfg['proposal_maximum_cap_hit_rate']
        and summary[str(cap)]['accuracy'] >= summary[str(max(cfg['caps']))]['accuracy']
            - cfg['proposal_maximum_accuracy_drop'] - 1e-12 for summary in summaries.values())]
    return {'by_ratio_and_cap': summaries, 'admissible_caps': eligible,
            'proposed_cap': min(eligible) if eligible else None,
            'all_registered_ratios_must_pass': True,
            'proposal_only_not_final_evaluation_protocol': True}


def prefix_prediction(full, source, cap, cfg, tokenizer):
    view = cap_view(full['sampled_token_ids'], cap, tokenizer.eos_token_id)
    text = tokenizer.decode(view['body_token_ids'], skip_special_tokens=True).strip()
    return {**full, 'parent_full_prediction_sha256': canonical_sha256(full),
            'sampled_token_ids': view['sampled_token_ids'], 'token_ids': view['body_token_ids'],
            'output_tokens': view['generated_tokens'], 'hit_max_new_tokens': view['hit_max_new_tokens'],
            'max_new_tokens': cap, 'prediction_text': text, 'grade': grade_student_prediction(text, source, cfg),
            'offline_prefix_only': True, 'prefix_latency_measured': False}


def prepare(config_path):
    from transformers import AutoConfig, AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    launch = Path(cfg['launch_root']); verify(launch/'FROZEN.json')
    if CODE != Path(cfg['code_root']) or root.exists():
        raise ValueError('Use a new result root and its frozen preparation source')
    if cfg['caps'] != sorted(set(cfg['caps'])) or min(cfg['caps']) <= 0:
        raise ValueError('Caps must be positive, unique and increasing')
    previous = Path(cfg['reference_cap_root']); cohorts = Path(cfg['cohort_root'])
    preflight = Path(cfg['prompt_preflight_root']); interface = Path(cfg['sft_interface_root'])
    markers = [launch/'FROZEN.json', previous/'protocol/FROZEN.json', cohorts/'COMPLETE.json',
               preflight/'COMPLETE.json', interface/'protocol/FROZEN.json', interface/'protocol/SOURCES.json']
    for marker in markers:
        verify(marker)
    old_cfg = read_json(previous/'protocol/frozen_config.json')
    for key in ('caps', 'proposal_maximum_cap_hit_rate', 'proposal_maximum_accuracy_drop'):
        if cfg[key] != old_cfg[key]:
            raise ValueError('The previously registered cap proposal rule changed')
    prompt_cfg = read_json(cfg['prompt_preflight_config'])
    for key in ('question_template', 'choice_instruction', 'tokenskip_ratios', 'cohort_root'):
        if cfg[key] != prompt_cfg[key]:
            raise ValueError('Student prompt/source differs from the completed preflight: '+key)
    if str(Path(cfg['prompt_preflight_config']).resolve()) not in verify(preflight/'COMPLETE.json')['hashes']:
        raise ValueError('Prompt config is not bound by the completed preflight')
    originals = list(read_jsonl(previous/'inputs/questions.jsonl'))
    reviewed = unique_index(list(read_jsonl(cohorts/'cohorts/development.jsonl')), 'student development')
    questions = []
    for old in originals:
        row = reviewed[old['problem_id']]
        if any(old[k] != value for k, value in row.items() if k not in ADDED_FIELDS):
            raise ValueError('Original development problem fields changed')
        questions.append(row)
    if len(questions) != cfg['development_questions'] or len(questions) != old_cfg['development_questions']:
        raise ValueError('The fixed student-development cohort changed')
    if any(q['question_role'] != 'development' or q['dataset'] != 'math_train' for q in questions):
        raise ValueError('Wrong development role or dataset')
    base = read_json(interface/'protocol/frozen_config.json')
    if cfg['students'] != base['students']:
        raise ValueError('Student models differ from the verified interface')
    versions = read_json(interface/'inputs/versions.json')
    if any(importlib.metadata.version(k) != v for k, v in versions.items()):
        raise ValueError('Student runtime differs from its tested interface')
    inputs = {}
    for student, spec in cfg['students'].items():
        tok = AutoTokenizer.from_pretrained(spec['snapshot_path'], local_files_only=True)
        context = AutoConfig.from_pretrained(spec['snapshot_path'], local_files_only=True).max_position_embeddings
        lengths = []
        for question in questions:
            for ratio in evaluation_ratios('base', cfg):
                text = build_unified_evaluation_prompt(question, cfg, ratio=ratio)
                rendered = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
                lengths.append(len(tok.encode(rendered, add_special_tokens=False)))
        if max(lengths) + max(cfg['caps']) > context:
            raise ValueError('Development input and output exceed the model context')
        inputs[student] = {'native_prompt_variants': len(lengths), 'maximum_prompt_tokens': max(lengths),
                           'model_context_tokens': context, 'context_overflows': 0}
    root.mkdir(parents=True)
    write_jsonl(root/'inputs/questions.jsonl', questions)
    save(root/'inputs/context.json', inputs); save(root/'inputs/runtime_versions.json', versions)
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(CODE/folder, root/'code'/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'), symlinks=True)
    cfg['code_root'] = str(root/'code')
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'protocol/FROZEN.json', [Path(config_path), *markers, previous/'inputs/questions.jsonl',
         root/'protocol/frozen_config.json', root/'protocol/SOURCES.json', *sorted((root/'inputs').glob('*'))],
         stage='unified_student_development_cap_inputs', formal_evaluation_ready=False)


def run(config_path, student, method, seed):
    import torch
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']):
        raise ValueError('Use the frozen student development decoder')
    versions = read_json(root/'inputs/runtime_versions.json')
    if any(importlib.metadata.version(k) != v for k, v in versions.items()):
        raise ValueError('Student evaluation runtime changed')
    binding = bound_student_model(student, method, seed, cfg)
    before = time.monotonic(); verify_model_files(binding); hash_seconds = time.monotonic()-before
    key = evaluation_cell(student, method, seed, cfg); out = root/'cells'/key
    out.mkdir(parents=True, exist_ok=False); admission(cfg); record_hardware(out)
    save(out/'model_binding.json', binding)
    before = time.monotonic()
    bundle = load_student_for_evaluation({'model_name': binding['model']['snapshot_path'],
                'torch_dtype': cfg['torch_dtype'], 'attn_implementation': cfg['attention_implementation']},
                adapter_path=binding['adapter_path'])
    load_seconds = time.monotonic()-before
    model = bundle['model']; tok = bundle['tokenizer']
    if bool(getattr(model, 'peft_config', None)) != (method != 'base'):
        raise ValueError('Actual adapter attachment differs from the registered cell')
    questions = list(read_jsonl(root/'inputs/questions.jsonl')); ratios = evaluation_ratios(method, cfg)
    full = []; prefixes = []; batches = []; direct = []; torch.cuda.reset_peak_memory_stats()
    with (out/'full_predictions.jsonl').open('x') as handle:
        for ratio in ratios:
            for offset in range(0, len(questions), cfg['batch_size']):
                batch = questions[offset:offset+cfg['batch_size']]
                rows, seconds = generate_student_batch(bundle, batch, cfg, ratio=ratio, max_new_tokens=max(cfg['caps']))
                batch_id = f'{ratio}/{offset}'
                batches.append({'batch_id': batch_id, 'ratio': ratio, 'problem_ids': [q['problem_id'] for q in batch],
                                'generation_wall_seconds': seconds})
                for row, source in zip(rows, batch):
                    row.update(cell=key, batch_id=batch_id)
                    audit_student_prediction(row, source, cfg, tok)
                    full.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
                    for cap in cfg['caps']:
                        view = prefix_prediction(row, source, cap, cfg, tok)
                        audit_student_prediction(view, source, cfg, tok); prefixes.append(view)
                if offset == 0:
                    check, elapsed = generate_student_batch(bundle, batch, cfg, ratio=ratio,
                                                             max_new_tokens=cfg['direct_check_cap'])
                    for row, actual in zip(rows, check):
                        if actual['sampled_token_ids'] != row['sampled_token_ids'][:cfg['direct_check_cap']]:
                            raise ValueError('Direct shorter decode differs from its saved prefix')
                    direct.append({'ratio': ratio, 'problem_ids': [q['problem_id'] for q in batch],
                                   'cap': cfg['direct_check_cap'], 'generation_wall_seconds': elapsed,
                                   'identical_sampled_prefixes': True})
                logging.info('Student cap %s ratio=%s completed=%d/%d', key, ratio, offset+len(batch), len(questions))
    audit_cap_grid(full, questions, ratios, [max(cfg['caps'])])
    summary = summarize_cap_grid(prefixes, questions, ratios, cfg)
    write_jsonl(out/'cap_predictions.jsonl', prefixes); write_jsonl(out/'batches.jsonl', batches)
    save(out/'direct_prefix_checks.json', {'checks': direct})
    save(out/'summary.json', {**summary, 'cell': key, 'full_predictions': len(full), 'prefix_predictions': len(prefixes),
         'model_hash_verification_seconds': hash_seconds, 'model_loading_seconds': load_seconds,
         'full_generation_seconds': sum(b['generation_wall_seconds'] for b in batches),
         'direct_check_generation_seconds': sum(b['generation_wall_seconds'] for b in direct),
         'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
         'model_default_repetition_penalty': model.generation_config.repetition_penalty,
         'effective_repetition_penalty': cfg['repetition_penalty'], 'actual_model_class': type(model).__name__,
         'formal_benchmark_predictions': 0, 'scope': cfg['claim_boundary']})
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', root/'protocol/SOURCES.json',
         *map(Path, binding['markers']), *sorted(out.glob('*'))],
         stage='unified_student_development_cap_validation', formal_evaluation_complete=False)
