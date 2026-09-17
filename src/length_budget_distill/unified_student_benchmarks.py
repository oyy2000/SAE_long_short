"""Freeze, generate and audit the complete five-benchmark student cohort."""
from collections import defaultdict
from pathlib import Path
import importlib.metadata
import json
import logging
import math
import shutil
import statistics
import time

from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import save, seal, verify, admission
from .baseline_reproduction import record_hardware
from .student_evaluation import load_student_for_evaluation
from .student_evaluation_protocol import (bound_student_model, evaluation_cell, verify_model_files)
from .student_cap_validation import summarize_cap_grid
from .student_prompts import build_unified_evaluation_prompt
from .unified_student_evaluation import generate_student_batch, audit_student_prediction

CODE = Path(__file__).resolve().parents[2]
COUNTS = {'gsm8k': 1269, 'gsm8k_hard': 1269, 'math500': 500, 'aqua_rat': 254, 'olympiadbench': 674}


def required_cells(cfg):
    phase = cfg['evaluation_stage']
    if not phase['students'] or not phase['methods'] or not phase['seeds']:
        raise ValueError('A student evaluation stage must contain complete registered cells')
    if len(set(phase['students'])) != len(phase['students']) or len(set(phase['methods'])) != len(phase['methods']):
        raise ValueError('Repeated evaluation-stage student or method')
    if len(set(phase['seeds'])) != len(phase['seeds']) or not set(phase['seeds']) <= set(cfg['student_seeds']):
        raise ValueError('Repeated or unregistered training seed')
    cells = []
    for student in phase['students']:
        for method in ['base', *phase['methods']]:
            for seed in [None] if method == 'base' else phase['seeds']:
                cells.append({'student': student, 'method': method, 'seed': seed,
                              'cell': evaluation_cell(student, method, seed, cfg)})
    return cells


def benchmark_ratios(method, cfg):
    if method == 'B6':
        return list(cfg['tokenskip_ratios'])
    return [None]


def common_development_cap(summaries, expected_cells):
    if set(summaries) != set(expected_cells) or not expected_cells:
        raise ValueError('All real student cells need completed development checks')
    eligible = set.intersection(*(set(summaries[key]['admissible_caps']) for key in expected_cells))
    if not eligible:
        raise ValueError('No common admissible development cap; preserve this outcome')
    return min(eligible)


def cap_summary_for_evaluation(prefixes, questions, method, development_cfg, cfg):
    ratios = benchmark_ratios(method, cfg)
    summary = summarize_cap_grid([r for r in prefixes if r['ratio'] in ratios], questions, ratios, development_cfg)
    return {**summary, 'evaluated_ratios': ratios}


def audit_benchmark_cohort(questions):
    ids = [q['problem_id'] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated locked benchmark problem')
    grouped = defaultdict(list)
    for row in questions:
        if row['question_role'] != 'locked_evaluation':
            raise ValueError('Training, development or smoke question in locked evaluation')
        grouped[row['dataset']].append(row)
    if {name: len(rows) for name, rows in grouped.items()} != COUNTS:
        raise ValueError('The complete registered five-benchmark cohort changed')
    if sorted(q['source_index'] for q in grouped['gsm8k']) != list(range(50, 1319)):
        raise ValueError('Locked GSM8K range changed')
    gsm_ids = {q['problem_id'] for q in grouped['gsm8k']}
    if any(not q.get('parent_problem_ids') or not set(q['parent_problem_ids']) <= gsm_ids for q in grouped['gsm8k_hard']):
        raise ValueError('GSM8K-Hard lost its locked parent mapping')


def audit_benchmark_grid(rows, questions, ratios, cap, cell):
    expected = {(q['problem_id'], ratio) for q in questions for ratio in ratios}
    actual = [(r['problem_id'], r['ratio']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing, duplicate or unexpected benchmark prediction')
    qmap = {q['problem_id']: q for q in questions}
    for row in rows:
        source = qmap[row['problem_id']]
        if (row['cell'] != cell or row['max_new_tokens'] != cap or
                row['dataset'] != source['dataset'] or row['question_role'] != 'locked_evaluation' or
                row['source_record_sha256'] != canonical_sha256(source) or
                row.get('parent_problem_ids') != source.get('parent_problem_ids')):
            raise ValueError('Benchmark cell, source, parent mapping or budget changed')


def audit_batch_timings(rows, batches):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['batch_id']].append(row)
    if len(batches) != len({b['batch_id'] for b in batches}) or set(grouped) != {b['batch_id'] for b in batches}:
        raise ValueError('Missing or duplicate generation timing batch')
    for batch in batches:
        values = grouped[batch['batch_id']]
        if [r['problem_id'] for r in values] != batch['problem_ids'] or any(r['ratio'] != batch['ratio'] for r in values):
            raise ValueError('Batch timing has different questions or ratio')
        seconds = batch['generation_wall_seconds']
        if not math.isfinite(seconds) or seconds <= 0 or any(
                not math.isfinite(r['amortized_generation_wall_seconds']) or
                abs(r['amortized_generation_wall_seconds'] - seconds/len(values)) > 1e-9 for r in values):
            raise ValueError('Generation time was duplicated or assigned inconsistently')


def prepare(config_path):
    from transformers import AutoConfig, AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg['result_root']); launch = Path(cfg['launch_root'])
    verify(launch/'FROZEN.json')
    if CODE != Path(cfg['code_root']) or root.exists():
        raise ValueError('Use a new evaluation root and its frozen preparation source')
    if cfg.get('max_new_tokens') is not None:
        raise ValueError('Resolve the common cap from completed development cells')
    cells = required_cells(cfg); development = Path(cfg['development_root'])
    markers = [launch/'FROZEN.json', development/'protocol/FROZEN.json', development/'protocol/SOURCES.json']
    for marker in markers:
        verify(marker)
    dcfg = read_json(development/'protocol/frozen_config.json')
    for key in ('students', 'sft_root', 'sft_interface_root', 'question_template', 'choice_instruction',
                'tokenskip_ratios', 'grading', 'grading_methods', 'repetition_penalty',
                'torch_dtype', 'attention_implementation', 'batch_size', 'cohort_root'):
        if cfg[key] != dcfg[key]:
            raise ValueError('Formal evaluation differs from the actual development decoder: '+key)
    training = read_json(Path(cfg['sft_root'])/'protocol/frozen_config.json')
    if (cfg['evaluation_stage']['students'] != training['primary_stage']['students'] or
            cfg['evaluation_stage']['methods'] != training['primary_stage']['methods'] or
            cfg['evaluation_stage']['seeds'] != training['student_seeds']):
        raise ValueError('Evaluation stage differs from its registered training stage')
    development_questions = list(read_jsonl(development/'inputs/questions.jsonl'))
    bindings = {}; cap_summaries = {}
    for cell in cells:
        student, method, seed, key = (cell[k] for k in ('student', 'method', 'seed', 'cell'))
        binding = bound_student_model(student, method, seed, cfg)
        directory = development/'cells'/key; marker = directory/'COMPLETE.json'; verify(marker); markers.append(marker)
        if read_json(directory/'model_binding.json') != binding:
            raise ValueError('The completed development run used another model/adapter')
        from .student_evaluation_protocol import evaluation_ratios
        prefixes = list(read_jsonl(directory/'cap_predictions.jsonl'))
        summary = summarize_cap_grid(prefixes, development_questions,
                                     evaluation_ratios(method, dcfg), dcfg)
        saved = read_json(directory/'summary.json')
        if any(saved[k] != v for k, v in summary.items()) or saved['formal_benchmark_predictions'] != 0:
            raise ValueError('Development cap summary differs from its complete predictions')
        # Use exactly the prompts this cell will receive in the main evaluation.
        # Extra base ratio prompts remain diagnostics; B6 retains every ratio.
        cap_summaries[key] = cap_summary_for_evaluation(prefixes, development_questions, method, dcfg, cfg)
        bindings[key] = binding
        markers.extend(map(Path, binding['markers']))
    cap = common_development_cap(cap_summaries, [c['cell'] for c in cells])
    cohorts = Path(cfg['cohort_root']); preflight = Path(cfg['prompt_preflight_root'])
    verify(cohorts/'COMPLETE.json'); preflight_doc = verify(preflight/'COMPLETE.json')
    markers += [cohorts/'COMPLETE.json', preflight/'COMPLETE.json']
    questions = []
    for name in COUNTS:
        path = cohorts/'evaluation'/(name+'.jsonl')
        if preflight_doc['hashes'].get(str(path.resolve())) != file_sha256(path):
            raise ValueError('Benchmark source differs from its completed prompt preflight')
        questions.extend(read_jsonl(path)); markers.append(path)
    audit_benchmark_cohort(questions)
    versions = read_json(development/'inputs/runtime_versions.json')
    if any(importlib.metadata.version(k) != v for k, v in versions.items()):
        raise ValueError('Formal student runtime differs from development')
    for student in cfg['evaluation_stage']['students']:
        spec = cfg['students'][student]; tok = AutoTokenizer.from_pretrained(spec['snapshot_path'], local_files_only=True)
        context = AutoConfig.from_pretrained(spec['snapshot_path'], local_files_only=True).max_position_embeddings
        for question in questions:
            for ratio in [None, *cfg['tokenskip_ratios']]:
                prompt = build_unified_evaluation_prompt(question, cfg, ratio=ratio)
                rendered = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                if len(tok.encode(rendered, add_special_tokens=False)) + cap > context:
                    raise ValueError('Locked benchmark input exceeds the chosen common context budget')
    if not isinstance(cfg['shards'], int) or not 1 <= cfg['shards'] <= len(questions):
        raise ValueError('Invalid evaluation shard count')
    root.mkdir(parents=True)
    write_jsonl(root/'inputs/questions.jsonl', questions)
    save(root/'inputs/model_bindings.json', bindings); save(root/'inputs/runtime_versions.json', versions)
    save(root/'inputs/development_cap_decision.json', {'by_cell': cap_summaries, 'common_cap': cap,
         'selection_uses_test_predictions': False, 'later_stage_budget_parity_requires_separate_audit': True})
    save(root/'inputs/shards.json', {'shards': {str(i): [q['problem_id'] for q in questions[i::cfg['shards']]]
                                               for i in range(cfg['shards'])}})
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(CODE/folder, root/'code'/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'), symlinks=True)
    cfg.update(code_root=str(root/'code'), max_new_tokens=cap, cells=cells)
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'protocol/FROZEN.json', [Path(config_path), *markers, root/'protocol/frozen_config.json',
         root/'protocol/SOURCES.json', *sorted((root/'inputs').glob('*'))], stage='complete_student_benchmark_inputs',
         evaluation_complete=False)


def load(config_path):
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']):
        raise ValueError('Use the frozen benchmark worker')
    if any(importlib.metadata.version(k) != v for k, v in read_json(root/'inputs/runtime_versions.json').items()):
        raise ValueError('Frozen student benchmark runtime changed')
    return cfg


def run(config_path, student, method, seed, shard):
    import torch
    cfg = load(config_path); root = Path(cfg['result_root']); key = evaluation_cell(student, method, seed, cfg)
    if key not in {c['cell'] for c in cfg['cells']} or not 0 <= shard < cfg['shards']:
        raise ValueError('Unregistered benchmark cell or shard')
    binding = bound_student_model(student, method, seed, cfg)
    if binding != read_json(root/'inputs/model_bindings.json')[key]:
        raise ValueError('Benchmark adapter binding changed')
    before = time.monotonic(); verify_model_files(binding); hash_seconds = time.monotonic()-before
    questions = list(read_jsonl(root/'inputs/questions.jsonl'))[shard::cfg['shards']]
    expected_ids = read_json(root/'inputs/shards.json')['shards'][str(shard)]
    if [q['problem_id'] for q in questions] != expected_ids:
        raise ValueError('Benchmark shard membership changed')
    out = root/'generation'/key/f'shard_{shard:02d}'; out.mkdir(parents=True, exist_ok=False)
    admission(cfg); record_hardware(out)
    if cfg['expected_gpu_name'] not in torch.cuda.get_device_name(0):
        raise ValueError('Benchmark latency comparison requires the registered GPU model')
    before = time.monotonic()
    bundle = load_student_for_evaluation({'model_name': binding['model']['snapshot_path'],
                'torch_dtype': cfg['torch_dtype'], 'attn_implementation': cfg['attention_implementation']},
                adapter_path=binding['adapter_path'])
    load_seconds = time.monotonic()-before; model = bundle['model']; tok = bundle['tokenizer']
    if bool(getattr(model, 'peft_config', None)) != (method != 'base'):
        raise ValueError('The registered real adapter is not attached')
    ratios = benchmark_ratios(method, cfg); rows = []; batches = []; torch.cuda.reset_peak_memory_stats()
    with (out/'predictions.jsonl').open('x') as handle:
        for ratio in ratios:
            for offset in range(0, len(questions), cfg['batch_size']):
                batch = questions[offset:offset+cfg['batch_size']]
                values, seconds = generate_student_batch(bundle, batch, cfg, ratio=ratio, max_new_tokens=cfg['max_new_tokens'])
                batch_id = f'{key}/{shard}/{ratio}/{offset}'
                batches.append({'batch_id': batch_id, 'ratio': ratio, 'problem_ids': [q['problem_id'] for q in batch],
                                'generation_wall_seconds': seconds})
                for row, question in zip(values, batch):
                    row.update(cell=key, shard=shard, batch_id=batch_id,
                               parent_problem_ids=question.get('parent_problem_ids'),
                               amortized_generation_wall_seconds=seconds/len(batch))
                    audit_student_prediction(row, question, cfg, tok)
                    rows.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
                logging.info('Student benchmark %s shard=%d ratio=%s completed=%d/%d', key, shard, ratio,
                             offset+len(batch), len(questions))
    audit_benchmark_grid(rows, questions, ratios, cfg['max_new_tokens'], key); audit_batch_timings(rows, batches)
    write_jsonl(out/'batches.jsonl', batches)
    save(out/'summary.json', {'cell': key, 'shard': shard, 'predictions': len(rows),
         'generation_wall_seconds': sum(b['generation_wall_seconds'] for b in batches),
         'model_file_verification_seconds': hash_seconds, 'model_loading_seconds': load_seconds,
         'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
         'model_default_repetition_penalty': model.generation_config.repetition_penalty,
         'effective_repetition_penalty': cfg['repetition_penalty'], 'actual_model_class': type(model).__name__,
         'scope': 'One audited shard, not a completed model evaluation. Batch-amortized time is not isolated single-query latency.'})
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', *map(Path, binding['markers']), *sorted(out.glob('*'))],
         stage='student_benchmark_shard', full_evaluation_complete=False)


def merge(config_path, student, method, seed):
    from transformers import AutoTokenizer
    cfg = load(config_path); root = Path(cfg['result_root']); key = evaluation_cell(student, method, seed, cfg)
    if key not in {c['cell'] for c in cfg['cells']}:
        raise ValueError('Unregistered evaluation cell')
    out = root/'merged'/key
    if out.exists():
        raise FileExistsError(out)
    questions = list(read_jsonl(root/'inputs/questions.jsonl')); audit_benchmark_cohort(questions)
    tok = AutoTokenizer.from_pretrained(cfg['students'][student]['snapshot_path'], local_files_only=True)
    rows = []; batches = []; manifests = []; markers = []
    ratios = benchmark_ratios(method, cfg)
    for shard in range(cfg['shards']):
        directory = root/'generation'/key/f'shard_{shard:02d}'; marker = directory/'COMPLETE.json'; doc = verify(marker)
        values = list(read_jsonl(directory/'predictions.jsonl')); groups = list(read_jsonl(directory/'batches.jsonl'))
        sources = questions[shard::cfg['shards']]; lookup = {q['problem_id']: q for q in sources}
        audit_benchmark_grid(values, sources, ratios, cfg['max_new_tokens'], key); audit_batch_timings(values, groups)
        if any(row['shard'] != shard for row in values):
            raise ValueError('Benchmark row came from a different shard')
        for row in values:
            audit_student_prediction(row, lookup[row['problem_id']], cfg, tok)
        hardware = read_json(directory/'hardware.json')
        if cfg['expected_gpu_name'] not in hardware['inventory_csv']:
            raise ValueError('A shard used another GPU model')
        manifests.append({'shard': shard, 'marker': str(marker), 'marker_sha256': file_sha256(marker),
                          'hashes': doc['hashes'], 'records': len(values)})
        markers.append(marker); rows.extend(values); batches.extend(groups)
    audit_benchmark_grid(rows, questions, ratios, cfg['max_new_tokens'], key); audit_batch_timings(rows, batches)
    metrics = {}
    for dataset in COUNTS:
        metrics[dataset] = {}
        for ratio in ratios:
            values = [r for r in rows if r['dataset'] == dataset and r['ratio'] == ratio]
            metrics[dataset][str(ratio)] = {'questions': len(values),
                'correct': sum(r['grade']['is_correct'] for r in values),
                'accuracy': statistics.mean(float(r['grade']['is_correct']) for r in values),
                'mean_generated_tokens': statistics.mean(r['output_tokens'] for r in values),
                'cap_hit_rate': statistics.mean(float(r['hit_max_new_tokens']) for r in values),
                'mean_batch_amortized_seconds': statistics.mean(r['amortized_generation_wall_seconds'] for r in values)}
    out.mkdir(parents=True); write_jsonl(out/'predictions.jsonl', rows); write_jsonl(out/'batches.jsonl', batches)
    save(out/'shard_manifest.json', {'shards': manifests})
    save(out/'summary.json', {'cell': key, 'predictions': len(rows), 'metrics': metrics,
         'generation_wall_seconds': sum(b['generation_wall_seconds'] for b in batches),
         'bootstrap_analysis_complete': False, 'full_research_matrix_complete': False,
         'scope': cfg['claim_boundary']})
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', *markers, *sorted(out.glob('*'))],
         stage='complete_single_student_benchmark_grid', research_matrix_complete=False)
