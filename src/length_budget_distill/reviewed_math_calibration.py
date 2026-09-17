"""Rebuild ASC calibration with reviewed answers and exact saved-prefix replay.

Reuse the published-method implementation and the original reference-pool,
prompt, sampling, length and first-eligible rules. Missing necessary attempts
are errors, never permission to skip a question or substitute a later trace.
"""
from pathlib import Path
import logging
import shutil

from .baseline_reproduction import freeze_protocol, grade_prediction, ordered, summarize_generations
from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .math_baseline_calibration import make_reference_pool, validate_roles
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .reviewed_math_cohort import ADDED_FIELDS, unique_index

CODE = Path(__file__).resolve().parents[2]


def replay_saved_prefix(pool, attempts, *, pairs, max_candidates):
    if len({r['problem_id'] for r in pool}) != len(pool):
        raise ValueError('Repeated reference-pool identity')
    indexed = {}
    for row in attempts:
        key = (row['problem_id'], row['candidate_index'])
        if key in indexed or not 0 <= row['candidate_index'] < max_candidates:
            raise ValueError('Repeated or unregistered saved candidate')
        indexed[key] = row
    chosen, consumed, visited = [], [], []
    for source in pool:
        pid = source['problem_id']
        visited.append(pid)
        for index in range(max_candidates):
            key = (pid, index)
            if key not in indexed:
                raise ValueError(f'Missing necessary saved candidate {key}; generation required before fitting')
            row = indexed[key]
            consumed.append(row)
            if row['eligible_pair']:
                chosen.append((source, row))
                break
        if len(chosen) == pairs:
            return chosen, consumed, visited
    raise ValueError('Reference pool cannot provide the required unique pairs')


def validate_saved_attempt(row, source, cfg, tokenizer):
    if row['gold_answer'] != source['answer']:
        raise ValueError('Saved candidate binds a different original answer')
    expected_seed = int(canonical_sha256([cfg['seed'], 'calibration', row['problem_id'], row['candidate_index']])[:8], 16)
    if row['seed'] != expected_seed:
        raise ValueError('Saved calibration random stream differs')
    if row['prompt_rendering'] != 'native_chat' or row['tokenizer_add_special_tokens'] is not False:
        raise ValueError('Saved calibration rendering differs')
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': source['prompt']}],
                                          tokenize=False, add_generation_prompt=True)
    if row['prompt_tokens'] != len(tokenizer.encode(prompt, add_special_tokens=False)):
        raise ValueError('Saved calibration prompt token count differs')
    tokens = row['sampled_token_ids']
    eos = tokenizer.eos_token_id
    ended = bool(tokens and tokens[-1] == eos)
    cap = cfg['calibration_generation']['max_new_tokens']
    if (not tokens or len(tokens) > cap or eos in tokens[:-1] or
            (not ended and len(tokens) != cap) or row['hit_max_new_tokens'] == ended):
        raise ValueError('Saved candidate has invalid cap/EOS boundaries')
    if tokenizer.decode(tokens, skip_special_tokens=True).strip() != row['solution']:
        raise ValueError('Saved candidate text differs from its token IDs')
    if (row['generated_tokens'] != len(tokens)-int(ended) or
            row['solution_token_count'] != len(tokenizer.encode(row['solution'], add_special_tokens=False))):
        raise ValueError('Saved candidate token accounting differs')


def prepare(config_path):
    from transformers import AutoTokenizer
    rcfg = read_json(config_path)
    if CODE != Path(rcfg['code_root']):
        raise ValueError('Use frozen calibration-rebuild source')
    root, parent, cohort, impact = map(Path, (rcfg['result_root'], rcfg['parent_root'], rcfg['cohort_root'], rcfg['impact_root']))
    launch_marker = Path(rcfg['launch_root'])/'FROZEN.json'
    markers = [launch_marker, parent/'protocol/FROZEN.json', parent/'protocol/SOURCES.json',
               parent/'asc_calibration/COMPLETE.json', cohort/'COMPLETE.json', impact/'COMPLETE.json',
               Path(rcfg['impact_review_root'])/'COMPLETE.json']
    for marker in markers:
        verify(marker)
    cfg = read_json(parent/'protocol/frozen_config.json')
    original_cfg = read_json(parent/'protocol/frozen_config.json')
    cfg.update(experiment_name=rcfg['experiment_name'], result_root=str(root),
               math_cohorts=str(cohort), grading={'method': 'reviewed_math_v1', 'config': read_json(CODE/rcfg['grading_config'])},
               grading_repair_parent=str(parent), grading_repair_reason=rcfg['reason'],
               allowed_stages=['asc-smoke', 'asc-train', 'asc-smoke-eval', 'asc-eval'])
    cfg['runtime'].update(rcfg['fit_runtime_overlay'])
    cfg['asc']['verbose_source'] = original_cfg['asc']['verbose_source']+' Saved immutable candidates replayed under reviewed answer definitions; no new generation.'
    root.mkdir(parents=True, exist_ok=False)
    roles = {role: list(read_jsonl(cohort/'cohorts'/f'{role}.jsonl')) for role in
             ('calibration', 'development', 'student_pool', 'dap_development_reserved')}
    validate_roles(*(roles[k] for k in ('calibration', 'development', 'student_pool', 'dap_development_reserved')))
    if len(roles['calibration']) != cfg['calibration_generation']['pool_size']:
        raise ValueError('Original calibration role count changed')
    tokenizer = AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'], local_files_only=True)
    pool, excluded = make_reference_pool(roles['calibration'], tokenizer, cfg)
    old_pool = list(read_jsonl(parent/'inputs/pairs.jsonl'))
    old_pool_index = unique_index(old_pool, 'original calibration pool')
    cal_index = unique_index(roles['calibration'], 'reviewed calibration role')
    # Preserve the complete source row, not merely the question text or gold.
    for old in old_pool:
        source = cal_index[old['problem_id']]
        for key, value in source.items():
            if key not in ADDED_FIELDS and old[key] != value:
                raise ValueError('Reviewed source differs from original calibration input: '+key)
    attempts = list(read_jsonl(parent/'asc_calibration/attempts.jsonl'))
    impact_rows = list(read_jsonl(impact/'asc_calibration_attempts.jsonl'))
    if len(attempts) != len(impact_rows) or len(attempts) != rcfg['expected_parent_attempts']:
        raise ValueError('Incomplete saved calibration or impact rows')
    regraded, changes = [], []
    admitted = {r['problem_id'] for r in pool}
    for old, assessment in zip(attempts, impact_rows):
        if canonical_sha256(old) != assessment['source_record_sha256']:
            raise ValueError('Impact score binds a different saved candidate')
        source = {**old_pool_index[old['problem_id']], **cal_index[old['problem_id']]}
        validate_saved_attempt(old, source, cfg, tokenizer)
        grade = grade_prediction(cfg, old['solution'], old['gold_answer'], source=source)
        if grade != assessment['reviewed_grading']:
            raise ValueError('Calibration grade differs from completed impact audit')
        length_ok = (not old['hit_max_new_tokens'] and old['solution_token_count'] > source['concise_tokens'] and
                     old['prompt_tokens']+old['generated_tokens']+1 <= cfg['asc']['max_sequence_length'])
        if old['eligible_pair'] != bool(old['is_correct'] and length_ok):
            raise ValueError('Original eligibility cannot be reconstructed')
        row = {**old, **grade, 'eligible_pair': bool(grade['is_correct'] and length_ok and old['problem_id'] in admitted),
               'legacy_is_correct': old['is_correct'], 'legacy_predicted_answer': old['predicted_answer'],
               'legacy_eligible_pair': old['eligible_pair'], 'parent_record_sha256': canonical_sha256(old)}
        regraded.append(row)
        if grade['is_correct'] != old['is_correct']:
            changes.append(row)
    chosen, consumed, visited = replay_saved_prefix(pool, regraded, pairs=cfg['asc']['pairs'],
                                                   max_candidates=cfg['calibration_generation']['max_candidates_per_question'])
    pairs = [{**source, 'verbose': row['solution'], 'verbose_tokens': row['solution_token_count'],
              'source_trace_id': f"target-{row['problem_id']}-{row['candidate_index']}",
              'original_teacher_source_trace_id': source['source_trace_id'],
              'verbose_teacher': cfg['teacher']['model_name'], 'verbose_teacher_revision': cfg['teacher']['revision'],
              'reused_parent_candidate_sha256': row['parent_record_sha256']} for source, row in chosen]
    old_sources = list(read_jsonl(parent/'inputs/sources.jsonl'))
    sources = [{**r, 'prompt': cfg['prompt_policy']['question_template'].format(question=r['question'])}
               for r in ordered(roles['development'], cfg['validation']['subset_seed'])[:cfg['validation']['count']]]
    if len(sources) != len(old_sources):
        raise ValueError('Incomplete development sources')
    for new, old in zip(sources, old_sources):
        if {k: v for k, v in new.items() if k not in ADDED_FIELDS} != old:
            raise ValueError('Development source order, prompt or original fields changed')
    inputs = root/'inputs'
    write_jsonl(inputs/'pairs.jsonl', pool)
    write_jsonl(inputs/'sources.jsonl', sources)
    write_jsonl(inputs/'reference_exclusions.jsonl', excluded)
    for name, expected in [('generic_text.jsonl', cfg['asc']['general_samples']),
                           ('generic_text_holdout.jsonl', cfg['asc']['general_holdout_samples'])]:
        original = parent/'inputs'/name
        if len(list(read_jsonl(original))) != expected:
            raise ValueError('Incomplete generic-text input')
        shutil.copyfile(original, inputs/name)
        if file_sha256(original) != file_sha256(inputs/name):
            raise ValueError('Generic text changed during copy')
    old_pairs = list(read_jsonl(parent/'asc_calibration/pairs.jsonl'))
    old_keys, new_keys = {r['source_trace_id'] for r in old_pairs}, {r['source_trace_id'] for r in pairs}
    summary = {**summarize_generations(consumed), 'parent_attempt_records': len(regraded),
               'replay_consumed_attempts': len(consumed), 'replay_visited_questions': len(visited),
               'reference_pool_questions': len(pool), 'reference_exclusions': len(excluded),
               'accepted_pairs': len(pairs), 'added_trace_ids': sorted(new_keys-old_keys),
               'removed_trace_ids': sorted(old_keys-new_keys), 'grading_changed_parent_records': len(changes),
               'new_generation_calls': 0, 'all_required_prefix_candidates_present': True,
               'selection': 'Full reviewed reference pool, stable original question order, first eligible candidate, exact stop at 100 pairs.',
               'unused_parent_attempts_retained_separately': True, 'historical_evidence_modified': False,
               'direction_refit_complete': False, 'training_release': False}
    if summary['added_trace_ids'] != rcfg['expected_added_trace_ids'] or summary['removed_trace_ids'] != rcfg['expected_removed_trace_ids']:
        raise ValueError('Complete prefix differs from reviewed selection impact')
    save(inputs/'audit.json', summary)
    # Copy and hash the existing method implementation using its shared freezer.
    freeze_protocol(config_path, cfg, root, extra_bindings=[Path(config_path), *markers,
        parent/'asc_calibration/attempts.jsonl', impact/'asc_calibration_attempts.jsonl',
        cohort/'cohorts/calibration.jsonl', cohort/'cohorts/development.jsonl'])
    out = root/'asc_calibration'
    write_jsonl(out/'attempts.jsonl', consumed)
    write_jsonl(out/'all_regraded_parent_attempts.jsonl', regraded)
    write_jsonl(out/'pairs.jsonl', pairs)
    write_jsonl(out/'grading_changes.jsonl', changes)
    save(out/'summary.json', summary)
    seal(out/'COMPLETE.json', [root/'protocol/FROZEN.json', *markers, *sorted(out.iterdir())],
         stage='reviewed_math_calibration_prefix_replay', training_release=False)
    logging.info('Rebuilt calibration: %s', summary)
