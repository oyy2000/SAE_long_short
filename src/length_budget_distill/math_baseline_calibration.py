"""Use the frozen MATH roles for task-specific ASC calibration and development.

The CES fitter and generator are shared with the original-setting checks. Only
input preparation and the context-aware answer backend differ from GSM8K.
"""
from pathlib import Path

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import file_sha256
from .ncsu_reproduction import verify, save
from .baseline_reproduction import ordered, freeze_protocol, grade_prediction


def validate_roles(calibration, development, student, reserved):
    groups = {'calibration':calibration, 'development':development,
              'student_pool':student, 'dap_development_reserved':reserved}
    seen_ids, seen_components = set(), set()
    for role, rows in groups.items():
        ids = {row['problem_id'] for row in rows}
        components = {row['near_component_id'] for row in rows}
        if not rows or len(ids) != len(rows) or any(row['question_role'] != role for row in rows):
            raise ValueError('Missing, duplicate, or incorrect question role: '+role)
        if seen_ids & ids or seen_components & components:
            raise ValueError('Question or near-duplicate component crosses roles')
        seen_ids.update(ids); seen_components.update(components)


def make_reference_pool(rows, tokenizer, cfg):
    if cfg['grading'].get('method') not in {'typed_math_v2', 'reviewed_math_v1'}:
        raise ValueError('Reference-pool grading must declare its mathematics backend')
    accepted, excluded = [], []
    for row in ordered(rows, cfg['seed']):
        # A corrected answer does not automatically repair its original reasoning.
        if row.get('reviewed_gold_override'):
            excluded.append({'problem_id':row['problem_id'], 'reason':'gold_corrected_original_reasoning_not_repaired'})
            continue
        reference = row['reference_solution']
        grade = grade_prediction(cfg, reference, row['answer'], source=row)
        if not grade['is_correct']:
            excluded.append({'problem_id':row['problem_id'], 'reason':'reference_not_verified', 'grade':grade})
            continue
        prompt = cfg['prompt_policy']['question_template'].format(question=row['question'])
        rendered = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(reference, add_special_tokens=False))
        if len(tokenizer.encode(rendered, add_special_tokens=False))+count+1 > cfg['asc']['max_sequence_length']:
            excluded.append({'problem_id':row['problem_id'], 'reason':'reference_exceeds_fitting_context'})
            continue
        accepted.append({**row, 'prompt':prompt, 'concise':reference, 'concise_tokens':count,
            'source_trace_id':'official-reference-'+row['problem_id'], 'source_kind':'official_math_reference',
            'source_split':'calibration', 'reference_grade':grade})
    if len(accepted) < cfg['asc']['pairs']:
        raise ValueError('Insufficient verified reference questions for target-teacher calibration')
    return accepted, excluded


def prepare(config_path):
    from transformers import AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    if root.exists(): raise FileExistsError(root)
    parent = Path(cfg['math_cohorts']); generic_parent = Path(cfg['generic_parent'])
    bindings = [parent/'COMPLETE.json', generic_parent/'protocol/FROZEN.json', generic_parent/'protocol/SOURCES.json']
    for marker in bindings: verify(marker)
    generic_cfg = read_json(generic_parent/'protocol/frozen_config.json')
    for key in ('general_samples','general_holdout_samples','general_text_source','general_text_subset','general_text_revision','general_parquet'):
        if cfg['asc'][key] != generic_cfg['asc'][key]: raise ValueError('Generic-text protocol differs: '+key)
    grading = read_json(parent/'grading_config.json')
    if cfg['grading']['method'] != 'typed_math_v2': raise ValueError('MATH must use the typed answer backend')
    cfg['grading']['config'] = grading
    roles = {name:list(read_jsonl(parent/'cohorts'/f'{name}.jsonl')) for name in
             ('calibration','development','student_pool','dap_development_reserved')}
    validate_roles(*(roles[k] for k in ('calibration','development','student_pool','dap_development_reserved')))
    if len(roles['calibration']) != cfg['calibration_generation']['pool_size']:
        raise ValueError('Calibration role size changed')
    # Explicitly bind the completed cap check used to justify this bounded run.
    cap_root = Path(cfg['cap_diagnostic_root'])
    for relative in ('protocol/FROZEN.json','protocol/SOURCES.json','qwen7b_teacher/COMPLETE.json'):
        marker = cap_root/relative; verify(marker); bindings.append(marker)
    cap_cfg = read_json(cap_root/'protocol/frozen_config.json')
    if any(cfg['teacher'][k] != cap_cfg['models']['qwen7b_teacher'][k] for k in ('model_name','revision','snapshot_path')):
        raise ValueError('MATH cap evidence belongs to a different teacher')
    cap_summary = read_json(cap_root/'qwen7b_teacher/summary.json')
    if cap_summary['proposed_cap'] is None or cfg['calibration_generation']['max_new_tokens'] < cap_summary['proposed_cap']:
        raise ValueError('Calibration budget is below its development proposal')
    tok = AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'], local_files_only=True)
    pairs, excluded = make_reference_pool(roles['calibration'], tok, cfg)
    sources = [{**r, 'prompt':cfg['prompt_policy']['question_template'].format(question=r['question'])}
               for r in ordered(roles['development'], cfg['validation']['subset_seed'])[:cfg['validation']['count']]]
    if len(sources) != cfg['validation']['count']: raise ValueError('Incomplete development cohort')
    inputs = root/'inputs'
    write_jsonl(inputs/'pairs.jsonl', pairs); write_jsonl(inputs/'sources.jsonl', sources)
    write_jsonl(inputs/'reference_exclusions.jsonl', excluded)
    for filename, expected in (('generic_text.jsonl',cfg['asc']['general_samples']),
                               ('generic_text_holdout.jsonl',cfg['asc']['general_holdout_samples'])):
        path = generic_parent/'inputs'/filename; rows = list(read_jsonl(path))
        if len(rows) != expected: raise ValueError('Incomplete generic-text source')
        write_jsonl(inputs/filename, rows); bindings.append(path)
    save(inputs/'audit.json', {'calibration_role_questions':len(roles['calibration']), 'eligible_reference_questions':len(pairs),
        'excluded_reference_questions':len(excluded), 'development_questions':len(sources),
        'student_pool_questions':len(roles['student_pool']), 'question_role_overlap':0, 'near_component_role_overlap':0,
        'all_reference_answers_verified':True, 'target_teacher_verbose_traces_generated':False,
        'student_training_authorized_from_these_records':False, 'source_audit':generic_cfg['source_audit'],
        'generic_parent_frozen_sha256':file_sha256(generic_parent/'protocol/FROZEN.json'),
        'cap_proposal':cap_summary['proposed_cap'], 'calibration_generation_cap':cfg['calibration_generation']['max_new_tokens'],
        'cap_scope':'Reference budget justification on 32 development questions; no guarantee for steered generation or student SFT.'})
    freeze_protocol(config_path, cfg, root, extra_bindings=bindings)
