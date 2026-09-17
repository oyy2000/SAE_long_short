"""Audit complete registered prediction artifacts under reviewed MATH grading.

This audit emits per-record comparisons and complete changed-output cases. It
does not itself clear those cases, refit a direction, or release student data.
"""
from collections import Counter
from pathlib import Path
import logging

from .experiment_io import read_json
from .factorial import canonical_sha256
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .reviewed_math_cohort import unique_index
from .reviewed_math_grading import grade_reviewed_response
from .typed_math_grading import grade_typed_response

CODE = Path(__file__).resolve().parents[2]


def nested_value(row, name):
    for part in name.split('.'):
        row = row[part]
    return row


def validate_prediction_identity(row, source, group):
    if row['problem_id'] != source['problem_id'] or source['question_role'] != group['question_role']:
        raise ValueError('Prediction identity or role differs')
    for key in ('question', 'dataset', 'question_role', 'near_component_id', 'reference_solution'):
        if key in row and row[key] != source[key]:
            raise ValueError('Prediction source differs: '+key)
    for key in ('answer', 'gold_answer'):
        if key in row and row[key] != source['answer']:
            raise ValueError('Prediction original gold differs')


def audit(config_path):
    cfg = read_json(config_path)
    if CODE != Path(cfg['code_root']):
        raise ValueError('Use frozen prediction-impact source')
    launch, out, cohort = map(Path, (cfg['launch_root'], cfg['result_root'], cfg['cohort_root']))
    verify(launch/'FROZEN.json')
    verify(cohort/'COMPLETE.json')
    bindings = [Path(config_path), launch/'FROZEN.json', cohort/'COMPLETE.json', CODE/cfg['grading_config']]
    source_rows = []
    for role in cfg['source_roles']:
        path = cohort/'cohorts'/f'{role}.jsonl'
        source_rows.extend(read_jsonl(path))
        bindings.append(path)
    sources = unique_index(source_rows, 'impact source cohort')
    grading = read_json(CODE/cfg['grading_config'])
    out.mkdir(parents=True, exist_ok=False)
    groups, changes, mismatches, errors = [], [], [], []
    checked_markers = set()
    for group in cfg['groups']:
        path, marker = Path(group['path']), Path(group['marker'])
        if marker not in checked_markers:
            doc = verify(marker)
            checked_markers.add(marker)
            bindings.append(marker)
        else:
            doc = read_json(marker)
        if str(path.resolve()) not in doc['hashes']:
            raise ValueError('Input is not directly bound by its completed marker')
        bindings.append(path)
        rows = list(read_jsonl(path))
        keys = [tuple(nested_value(r, k) for k in group['key_fields']) for r in rows]
        if len(rows) != group['expected_records'] or len(set(keys)) != len(keys):
            raise ValueError('Missing or duplicate prediction records: '+group['name'])
        if len({r['problem_id'] for r in rows}) != group['expected_questions']:
            raise ValueError('Incomplete problem support: '+group['name'])
        results = []
        for index, row in enumerate(rows):
            source = sources[row['problem_id']]
            validate_prediction_identity(row, source, group)
            response = nested_value(row, group['response_field'])
            old = grade_typed_response(response, source, grading)
            new = grade_reviewed_response(response, source, grading)
            stored = nested_value(row, group['correct_field']) if 'correct_field' in group else group['expected_stored_correct']
            if not isinstance(stored, bool):
                raise ValueError('Historical correctness must be explicit boolean')
            result = {'group': group['name'], 'record_index': index, 'record_key': list(keys[index]),
                      'problem_id': row['problem_id'], 'source_record_sha256': canonical_sha256(row),
                      'historical_is_correct': stored, 'recomputed_v2_grading': old,
                      'reviewed_grading': new, 'correctness_changed': stored != new['is_correct'],
                      'v2_matches_stored_label': stored == old['is_correct']}
            results.append(result)
            if result['correctness_changed'] or not result['v2_matches_stored_label']:
                case = {**result, 'question': source['question'], 'original_answer': source['answer'],
                        'reviewed_answer': source['reviewed_answer'], 'response': response,
                        'source_file': str(path), 'case_review_complete': False}
                changes.append(case)
            if not result['v2_matches_stored_label']:
                mismatches.append(result)
            if new['status'] == 'reviewed_grading_error':
                errors.append(result)
        write_jsonl(out/(group['name']+'.jsonl'), results)
        summary = {'name': group['name'], 'records': len(rows), 'questions': group['expected_questions'],
                   'historical_correct': sum(r['historical_is_correct'] for r in results),
                   'reviewed_correct': sum(r['reviewed_grading']['is_correct'] for r in results),
                   'correctness_changes': sum(r['correctness_changed'] for r in results),
                   'v2_stored_label_mismatches': sum(not r['v2_matches_stored_label'] for r in results),
                   'reviewed_status_counts': dict(Counter(r['reviewed_grading']['status'] for r in results))}
        groups.append(summary)
        logging.info('Prediction impact %s: %s', group['name'], summary)
    write_jsonl(out/'changed_output_review_cases.jsonl', changes)
    write_jsonl(out/'v2_stored_label_mismatches.jsonl', mismatches)
    write_jsonl(out/'parser_error_records.jsonl', errors)
    save(out/'summary.json', {'status': 'prediction_impact_computed_review_pending', 'groups': groups,
                             'records': sum(g['records'] for g in groups), 'changed_output_cases': len(changes),
                             'v2_stored_label_mismatches': len(mismatches),
                             'direction_reuse_cleared': False, 'operating_points_recomputed': False,
                             'full_student_pool_regraded': any(
                                 g['question_role'] == 'student_pool' and
                                 g['expected_questions'] == sum(s['question_role'] == 'student_pool' for s in sources.values())
                                 for g in cfg['groups']),
                             'student_selection_migration_complete': False, 'training_release': False,
                             'historical_predictions_or_scores_modified': False})
    seal(out/'COMPLETE.json', [*bindings, *sorted(out.iterdir())],
         stage='reviewed_math_prediction_impact_diagnostics', training_release=False)
