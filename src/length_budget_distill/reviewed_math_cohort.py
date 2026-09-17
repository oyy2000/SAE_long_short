"""Build a source-preserving MATH answer-definition overlay and diagnose it.

Explicit reviews cover scanner flags. Other definitions inherit the registered
typed target with automatic provenance; they are not called semantic reviews.
The old answer, question, role, split, and near-component fields remain intact.
"""
from collections import Counter
from pathlib import Path
import logging
import shutil

from .experiment_io import read_json
from .factorial import canonical_sha256
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .reviewed_math_definition_validation import definition_probes
from .reviewed_math_grading import grade_reviewed_response, validate_definition
from .typed_math_grading import compile_answer_spec, grade_typed_response

CODE = Path(__file__).resolve().parents[2]
ADDED_FIELDS = {'reviewed_answer', 'answer_definition_provenance'}


def convert_unflagged(row, grading):
    """Translate existing metadata without inferring new targets from prose."""
    if row['dataset'] not in {'math_train', 'math500'}:
        raise ValueError('This overlay is restricted to MATH and MATH-500')
    prior = compile_answer_spec(row, grading)
    if prior != row['answer_spec']:
        raise ValueError('Registered typed definition differs from its source')
    spec = {'mode': 'all' if prior['multiple'] else 'single',
            'kind': prior['kind'], 'answers': prior['gold_components']}
    if prior['multiple']:
        spec['ordered'] = prior['ordered_multiple']
    if prior['kind'] == 'radix':
        spec['base'] = prior['base']
    if prior['rounded_answer_requested']:
        spec['rounded_answer_requested'] = True
    validate_definition(spec)
    return spec


def overlay_row(row, diagnostic, reviewed, grading):
    if ADDED_FIELDS & row.keys():
        raise ValueError('Source already contains overlay fields')
    digest = canonical_sha256(row)
    if (diagnostic['problem_id'] != row['problem_id'] or
            diagnostic['input_row_sha256'] != digest or
            diagnostic['question_role'] != row['question_role']):
        raise ValueError('Scan no longer binds this source row')
    if reviewed is not None:
        if (reviewed['problem_id'] != row['problem_id'] or
                reviewed['input_row_sha256'] != digest or
                reviewed['question_role'] != row['question_role']):
            raise ValueError('Reviewed definition does not bind this source')
        spec = reviewed['reviewed_answer']
        validate_definition(spec)
        origin = 'assistant_question_and_reference_target_type_review'
    else:
        if diagnostic['review_flags']:
            raise ValueError('Flagged source lacks a validated explicit review')
        spec = convert_unflagged(row, grading)
        origin = 'automatic_translation_of_unflagged_registered_typed_definition'
    return {**row, 'reviewed_answer': spec,
            'answer_definition_provenance': {
                'origin': origin, 'input_row_sha256': digest,
                'scanner_flags': diagnostic['review_flags'],
                'original_fields_preserved': True,
                'independent_mathematical_proof_review': False}}


def unique_index(rows, label):
    indexed = {r['problem_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate identities in {label}')
    return indexed


def build(config_path):
    cfg = read_json(config_path)
    if CODE != Path(cfg['code_root']):
        raise ValueError('Use frozen cohort-overlay source')
    launch, out = Path(cfg['launch_root']), Path(cfg['result_root'])
    source, scan, validation = map(Path, (cfg['cohort_root'], cfg['scan_root'], cfg['validation_root']))
    bindings = [Path(config_path), launch/'FROZEN.json']
    verify(launch/'FROZEN.json')
    for root in (source, scan, validation):
        verify(root/'COMPLETE.json')
        bindings.append(root/'COMPLETE.json')
    diagnostics = unique_index(list(read_jsonl(scan/'diagnostics.jsonl')), 'scan')
    definitions = unique_index(list(read_jsonl(validation/'validated_definitions.jsonl')), 'reviewed definitions')
    if len(definitions) != cfg['expected_reviewed_questions']:
        raise ValueError('Incomplete definition input')
    bindings.extend([scan/'diagnostics.jsonl', validation/'validated_definitions.jsonl', CODE/cfg['grading_config']])
    grading = read_json(CODE/cfg['grading_config'])
    out.mkdir(parents=True, exist_ok=False)
    all_rows, probes, checks, cases, construction_errors = [], [], [], [], []
    for item in cfg['inputs']:
        original_path = source/item['path']
        rows = list(read_jsonl(original_path))
        if len(rows) != item['expected_rows'] or any(r['question_role'] != item['role'] for r in rows):
            raise ValueError('Original role/count differs from protocol')
        bindings.append(original_path)
        converted = []
        for row in rows:
            pid = row['problem_id']
            try:
                annotated = overlay_row(row, diagnostics[pid], definitions.get(pid), grading)
            except (ValueError, KeyError) as error:
                construction_errors.append({'problem_id': pid, 'error': str(error)})
                cases.append({**row, 'overlay_diagnostic': {'construction_error': str(error)}})
                continue
            if {k: v for k, v in annotated.items() if k not in ADDED_FIELDS} != row:
                raise ValueError('Overlay changed an original source field')
            converted.append(annotated)
            row_probes = []
            for name, value, expected in definition_probes(annotated['reviewed_answer']):
                result = grade_reviewed_response(r'\boxed{'+value+'}', annotated, grading)
                row_probes.append({'problem_id': pid, 'probe': name, 'answer': value,
                                   'expected_correct': expected, 'grading': result,
                                   'passed': result['is_correct'] == expected})
            probes.extend(row_probes)
            old = grade_typed_response(row['reference_solution'], row, grading)
            new = grade_reviewed_response(row['reference_solution'], annotated, grading)
            check = {'problem_id': pid, 'question_role': row['question_role'],
                     'definition_origin': annotated['answer_definition_provenance']['origin'],
                     'old_grading': old, 'overlay_grading': new,
                     'failed_probe_count': sum(not r['passed'] for r in row_probes)}
            checks.append(check)
            if check['failed_probe_count'] or (not new['is_correct'] and pid not in cfg['expected_source_parser_limitations']):
                cases.append({**annotated, 'overlay_diagnostic': check})
            if len(checks) % 500 == 0:
                logging.info('Diagnosed %d MATH sources; %d follow-up cases', len(checks), len(cases))
        destination = out/item['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination, converted)
        all_rows.extend(rows)
    indexed = unique_index(all_rows, 'complete source cohort')
    if set(indexed) != set(diagnostics) or not set(definitions) <= set(indexed):
        raise ValueError('Scan, reviews and complete source cohort identities differ')
    if len(all_rows) != cfg['expected_questions']:
        raise ValueError('Incomplete source cohort')
    for name in cfg['passthrough_evaluation_files']:
        path = source/name
        dest = out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        bindings.append(path)
    write_jsonl(out/'definition_probes.jsonl', probes)
    write_jsonl(out/'source_parser_diagnostics.jsonl', checks)
    write_jsonl(out/'review_cases.jsonl', cases)
    failures = sorted(r['problem_id'] for r in checks if not r['overlay_grading']['is_correct'])
    cleared = not cases and not construction_errors and failures == sorted(cfg['expected_source_parser_limitations'])
    summary = {'status': 'cohort_overlay_ready_for_prediction_impact' if cleared else 'cohort_overlay_requires_reference_followup',
               'source_questions': len(all_rows), 'converted_questions': len(checks),
               'explicitly_reviewed_questions': len(definitions),
               'automatically_translated_questions': len(checks)-len(definitions),
               'role_counts': dict(Counter(r['question_role'] for r in all_rows)),
               'definition_origins': dict(Counter(r['definition_origin'] for r in checks)),
               'probe_count': len(probes), 'failed_probes': sum(not r['passed'] for r in probes),
               'source_parser_failures': failures, 'followup_cases': len(cases),
               'construction_errors': construction_errors, 'original_source_fields_preserved': True,
               'independent_proof_review': False, 'actual_prediction_impact_validated': False,
               'historical_predictions_or_scores_modified': False, 'training_release': False}
    save(out/'summary.json', summary)
    outputs = sorted(p for p in out.rglob('*') if p.is_file())
    seal(out/'DIAGNOSTIC_COMPLETE.json', [*bindings, *outputs], stage='reviewed_math_cohort_diagnostics', training_release=False)
    if not cleared:
        raise ValueError(f'{len(cases)} source follow-up cases; overlay not released')
    seal(out/'COMPLETE.json', [out/'DIAGNOSTIC_COMPLETE.json', *bindings, *outputs],
         stage='reviewed_math_cohort_overlay', prediction_impact_required=True, training_release=False)
