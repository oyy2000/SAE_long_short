"""Validate explicitly reviewed targets against immutable source rows.

This produces preparation evidence only. It neither replaces cohort gold nor
releases downstream jobs. Source self-grading is a parser diagnostic, separate
from the assistant's semantic review and from actual model-output validation.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .experiment_io import read_json
from .factorial import canonical_sha256
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .reviewed_math_grading import grade_reviewed_response, validate_definition
from .typed_math_grading import grade_typed_response

CODE = Path(__file__).resolve().parents[2]


def validate_binding(decision, case):
    if decision['problem_id'] != case['problem_id']:
        raise ValueError('Review identity differs from input')
    source = {k: v for k, v in case.items() if k != 'integrity_diagnostic'}
    if canonical_sha256(source) != decision['input_row_sha256']:
        raise ValueError('Review binds a different original source row')
    if canonical_sha256(case) != decision['review_case_sha256']:
        raise ValueError('Review binds a different diagnostic case')
    if decision['question_role'] != source['question_role']:
        raise ValueError('Review changes the original question role')
    if decision['applied'] is not False or not decision.get('reviewed_scope') or not decision.get('reason'):
        raise ValueError('Expected an explicit, unapplied semantic proposal')
    validate_definition(decision['reviewed_answer_proposal'])
    return source


def definition_probes(spec):
    """Check required cardinality/order and each legitimate alternative."""
    answers = spec['answers']
    positives = answers if spec['mode'] == 'any' else [';'.join(answers)]
    probes = [('complete_canonical', value, True) for value in positives]
    if spec['mode'] == 'all' and len(answers) > 1:
        probes += [('missing_required_component', ';'.join(answers[:i]+answers[i+1:]), False)
                   for i in range(len(answers))]
        probes.append(('extra_required_component', ';'.join(answers+[answers[-1]]), False))
        if spec.get('ordered'):
            probes.append(('reversed_required_order', ';'.join(reversed(answers)), False))
    elif spec['mode'] == 'any' and len(answers) > 1:
        probes.append(('multiple_alternatives_when_one_requested', ';'.join(answers), False))
    if spec['mode'] == 'single' and spec.get('unit_aliases'):
        value = answers[0]
        probes.extend(('registered_quantity_unit', value+' '+unit, True)
                      for unit in spec['unit_aliases'])
        wrong_unit = next(unit for unit in ('kg', 'pounds', 'seconds')
                          if unit not in spec['unit_aliases'])
        probes.append(('unregistered_quantity_unit', value+r'\text{ '+wrong_unit+'}', False))
    if spec['mode'] == 'single' and spec['kind'] == 'percentage':
        value = answers[0]
        probes.extend([('explicit_percentage_suffix', value+r'\%', True),
                       ('percentage_word_suffix', value+' percent', True)])
        try:
            number = Decimal(value)
        except InvalidOperation:
            number = None
        if number is not None and number.is_finite() and number != 0:
            proportion = format(number/100, 'f')
            probes.extend([('unmarked_proportion_is_not_percentage_points', proportion, False),
                           ('consistent_percentage_equivalence', proportion+'='+value+r'\%', True),
                           ('contradictory_percentage_equivalence', format(number/100+1, 'f')+'='+value+r'\%', False)])
    return probes


def validate(config_path):
    cfg = read_json(config_path)
    if Path(cfg['code_root']) != CODE: raise ValueError('Use the frozen validation source')
    launch = Path(cfg['launch_root'])
    verify(launch/'FROZEN.json')
    review_roots = [Path(p) for p in cfg['review_roots']] if 'review_roots' in cfg else [Path(cfg['review_root'])]
    inputs, decisions, review_bindings = [], [], []
    for review_root in review_roots:
        verify(review_root/'COMPLETE.json')
        inputs.extend(read_jsonl(review_root/'reviewed_inputs.jsonl'))
        decisions.extend(read_json(review_root/'review.json')['decisions'])
        review_bindings.extend([review_root/'COMPLETE.json', review_root/'review.json', review_root/'reviewed_inputs.jsonl'])
    ids = [r['problem_id'] for r in inputs]
    decision_ids = [d['problem_id'] for d in decisions]
    if len(ids) != cfg['expected_questions'] or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate reviewed source rows')
    if len(set(decision_ids)) != len(decision_ids) or set(ids) != set(decision_ids):
        raise ValueError('Review/input identity mismatch')
    by_id = {r['problem_id']: r for r in inputs}
    grading = read_json(CODE/cfg['grading_config'])
    out = Path(cfg['result_root'])
    out.mkdir(parents=True, exist_ok=False)
    checks, source_checks, definitions = [], [], []
    for decision in decisions:
        source = validate_binding(decision, by_id[decision['problem_id']])
        spec = decision['reviewed_answer_proposal']
        row = {**source, 'reviewed_answer': spec}
        definitions.append({'problem_id': source['problem_id'], 'input_row_sha256': decision['input_row_sha256'],
                            'question_role': source['question_role'], 'reviewed_answer': spec})
        for name, value, expected in definition_probes(spec):
            result = grade_reviewed_response(r'\boxed{'+value+'}', row, grading)
            checks.append({'problem_id': source['problem_id'], 'probe': name, 'answer': value,
                           'expected_correct': expected, 'grading': result,
                           'passed': result['is_correct'] == expected})
        source_checks.append({'problem_id': source['problem_id'],
                              'old_grading': grade_typed_response(source['reference_solution'], source, grading),
                              'reviewed_grading': grade_reviewed_response(source['reference_solution'], row, grading)})
    write_jsonl(out/'definition_probes.jsonl', checks)
    write_jsonl(out/'source_parser_diagnostics.jsonl', source_checks)
    write_jsonl(out/'validated_definitions.jsonl', definitions)
    failures = [c for c in checks if not c['passed']]
    source_failures = sorted(c['problem_id'] for c in source_checks if not c['reviewed_grading']['is_correct'])
    summary = {'status': 'reviewed_definition_validation_complete' if not failures else 'definition_validation_failed',
               'questions': len(definitions), 'probe_count': len(checks), 'failed_probes': len(failures),
               'modes': dict(Counter(d['reviewed_answer']['mode'] for d in definitions)),
               'kinds': dict(Counter(d['reviewed_answer']['kind'] for d in definitions)),
               'source_parser_failures': source_failures,
               'source_parser_limitations_match_registration': source_failures == sorted(cfg['expected_source_parser_limitations']),
               'source_self_grading_proves_reference_semantics': False,
               'canonical_probe_success_proves_actual_model_grading': False,
               'gold_changes_applied': False, 'complete_cohort_review': False, 'training_release': False}
    save(out/'summary.json', summary)
    if failures: raise ValueError(f'{len(failures)} definition probes failed')
    if not summary['source_parser_limitations_match_registration']:
        raise ValueError('Unexpected source parser limitations require investigation')
    seal(out/'COMPLETE.json', [Path(config_path), launch/'FROZEN.json', *review_bindings,
                             CODE/cfg['grading_config'], *sorted(out.iterdir())],
         stage='reviewed_math_definition_validation', training_release=False)
