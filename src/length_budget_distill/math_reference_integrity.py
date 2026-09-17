"""Flag incomplete boxed references and contextual grading risks for review.

Multiple boxes are not automatically concatenated: some are intermediate work.
This scan preserves the registered gold and the complete original solution.
"""
from collections import Counter
from pathlib import Path
import re

from .experiment_io import read_json
from .factorial import canonical_sha256
from .math_grading import extract_boxed
from .ncsu_reproduction import seal, save, verify
from .records import read_jsonl, write_jsonl

CODE = Path(__file__).resolve().parents[2]


def reference_boxes(text):
    starts = [m.start() for m in re.finditer(r'\\(?:boxed|fbox)\s*\{', text)]
    # Each slice contains one opening box. Nested boxes remain explicitly
    # unresolved rather than silently treating the inner box as the full answer.
    return [{'start': start, 'expression': extract_boxed(text[start:end])}
            for start, end in zip(starts, starts[1:]+[len(text)])]


def diagnostic(row):
    boxes = reference_boxes(row.get('reference_solution') or '')
    question, gold = row['question'], str(row['answer'])
    flags = []
    if len(boxes) > 1: flags.append('multiple_reference_boxes')
    if not boxes: flags.append('no_reference_box')
    if any(r['expression'] is None for r in boxes): flags.append('unresolved_nested_or_unbalanced_box')
    if re.search(r'(?i)find\s+all|all\s+(?:possible\s+)?(?:values|solutions|roots)|(?:separated|separate).*(?:comma|semicolon)|(?:enter|find).*angles', question):
        flags.append('multiple_answer_request')
    if re.search(r'(?i)percent|percentage', question) or r'\%' in gold:
        flags.append('percentage_context')
    radix = re.search(r'(?i)base\s*[- ]?\s*(?:\d+|two|three|four|five|six|seven|eight|nine|ten|twelve|sixteen)|binary|octal|hexadecimal', question)
    if radix and (row.get('answer_spec') or {}).get('kind') != 'radix':
        flags.append('radix_context_without_typed_metadata')
    if re.search(r'\\(?:text|mbox|mathrm)\{', gold) or re.fullmatch(r'(?i)(?:one|two|three|four|five|six|seven|eight|nine|ten)', gold):
        flags.append('textual_reference_answer')
    return {'problem_id': row['problem_id'], 'question_role': row['question_role'],
            'dataset': row['dataset'], 'input_row_sha256': canonical_sha256(row),
            'registered_gold': row['answer'], 'reference_boxes': boxes, 'review_flags': flags,
            'gold_automatically_corrected': False, 'semantic_reference_review_complete': False}


def scan(config_path):
    cfg = read_json(config_path)
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen integrity-scan source')
    launch, out, cohorts = Path(cfg['launch_root']), Path(cfg['result_root']), Path(cfg['cohort_root'])
    verify(launch/'FROZEN.json')
    verify(cohorts/'COMPLETE.json')
    out.mkdir(parents=True, exist_ok=False)
    rows, bindings = [], [Path(config_path), launch/'FROZEN.json', cohorts/'COMPLETE.json']
    for spec in cfg['inputs']:
        path = cohorts/spec['path']
        selected = list(read_jsonl(path))
        if len(selected) != spec['expected_rows']: raise ValueError('Registered role count differs')
        for row in selected:
            if row['question_role'] != spec['role']: raise ValueError('Source role differs')
        rows.extend(selected)
        bindings.append(path)
    if len({r['problem_id'] for r in rows}) != len(rows): raise ValueError('Repeated source problem identity')
    records = [diagnostic(r) for r in rows]
    by_id = {r['problem_id']: r for r in rows}
    write_jsonl(out/'diagnostics.jsonl', records)
    write_jsonl(out/'review_cases.jsonl', [{**by_id[r['problem_id']], 'integrity_diagnostic': r}
                                        for r in records if r['review_flags']])
    summary = {'status': 'reference_integrity_scan_complete_review_required', 'questions': len(rows),
        'role_counts': dict(Counter(r['question_role'] for r in rows)),
        'review_cases': sum(bool(r['review_flags']) for r in records),
        'flags': dict(Counter(f for r in records for f in r['review_flags'])),
        'flags_by_role': {role: dict(Counter(f for r in records if r['question_role'] == role for f in r['review_flags']))
                          for role in sorted({r['question_role'] for r in rows})},
        'gold_corrections_applied': 0, 'all_reference_semantics_validated': False,
        'required_next': ['Review complete question/solution for flagged references; do not concatenate intermediate boxes automatically.',
                          'Freeze corrected gold and context-aware grading separately; preserve original traces and grader evidence.',
                          'Audit calibration/development impact before retaining or revising selected steering configurations.',
                          'Regrade complete raw/steered pools before releasing text compression and student data preparation.']}
    save(out/'summary.json', summary)
    seal(out/'COMPLETE.json', [*bindings, *sorted(out.iterdir())], stage='math_reference_integrity_scan', semantic_review_complete=False)
