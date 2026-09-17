"""Check source-answer parsing, exact MATH anchors, and internal near groups.

All candidates and failures remain visible. This stage does not select a 25K
cohort or certify unanchored reference answers as correct.
"""
from collections import Counter, defaultdict
from pathlib import Path
import importlib.metadata
import logging
import time

from .baseline_data_preflight import near_matches
from .dap_paired_sources import question_key
from .experiment_io import read_json
from .factorial import canonical_sha256
from .math_cohort_freeze import components
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .typed_math_grading import compile_answer_spec, grade_typed_response

CODE = Path(__file__).resolve().parents[2]


def context(config_path, prepared=True):
    cfg = read_json(config_path)
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen reference-audit source')
    launch, root = Path(cfg['launch_root']), Path(cfg['result_root'])
    verify(launch/'FROZEN.json')
    verify(launch/'TEST_COMPLETE.json')
    if prepared: verify(root/'inputs/COMPLETE.json')
    return cfg, root


def check_reference(row, grading):
    """Separate self-comparison from agreement with another published source."""
    from math_verify.errors import TimeoutException
    answer = row.get('reference_box_candidate')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Missing original reference box; generated answers cannot fill it')
    target = {'problem_id': row['problem_id'], 'question': row['question'],
              'dataset': 'openthoughts', 'answer': answer}
    start = time.monotonic()
    try:
        if len(answer) > grading['max_expression_characters']: raise ValueError('Answer exceeds parsing budget')
        spec = compile_answer_spec(target, grading)
        self_grade = grade_typed_response(r'\boxed{'+answer+'}', target, grading)
    except (Exception, TimeoutException) as error:
        spec = None
        self_grade = {'is_correct': False, 'status': 'reference_spec_error',
                      'error_type': type(error).__name__, 'error': str(error)}
    anchor = row.get('math_anchor')
    if anchor is not None and question_key(row['question']) != question_key(anchor['question']):
        raise ValueError('MATH anchor is not the same normalized question')
    anchor_grade = grade_typed_response(r'\boxed{'+answer+'}', anchor, grading) if anchor else None
    return {'problem_id': row['problem_id'], 'input_row_sha256': canonical_sha256(row),
            'reference_box_candidate': answer, 'answer_spec': spec, 'reference_self_grade': self_grade,
            'math_anchor_id': anchor['problem_id'] if anchor else None,
            'math_anchor_grade': anchor_grade, 'seconds': time.monotonic()-start,
            'requires_review': not self_grade['is_correct'] or (anchor_grade is not None and not anchor_grade['is_correct']),
            'formal_training_ready': False,
            'limitation': 'Self-comparison diagnoses parser coverage; anchor agreement is cross-source consistency, not independent proof of mathematical correctness.'}


def propagated_holdouts(groups, direct_holdouts):
    direct = set(direct_holdouts)
    known = {pid for group in groups for pid in group}
    if not direct <= known: raise ValueError('Unknown direct holdout in near-component audit')
    return {pid for group in groups if direct.intersection(group) for pid in group}


def prepare(config_path):
    cfg, root = context(config_path, prepared=False)
    if root.exists(): raise FileExistsError(root)
    inventory = Path(cfg['inventory_root'])
    verify(inventory/'COMPLETE.json')
    verify(Path(cfg['inventory_verification_root'])/'COMPLETE.json')
    cohort = Path(cfg['cohort_root'])
    verify(cohort/'COMPLETE.json')
    grading = read_json(cfg['grading_config'])
    if grading != read_json(cohort/'grading_config.json'): raise ValueError('Grading differs from existing MATH protocol')
    versions = {name: importlib.metadata.version(name) for name in cfg['expected_versions']}
    if versions != cfg['expected_versions']: raise ValueError('Pinned grading dependencies differ')
    candidates = list(read_jsonl(inventory/'prescreen_candidate_ids.jsonl'))
    ids = {r['problem_id'] for r in candidates}
    if len(ids) != len(candidates) or len(ids) != cfg['expected_candidates']: raise ValueError('Candidate identity/count differs')
    math = list(read_jsonl(cohort/'cohorts/student_pool.jsonl'))
    if len(math) != cfg['expected_math_anchors']: raise ValueError('MATH anchor pool count differs')
    anchors = defaultdict(list)
    for row in math: anchors[question_key(row['question'])].append(row)
    if any(len(v) != 1 for v in anchors.values()): raise ValueError('Ambiguous exact MATH anchor')
    selected, all_questions = [], []
    for row in read_jsonl(inventory/'math_records.jsonl'):
        all_questions.append({'problem_id': row['problem_id'], 'question': row['question']})
        if row['problem_id'] not in ids: continue
        if row['review_flags'] or row['reference_field'] != 'ground_truth_solution': raise ValueError('Unexpected prescreen source')
        matches = anchors.get(question_key(row['question']), [])
        selected.append({**row, 'math_anchor': matches[0] if matches else None})
    if {r['problem_id'] for r in selected} != ids or len(selected) != len(ids): raise ValueError('Missing or duplicated candidate source row')
    if len(all_questions) != cfg['expected_math_questions']: raise ValueError('All-math source count differs')
    direct = sorted({r['query_id'] for file in ('exact_overlap.jsonl', 'near_overlap_review.jsonl')
                     for r in read_jsonl(inventory/file) if r['exclusion_scope'] == 'holdout'})
    inputs = root/'inputs'
    inputs.mkdir(parents=True)
    write_jsonl(inputs/'all_math_questions.jsonl', all_questions)
    save(inputs/'direct_holdout_ids.json', {'problem_ids': direct})
    for shard in range(cfg['shards']):
        write_jsonl(inputs/f'shard_{shard:02d}.jsonl', selected[shard::cfg['shards']])
    save(inputs/'grading_config.json', grading)
    save(inputs/'versions.json', versions)
    save(inputs/'summary.json', {'candidate_questions': len(selected), 'all_math_questions': len(all_questions),
        'exact_math_anchored_candidates': sum(r['math_anchor'] is not None for r in selected),
        'direct_holdout_questions': len(direct), 'shards': cfg['shards'], 'formal_training_ready': False})
    bindings = [Path(config_path), Path(cfg['launch_root'])/'FROZEN.json', inventory/'COMPLETE.json',
                Path(cfg['inventory_verification_root'])/'COMPLETE.json', cohort/'COMPLETE.json']
    seal(inputs/'COMPLETE.json', [*bindings, *sorted(inputs.iterdir())], stage='openthoughts_reference_inputs', formal_training_ready=False)


def audit(config_path, shard):
    cfg, root = context(config_path)
    if not 0 <= shard < cfg['shards']: raise ValueError('Invalid audit shard')
    out = root/'checks'/f'shard_{shard:02d}'
    out.mkdir(parents=True, exist_ok=False)
    source = root/'inputs'/f'shard_{shard:02d}.jsonl'
    grading = read_json(root/'inputs/grading_config.json')
    counts, kinds = Counter(), Counter()
    def records():
        for i, row in enumerate(read_jsonl(source)):
            result = check_reference(row, grading)
            counts['questions'] += 1
            counts['self_grade_passed'] += result['reference_self_grade']['is_correct']
            counts['requires_review'] += result['requires_review']
            if result['math_anchor_grade'] is not None:
                counts['math_anchored'] += 1
                counts['math_anchor_agreement'] += result['math_anchor_grade']['is_correct']
            kinds[(result['answer_spec'] or {}).get('kind', 'spec_error')] += 1
            yield result
            if i % 500 == 0: logging.info('Reference audit shard=%d records=%d', shard, i+1)
    write_jsonl(out/'reference_checks.jsonl', records())
    save(out/'summary.json', {'counts': dict(counts), 'answer_kinds': dict(kinds), 'formal_training_ready': False})
    seal(out/'COMPLETE.json', [Path(config_path), root/'inputs/COMPLETE.json', source, *sorted(out.iterdir())],
         stage='openthoughts_reference_checks', shard=shard, formal_training_ready=False)


def near(config_path):
    cfg, root = context(config_path)
    out = root/'near'
    out.mkdir(parents=True, exist_ok=False)
    source = root/'inputs/all_math_questions.jsonl'
    rows = [{**r, 'question': question_key(r['question'])} for r in read_jsonl(source)]
    logging.info('Internal near screen: %d math questions', len(rows))
    pairs = near_matches(rows, rows, n=cfg['near_duplicate']['ngram'], threshold=cfg['near_duplicate']['threshold'])
    edges = {tuple(sorted((r['query_id'], r['reference_id']))): r for r in pairs}
    groups = components([r['problem_id'] for r in rows], edges)
    direct = read_json(root/'inputs/direct_holdout_ids.json')['problem_ids']
    reserved = propagated_holdouts(groups, direct)
    write_jsonl(out/'near_pairs.jsonl', list(edges.values()))
    write_jsonl(out/'components.jsonl', [{'component_id': canonical_sha256(g), 'problem_ids': g,
        'holdout_connected': bool(set(g)&reserved)} for g in groups])
    save(out/'summary.json', {'math_questions': len(rows), 'unique_near_pairs': len(edges), 'components': len(groups),
        'largest_component': max(map(len, groups)), 'direct_holdout_questions': len(direct),
        'holdout_connected_questions': len(reserved), 'extra_holdout_neighbors': len(reserved-set(direct)),
        'near_matches_are_review_flags': True, 'formal_training_ready': False})
    seal(out/'COMPLETE.json', [Path(config_path), root/'inputs/COMPLETE.json', source, *sorted(out.iterdir())],
         stage='openthoughts_internal_near_audit', formal_training_ready=False)


def merge(config_path):
    cfg, root = context(config_path)
    out = root/'merged'
    out.mkdir(parents=True, exist_ok=False)
    verify(root/'near/COMPLETE.json')
    groups = list(read_jsonl(root/'near/components.jsonl'))
    group_by_id = {pid: r for r in groups for pid in r['problem_ids']}
    checks, inputs, markers = {}, {}, [root/'inputs/COMPLETE.json', root/'near/COMPLETE.json']
    for shard in range(cfg['shards']):
        stage = root/'checks'/f'shard_{shard:02d}'
        verify(stage/'COMPLETE.json')
        markers.append(stage/'COMPLETE.json')
        source = list(read_jsonl(root/'inputs'/f'shard_{shard:02d}.jsonl'))
        records = list(read_jsonl(stage/'reference_checks.jsonl'))
        if [r['problem_id'] for r in records] != [r['problem_id'] for r in source]: raise ValueError('Shard record order/membership differs')
        for record, original in zip(records, source):
            if record['problem_id'] in checks or record['input_row_sha256'] != canonical_sha256(original):
                raise ValueError('Repeated or changed source check')
            checks[record['problem_id']], inputs[record['problem_id']] = record, original
    if len(checks) != cfg['expected_candidates']: raise ValueError('Incomplete reference-check coverage')
    review = [r for r in checks.values() if r['requires_review']]
    provisional = [r for r in checks.values() if not r['requires_review'] and not group_by_id[r['problem_id']]['holdout_connected']]
    write_jsonl(out/'reference_review_cases.jsonl', [{**inputs[r['problem_id']], 'diagnostic': r} for r in review])
    write_jsonl(out/'provisional_parseable_ids.jsonl', [{'problem_id': r['problem_id'],
        'near_component_id': group_by_id[r['problem_id']]['component_id'],
        'math_anchor_id': r['math_anchor_id'], 'formal_training_ready': False} for r in provisional])
    # A source-only quality sample, chosen before reviewing outcomes. It is not
    # a model-development cohort or an independent annotation exercise.
    sample = sorted(inputs.values(), key=lambda r: canonical_sha256([cfg['quality_sample_seed'], r['problem_id']]))[:cfg['quality_sample_size']]
    write_jsonl(out/'source_quality_sample.jsonl', [{**r, 'diagnostic': checks[r['problem_id']]} for r in sample])
    anchored = [r for r in checks.values() if r['math_anchor_grade'] is not None]
    summary = {'status': 'reference_and_near_audit_complete_not_training_ready', 'candidate_questions': len(checks),
        'self_grade_passed': sum(r['reference_self_grade']['is_correct'] for r in checks.values()),
        'reference_review_cases': len(review), 'math_anchored_candidates': len(anchored),
        'math_anchor_agreement': sum(r['math_anchor_grade']['is_correct'] for r in anchored),
        'answer_kinds': dict(Counter((r['answer_spec'] or {}).get('kind', 'spec_error') for r in checks.values())),
        'additional_prescreen_candidates_holdout_connected': sum(group_by_id[pid]['holdout_connected'] for pid in checks),
        'provisional_parseable_questions': len(provisional),
        'near_components_with_provisional_question': len({group_by_id[r['problem_id']]['component_id'] for r in provisional}),
        'source_quality_sample_size': len(sample), 'source_quality_sample_reviewed': False,
        'formal_training_ready': False, 'selected_training_questions': 0,
        'limitations': ['Self-comparison does not validate reference correctness, extraction completeness, language or answerability.',
            'MATH anchor agreement is cross-source consistency on exactly matched student-pool questions, not full-corpus correctness.',
            'All near components use the registered conservative lexical screen; connected variants need not be identical problems.',
            'No teacher coverage, common support, SFT, or original LiteCoT 25K identity is established.']}
    save(out/'summary.json', summary)
    report = ('# OpenThoughts 参考解析与内部近似题审计\n\n'
        f"全部 {len(checks):,} 个预筛候选完成原参考答案自比较，其中 {summary['self_grade_passed']:,} 个通过；共有 {len(review):,} 个解析或跨源一致性复核 case。\n\n"
        f"与 MATH 学生池精确同题的 {len(anchored):,} 个候选中，{summary['math_anchor_agreement']:,} 个在既有 typed grader 下与 MATH 参考一致。未用模型生成答案作为 gold。\n\n"
        f"全部数学来源完成内部 near 连通组检查，另有 {summary['additional_prescreen_candidates_holdout_connected']:,} 个原预筛候选通过近似链连接到保留题。排除该类及复核 case 后，暂有 {len(provisional):,} 个可解析候选，覆盖 {summary['near_components_with_provisional_question']:,} 个 near 组。\n\n"
        '可解析不代表参考答案正确。尚需处理复核 case、阅读预先确定的来源质量样本、验证题型与参考语义，再按原 7.5K 结果和成本检查推进 25K；本阶段没有选择训练题池。\n')
    with (out/'report_zh.md').open('x') as handle: handle.write(report)
    seal(out/'COMPLETE.json', [Path(config_path), *markers, *sorted(out.iterdir())],
         stage='openthoughts_reference_near_merge', formal_training_ready=False)
