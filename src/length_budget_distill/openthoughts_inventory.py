"""Inventory pinned OpenThoughts references before proposing a 25K cohort.

This stage performs structural checks and overlap screening, not gold validation,
teacher filtering, or cohort selection. Generated DeepSeek answers are never gold.
"""
from collections import Counter, defaultdict
from pathlib import Path
import logging
import re

from .baseline_data_preflight import near_matches, normalize_question
from .dap_paired_sources import question_key
from .experiment_io import read_json
from .factorial import canonical_sha256, file_sha256
from .math_grading import extract_boxed
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl

CODE = Path(__file__).resolve().parents[2]
COLUMNS = ['problem', 'ground_truth_solution', 'domain', 'source']


def reference_screen(row):
    """Conservative lexical review flags; absence of flags is not eligibility."""
    question = row.get('problem') or ''
    reference = row.get('ground_truth_solution') or ''
    if not isinstance(question, str) or not isinstance(reference, str):
        raise ValueError('Question/reference fields must be strings or null')
    flags = []
    if not question.strip(): flags.append('missing_question')
    if not reference.strip(): flags.append('missing_reference')
    if re.search(r'\b(?:prove|proof|show\s+that|demonstrate\s+that)\b', question, re.I):
        flags.append('proof_wording_review')
    if re.search(r'\b(?:diagram|figure|image|illustration)\b|\[asy\]|<img|!\[', question, re.I):
        flags.append('visual_dependency_review')
    if re.search(r'[\u3400-\u9fff]', question): flags.append('cjk_text_review')
    answer = extract_boxed(reference)
    if not answer: flags.append('missing_or_empty_reference_box')
    return {'reference_box_candidate': answer, 'review_flags': flags,
            'formal_training_ready': False}


def bound_reference_rows(spec):
    """Check the specific input's original binding without claiming recursive audit."""
    path = Path(spec['path']).resolve()
    marker = read_json(spec['marker'])
    if marker['status'] != 'complete' or marker['hashes'].get(str(path)) != file_sha256(path):
        raise ValueError('Reference input no longer matches its original binding: '+str(path))
    rows = list(read_jsonl(path))
    if len(rows) != spec['expected_rows']:
        raise ValueError('Reference count changed: '+spec['name'])
    return [{'problem_id': spec['name']+'::'+r['problem_id'],
             'original_problem_id': r['problem_id'], 'question': r['question'],
             'reference_group': spec['name'], 'exclusion_scope': spec['exclusion_scope']}
            for r in rows]


def exact_overlap(queries, references):
    index = defaultdict(list)
    for row in references: index[question_key(row['question'])].append(row)
    return [{'query_id': q['problem_id'], 'reference_id': r['problem_id'],
             'reference_group': r['reference_group'], 'exclusion_scope': r['exclusion_scope']}
            for q in queries for r in index.get(question_key(q['question']), ())]


def run(config_path):
    import pyarrow.parquet as pq
    cfg = read_json(config_path)
    launch, out = Path(cfg['launch_root']), Path(cfg['result_root'])
    if CODE != Path(cfg['code_root']): raise ValueError('Use the frozen inventory source')
    verify(launch/'FROZEN.json')
    verify(launch/'TEST_COMPLETE.json')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    save(out/'config.json', cfg)
    spec = cfg['dataset']
    info = read_json(spec['metadata_path'])
    if info['sha'] != spec['revision']: raise ValueError('Upstream revision differs')
    items = sorted((r for r in info['siblings'] if r['rfilename'].startswith('metadata/')
                    and r['rfilename'].endswith('.parquet')), key=lambda r: r['rfilename'])
    if len(items) != cfg['expected_files']: raise ValueError('Metadata shard set differs')
    files = []
    for item in items:
        path = Path(cfg['staged_root'])/'openthoughts'/item['rfilename']
        digest = file_sha256(path)
        if path.stat().st_size != item['size'] or digest != item['lfs']['sha256']:
            raise ValueError('Upstream LFS size/hash differs: '+str(path))
        parquet = pq.ParquetFile(path)
        if not set(COLUMNS) <= set(parquet.schema_arrow.names): raise ValueError('Missing metadata columns')
        if any(str(parquet.schema_arrow.field(c).type) != 'string' for c in COLUMNS):
            raise ValueError('Unexpected metadata column types')
        files.append({'path': str(path), 'filename': item['rfilename'], 'sha256': digest,
                      'bytes': item['size'], 'rows': parquet.metadata.num_rows,
                      'schema': str(parquet.schema_arrow)})
        logging.info('Verified upstream metadata %s rows=%d', item['rfilename'], parquet.metadata.num_rows)
    if sum(r['rows'] for r in files) != cfg['expected_rows']: raise ValueError('Metadata total row count differs')
    save(out/'source_files.json', {'repo_id': spec['repo_id'], 'revision': spec['revision'], 'files': files})
    counts, domains, sources, flags = Counter(), Counter(), Counter(), Counter()
    groups = defaultdict(list)
    row_manifest = []

    def math_records():
        global_index = 0
        for shard, entry in enumerate(files):
            offset = 0
            for batch in pq.ParquetFile(entry['path']).iter_batches(batch_size=cfg['batch_size'], columns=COLUMNS):
                for local, row in enumerate(batch.to_pylist()):
                    pid = f'openthoughts-metadata-{shard:02d}-{offset+local:06d}'
                    domains[str(row['domain'])] += 1
                    sources[str(row['domain'])+'::'+str(row['source'])] += 1
                    identity = {'problem_id': pid, 'source_file': entry['filename'],
                                'source_row_index': offset+local, 'global_row_index': global_index,
                                'domain': row['domain'], 'source': row['source'],
                                'source_fields_sha256': canonical_sha256(row)}
                    global_index += 1
                    row_manifest.append(identity)
                    if row['domain'] != 'math': continue
                    screen = reference_screen(row)
                    question = row['problem'] or ''
                    key = question_key(question)
                    record = {**identity, 'question': question,
                              'reference_solution': row['ground_truth_solution'],
                              'reference_field': 'ground_truth_solution',
                              'question_key_sha256': canonical_sha256(key), **screen}
                    flags.update(screen['review_flags'])
                    counts['math_rows'] += 1
                    counts['with_reference_box'] += bool(screen['reference_box_candidate'])
                    counts['without_lexical_review_flags'] += not screen['review_flags']
                    # Keep compact group members; full unchanged text is in math_records.jsonl.
                    groups[key].append({k: record[k] for k in ('problem_id', 'question', 'source_file',
                        'source_row_index', 'reference_box_candidate', 'review_flags')})
                    yield record
                offset += batch.num_rows
            if offset != entry['rows']: raise ValueError('Parquet batch count differs')
            logging.info('Inventoried %s total=%d math=%d', entry['filename'], global_index, counts['math_rows'])
        if global_index != cfg['expected_rows']: raise ValueError('Streamed total differs')

    written = write_jsonl(out/'math_records.jsonl', math_records())
    if written != counts['math_rows']: raise ValueError('Math output count differs')
    write_jsonl(out/'row_manifest.jsonl', row_manifest)
    queries = [members[0] for members in groups.values()]
    duplicates = []
    conflict_ids = set()
    for members in groups.values():
        if len(members) <= 1: continue
        boxes = {normalize_question(r['reference_box_candidate'] or '') for r in members}
        conflict = len(boxes) > 1
        if conflict: conflict_ids.add(members[0]['problem_id'])
        duplicates.append({'representative_id': members[0]['problem_id'],
                           'member_ids': [r['problem_id'] for r in members],
                           'reference_box_string_variants': len(boxes),
                           'reference_box_conflict_review': conflict})
    write_jsonl(out/'duplicate_groups.jsonl', duplicates)
    references = [row for spec in cfg['references'] for row in bound_reference_rows(spec)]
    if len({r['problem_id'] for r in references}) != len(references): raise ValueError('Repeated reference identity')
    write_jsonl(out/'reference_questions.jsonl', references)
    exact = exact_overlap(queries, references)
    write_jsonl(out/'exact_overlap.jsonl', exact)
    # Use the same documented upstream-prefix normalization for both match modes.
    normalized_q = [{**r, 'question': question_key(r['question'])} for r in queries]
    normalized_r = [{**r, 'question': question_key(r['question'])} for r in references]
    logging.info('Starting near-match screen: %d unique math questions, %d references', len(queries), len(references))
    near = near_matches(normalized_q, normalized_r, n=cfg['near_duplicate']['ngram'],
                        threshold=cfg['near_duplicate']['threshold'], exclude_identical=True)
    ref_index = {r['problem_id']: r for r in references}
    for pair in near:
        ref = ref_index[pair['reference_id']]
        pair.update(reference_group=ref['reference_group'], exclusion_scope=ref['exclusion_scope'])
    write_jsonl(out/'near_overlap_review.jsonl', near)
    excluded = {r['query_id'] for r in exact+near if r['exclusion_scope'] == 'holdout'}
    pool = [r for r in queries if not r['review_flags'] and r['problem_id'] not in excluded|conflict_ids]
    write_jsonl(out/'prescreen_candidate_ids.jsonl', [
        {'problem_id': r['problem_id'], 'source_file': r['source_file'], 'source_row_index': r['source_row_index'],
         'status': 'requires_reference_validation_and_frozen_cohort', 'formal_training_ready': False}
        for r in pool])
    summary = {'status': 'source_inventory_complete_not_training_ready',
               'source_rows': len(row_manifest), 'domain_counts': dict(domains), 'source_counts': dict(sources),
               'math_counts': dict(counts), 'lexical_review_flags': dict(flags),
               'unique_normalized_math_questions': len(queries),
               'duplicate_question_groups': len(duplicates), 'reference_box_conflict_groups': len(conflict_ids),
               'exact_pairs_by_reference': dict(Counter(r['reference_group'] for r in exact)),
               'near_pairs_by_reference': dict(Counter(r['reference_group'] for r in near)),
               'unique_math_with_exact_or_near_holdout_match': len(excluded),
               'unique_prescreen_candidates': len(pool), 'target_unique_questions': cfg['target_unique_questions'],
               'formal_training_ready': False, 'validated_reference_answers': 0,
               'generated_teacher_answers_used_as_gold': 0, 'selected_training_questions': 0,
               'limitations': [
                   'Box extraction and lexical flags are screening heuristics, not answer or task validation.',
                   'Proof/visual/CJK flags can have false positives and false negatives; no flags does not prove text-only English answerability.',
                   'Near matches require review. Exact/near holdout matches are withheld from the prescreen candidate list.',
                   'Duplicate reference-box strings are compared lexically, not by mathematical equivalence.',
                   'Existing student-pool overlap is reported separately and is not a holdout exclusion.',
                   'Within-source near duplication and semantic contamination remain unaudited.',
                   'No LiteCoT original 25K sample IDs are claimed; no new student cohort or teacher correctness is established.']}
    save(out/'summary.json', summary)
    report = ('# OpenThoughts 25K 来源可行性审计\n\n'
              f"固定 revision `{spec['revision']}` 的 {len(files)} 个 metadata 分片通过上游 LFS SHA-256、大小、schema 和行数检查，共 {len(row_manifest):,} 行。\n\n"
              f"数学记录 {counts['math_rows']:,} 条，规范化唯一问题 {len(queries):,} 个。含参考 box 的记录 {counts['with_reference_box']:,} 条；去重后、移除词汇复核标记、冲突 box 及精确/近似保留集匹配，剩余 {len(pool):,} 个预筛候选。\n\n"
              '这些数量不是可用训练样本数。仅读取 `ground_truth_solution`，未将生成的 DeepSeek 解答当作标准答案；尚未验证参考答案、冻结 25K 题池或生成学生监督。\n\n'
              '证明、图像和 CJK 检查为保守词汇标记，不是可靠任务/语言分类。近似匹配为既有 5-gram Jaccard 规则，仍需复核；来源内部近似重复尚未处理。保留当前 MATH 学生池重叠为信息项，与评测及校准/开发/SAE 保留题分开。\n\n'
              '下一阶段需核验候选参考、问题类型及来源内部近似重复，再按已批准的 7.5K 结果与成本检查决定 25K 扩展。没有原 LiteCoT 25K IDs，不声称同样本复现。\n')
    with (out/'report_zh.md').open('x') as handle: handle.write(report)
    bindings = [Path(config_path), launch/'FROZEN.json', launch/'TEST_COMPLETE.json', Path(spec['metadata_path'])]
    bindings += [Path(r['path']) for r in files]
    bindings += [Path(s[k]) for s in cfg['references'] for k in ('path', 'marker')]
    seal(out/'COMPLETE.json', [*bindings, *sorted(p for p in out.iterdir() if p.is_file())],
         stage='openthoughts_source_inventory', formal_training_ready=False)
    logging.info('Source inventory completed: %s', summary)
