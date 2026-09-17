"""Freeze MATH problem roles without yet freezing a student training protocol.

Near-duplicate components stay together. DAP development sources and their
connected neighbors are excluded from the student pool before allocating the
calibration/development holdouts. Reuses the existing normalization and grader.
"""
from pathlib import Path
from collections import Counter, defaultdict
import logging
import shutil

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save, seal, verify
from .baseline_data_preflight import near_matches, normalize_question, load_source
from .typed_math_grading import compile_answer_spec

CODE = Path(__file__).resolve().parents[2]


def components(problem_ids, edges):
    ids = list(problem_ids)
    if len(set(ids)) != len(ids): raise ValueError('Duplicate problem IDs')
    parent = {pid:pid for pid in ids}
    def find(pid):
        if pid not in parent: raise ValueError('Unknown component member')
        while pid != parent[pid]:
            parent[pid] = parent[parent[pid]]; pid = parent[pid]
        return pid
    for left, right in edges:
        a, b = find(left), find(right)
        if a != b: parent[max(a, b)] = min(a, b)
    grouped = defaultdict(list)
    for pid in sorted(ids): grouped[find(pid)].append(pid)
    return sorted(grouped.values())


def take_components(groups, target, seed, role):
    """Stable subset-sum selection; never split a near-duplicate component."""
    if target < 0: raise ValueError('Negative target')
    ordered = sorted(groups, key=lambda g:canonical_sha256([seed, role, g]))
    reachable = {0:()}
    for index, group in enumerate(ordered):
        size = len(group)
        for total, selected in list(reachable.items()):
            new = total+size
            if new <= target and new not in reachable: reachable[new] = selected+(index,)
        if target in reachable: break
    if target not in reachable: raise ValueError('Cannot meet exact holdout size without splitting a component')
    chosen = set(reachable[target])
    return [g for i, g in enumerate(ordered) if i in chosen], [g for i, g in enumerate(ordered) if i not in chosen]


def assign_roles(groups, reserved_ids, holdouts, seed):
    reserved_ids = set(reserved_ids); assignments = {}; available = []
    for group in groups:
        if reserved_ids & set(group): assignments.update({pid:'dap_development_reserved' for pid in group})
        else: available.append(group)
    for role, count in holdouts.items():
        chosen, available = take_components(available, count, seed, role)
        for group in chosen: assignments.update({pid:role for pid in group})
    for group in available: assignments.update({pid:'student_pool' for pid in group})
    if len(assignments) != sum(map(len, groups)): raise ValueError('Assignment lost or duplicated questions')
    return assignments


def freeze(config_path):
    cfg = read_json(config_path); project = Path(cfg['project_root']); out = project/cfg['result_root']
    if out.exists(): raise FileExistsError(out)
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen cohort-preparation source')
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    preflight = project/cfg['preflight_root']; review = project/cfg['review_root']
    dap = project/cfg['dap_source_root']; grading_audit = project/cfg['grading_audit_root']
    bindings = [Path(config_path), Path(cfg['launch_root'])/'FROZEN.json']
    for parent in (preflight, review, dap, grading_audit):
        verify(parent/'COMPLETE.json'); bindings.append(parent/'COMPLETE.json')
    if read_json(grading_audit/'summary.json')['reference_failures'] != 0: raise ValueError('Reference grading gaps remain')
    grading_path = project/cfg['grading_config']; grading = read_json(grading_path)
    if grading != read_json(grading_audit/'provenance/grading_config.json'): raise ValueError('Grading configuration differs from audited version')
    bindings.append(grading_path)
    eligible = list(read_jsonl(review/'math_train_eligible_for_split.jsonl'))
    if len(eligible) != cfg['expected_eligible']: raise ValueError('Reviewed pool changed')
    if len({normalize_question(r['question']) for r in eligible}) != len(eligible): raise ValueError('Exact duplicate remains')
    reserved = list(read_jsonl(dap/'reserved_math_development_ids.jsonl')); reserved_ids = {r['problem_id'] for r in reserved}
    if len(reserved) != cfg['expected_dap_reserved'] or len(reserved_ids) != len(reserved): raise ValueError('DAP holdout changed')
    logging.info('Computing all MATH near-duplicate components for %d reviewed questions', len(eligible))
    near = near_matches(eligible, eligible, n=cfg['near_duplicate']['ngram'], threshold=cfg['near_duplicate']['threshold'])
    edge_by_key = {tuple(sorted([r['query_id'], r['reference_id']])):r for r in near}
    groups = components([r['problem_id'] for r in eligible], edge_by_key)
    assignments = assign_roles(groups, reserved_ids, cfg['holdout_counts'], cfg['split_seed'])
    for left, right in edge_by_key:
        if assignments[left] != assignments[right]: raise ValueError('Cross-role near-duplicate leakage')
    group_by_id = {pid:canonical_sha256(group) for group in groups for pid in group}
    raw_cfg = read_json(preflight/'config_snapshot.json'); raw = load_source(raw_cfg['sources']['math_train'])
    rows = []
    for original in eligible:
        raw_row = raw[original['source_index']]
        if raw_row['problem'] != original['question']: raise ValueError('Raw MATH metadata identity differs')
        row = {**original, 'question_role':assignments[original['problem_id']], 'near_component_id':group_by_id[original['problem_id']],
            'source_level':raw_row.get('level'), 'source_subject':raw_row.get('type'),
            'answer_spec':compile_answer_spec(original, grading)}
        rows.append(row)
    role_counts = dict(Counter(r['question_role'] for r in rows))
    for role, count in cfg['holdout_counts'].items():
        if role_counts.get(role) != count: raise ValueError('Holdout target differs')
    by_id = {r['problem_id']:r for r in rows}
    if any(by_id[pid]['question_role'] != 'dap_development_reserved' for pid in reserved_ids & set(by_id)):
        raise ValueError('DAP question reached student/parameter tuning pool')
    hard_map = {r['hard_id']:r['parent_ids'][0] for r in read_jsonl(review/'gsmhard_locked_parent_cohort.jsonl')}
    evals = {}; smoke = {}
    for dataset, expected in cfg['expected_eval_counts'].items():
        source = list(read_jsonl(preflight/'sources'/f'{dataset}.jsonl'))
        if dataset == 'gsm8k':
            smoke[dataset] = [r for r in source if r['source_index'] < 50]
            source = [r for r in source if 50 <= r['source_index'] < 1319]
        elif dataset == 'gsm8k_hard':
            source = [{**r, 'parent_problem_ids':[hard_map[r['problem_id']]], 'parent_problem_id':hard_map[r['problem_id']]}
                      for r in source if r['problem_id'] in hard_map]
        if len(source) != expected or len({r['problem_id'] for r in source}) != len(source): raise ValueError('Evaluation cohort changed: '+dataset)
        evals[dataset] = [{**r, 'question_role':'locked_evaluation', 'answer_spec':compile_answer_spec(r, grading)} for r in source]
    if set(hard_map.values()) != {r['problem_id'] for r in evals['gsm8k']}: raise ValueError('GSM-Hard parent cohort is not locked GSM8K cohort')
    original_exclusions = list(read_jsonl(review/'training_exclusions.jsonl'))
    out.mkdir(parents=True, exist_ok=False)
    for role in role_counts: write_jsonl(out/'cohorts'/f'{role}.jsonl', [r for r in rows if r['question_role'] == role])
    for name, cohort in evals.items(): write_jsonl(out/'evaluation'/f'{name}.jsonl', cohort)
    for name, cohort in smoke.items(): write_jsonl(out/'smoke'/f'{name}.jsonl', cohort)
    write_jsonl(out/'audits/near_duplicate_pairs.jsonl', list(edge_by_key.values()))
    write_jsonl(out/'audits/components.jsonl', [{'component_id':canonical_sha256(g), 'problem_ids':g,
        'question_role':assignments[g[0]]} for g in groups])
    write_jsonl(out/'audits/prior_training_exclusions.jsonl', original_exclusions)
    write_jsonl(out/'audits/role_manifest.jsonl', [{k:r[k] for k in ('problem_id','question_role','near_component_id','source_level','source_subject')} for r in rows])
    summary = {'status':'problem_roles_frozen_not_sft_protocol', 'source_questions':len(raw), 'reviewed_eligible':len(eligible),
        'prior_exclusions':len(original_exclusions), 'unique_near_pairs':len(edge_by_key), 'components':len(groups),
        'largest_component':max(map(len, groups)), 'role_counts':role_counts,
        'dap_original_reserved_ids':len(reserved_ids), 'dap_already_excluded':sorted(reserved_ids-set(by_id)),
        'dap_near_neighbors_reserved':sorted({pid for pid, role in assignments.items() if role == 'dap_development_reserved'}-reserved_ids),
        'evaluation_counts':{name:len(r) for name, r in evals.items()}, 'cross_role_near_pairs':0,
        'role_distributions':{role:{field:dict(Counter(str(r[field]) for r in rows if r['question_role'] == role))
            for field in ('source_subject','source_level')} for role in role_counts},
        'calibration_usage':'Only this role may fit ASC/dense/SAE directions or calibration statistics on MATH; source filtering still determines usable correct pairs.',
        'development_usage':'Prompt/cap/dose/ratio decisions; never student SFT. Same near component remains together.',
        'student_pool_usage':'Eligible candidate-generation problem pool, not yet common correct support or actual SFT examples.',
        'formal_training_ready':False, 'pending':['Teacher/student generation cap calibration', 'Baseline/SAE final parameters', 'Four-candidate generation, coverage and common-support audit', 'SFT hyperparameter and budget freeze']}
    save(out/'summary.json', summary); save(out/'grading_config.json', grading); save(out/'config.json', cfg)
    report = ['# MATH 第一规模固定题目角色', '',
        '此标记冻结问题角色、近似题组及评测身份；不表示教师候选、正确共同支持或 SFT 协议已完成。', '',
        '| 角色 | 唯一问题数 |', '| --- | ---: |']
    for role, count in role_counts.items(): report.append(f'| {role} | {count} |')
    report += ['', f"原池 {len(raw)} 题，先前去污染/精确去重后 {len(eligible)} 题；近似题对 {len(edge_by_key)}、最大连通组 {summary['largest_component']} 题。组内问题始终同角色，跨角色达到登记相似阈值的边为零。",
        '阈值来自已登记的词汇 5-gram Jaccard，不是语义重复或预训练污染的完整保证。组件按固定种子排序，使用确定性 subset-sum 满足留出题数，不拆分组件。',
        '64 道 DAP 长轨迹方法检查题及其连通近邻保留为开发资源，后续学生池不得使用；实际额外保留数见 summary.json。校准/开发问题不属于学生监督。',
        '五评测集保持已审计版本。GSM8K test[:50] 保留 smoke、test[50:1319] 锁定评测；GSM8K-Hard 与锁定父题一一对应。MATH-500 是同域留出，不能称为相对 MATH 训练的独立 OOD。',
        '7,500 是来源规模，最终训练问题数必须另报正确共同交集后的真实数目，不重复样本凑数。生成 cap、方法参数和学生预算待冻结。', '']
    (out/'report_zh.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json', bindings+[p for p in out.rglob('*') if p.is_file()], formal_training_ready=False,
         stage='math_problem_role_and_evaluation_identity_freeze')
    logging.info('MATH role freeze: %s', summary)
