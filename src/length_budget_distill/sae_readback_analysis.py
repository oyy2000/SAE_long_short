"""Audited descriptive summaries of fresh discovery and development SAE probes."""
from collections import defaultdict, Counter
from pathlib import Path
import json
import re
import statistics

import numpy as np

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import verify, save, seal
from .factorial import canonical_sha256
from .baseline_method_analysis import audit_cohort
from .sae_mechanism_readback import response_regions, probe_positions
from .utility_analysis import paired_question_bootstrap
from .factorial_analysis import holm_adjust


def audit_probes(rows, expected_ids, directions, doses, source_by_id, pattern):
    """Require the complete registered branch grid and identical baseline features."""
    by_key = {}
    for row in rows:
        key = (row['problem_id'], row['position_name'], row['direction'], row['requested_rho'])
        if key in by_key: raise ValueError('Duplicate probe measurement')
        if row['problem_id'] not in expected_ids: raise ValueError('Unexpected probe question')
        source = source_by_id[row['problem_id']]
        if row['response_text_sha256'] != canonical_sha256(source['solution']): raise ValueError('Changed probe response')
        if not row['incoming_state_identical_across_branches']: raise ValueError('Mismatched incoming states')
        numeric = [row[k] for k in ('forward_kl', 'actual_delta_fraction', 'answer_next_probability', 'eos_next_probability')]
        if not all(np.isfinite(numeric)): raise ValueError('Nonfinite probe measurement')
        if row['forward_kl'] < -1e-5: raise ValueError('KL is negative beyond floating-point tolerance')
        for k in ('answer_next_probability', 'eos_next_probability'):
            if not 0 <= row[k] <= 1.000001: raise ValueError('Invalid next-token probability')
        by_key[key] = row
    expected = set()
    for pid in expected_ids:
        source = source_by_id[pid]
        positions = probe_positions(response_regions(source['solution'], source['token_offsets'],
            'unmodified_generated', pattern)[0])
        for position, token_index in positions.items():
            base_key = (pid, position, 'zero', 0.)
            if base_key not in by_key: raise ValueError('Missing zero branch')
            base = by_key[base_key]; expected.add(base_key)
            if base['processed_response_token_index'] != token_index: raise ValueError('Wrong zero prediction position')
            if base['actual_delta_fraction'] != 0 or base['feature_before'] != base['feature_after'] or base['forward_kl'] != 0:
                raise ValueError('Changed zero branch')
            for direction in directions:
                for dose in doses:
                    key = (pid, position, direction, dose); expected.add(key)
                    if key not in by_key: raise ValueError('Missing direction/dose branch')
                    row = by_key[key]
                    if row['processed_response_token_index'] != token_index: raise ValueError('Wrong prediction position')
                    for field in ('prefix_tokens', 'prefix_token_sha256', 'feature_ids', 'feature_before'):
                        if row[field] != base[field]: raise ValueError('Branch baseline identity changed: '+field)
    if set(by_key) != expected: raise ValueError('Unexpected direction/dose/position')
    return by_key


def analyze(config_path):
    from transformers import AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg['measurement_root']); out = Path(cfg['result_root'])
    if Path(__file__).resolve().parents[2] != Path(cfg['code_root']): raise ValueError('Use frozen analysis code')
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json'); verify(root/'directions/COMPLETE.json')
    protocol = read_json(root/'protocol/frozen_config.json'); parent = Path(protocol['input_root'])
    verify(parent/'protocol/FROZEN.json'); verify(parent/'baseline/merged/COMPLETE.json')
    questions = list(read_jsonl(parent/'inputs/questions.jsonl'))
    discovery = [q['problem_id'] for q in questions if q['question_split'] == 'discovery']
    dev = [q['problem_id'] for q in questions if q['question_split'] == 'dev']
    if len(discovery) != protocol['expected_counts']['discovery'] or len(dev) != protocol['expected_counts']['dev']:
        raise ValueError('Registered cohort changed')
    features = []; probes = []; markers = [root/'protocol/FROZEN.json', root/'directions/COMPLETE.json']
    source_by_id = {r['problem_id']: r for r in read_jsonl(parent/'baseline/merged/predictions.jsonl') if r['problem_id'] in set(dev)}
    tok = AutoTokenizer.from_pretrained(protocol['teacher']['snapshot_path'], local_files_only=True)
    for row in source_by_id.values():
        row['token_offsets'] = tok(row['solution'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    for shard in range(protocol['shards']):
        ext = root/'extraction/discovery'/f'shard_{shard:02d}'; probe = root/'probes/dev'/f'shard_{shard:02d}'
        for path in (ext/'COMPLETE.json', probe/'COMPLETE.json'):
            verify(path); markers.append(path)
        ext_rows = list(read_jsonl(ext/'features.jsonl')); probe_rows = list(read_jsonl(probe/'measurements.jsonl'))
        expected_disc = [pid for i, pid in enumerate(discovery) if i % protocol['shards'] == shard]
        for variant in protocol['required_variants']:
            audit_cohort([r for r in ext_rows if r['variant'] == variant], expected_disc)
        expected_dev = [pid for i, pid in enumerate(dev) if i % protocol['shards'] == shard]
        audit_probes(probe_rows, expected_dev, cfg['directions'], protocol['doses'], source_by_id, protocol['answer_heading_pattern'])
        features.extend(ext_rows); probes.extend(probe_rows)
    ids = protocol['sae']['short_features']; per_question = []; region_summary = []
    texts = {(r['problem_id'], r['variant']): r['text'] for r in read_jsonl(parent/'inputs/reference_variants.jsonl') if r['problem_id'] in set(discovery)}
    texts.update({(r['problem_id'], 'unmodified_generated'): r['solution'] for r in read_jsonl(parent/'baseline/merged/predictions.jsonl') if r['problem_id'] in set(discovery)})
    seen = set()
    for row in features:
        key = (row['problem_id'], row['variant'])
        if key in seen or key not in texts: raise ValueError('Unexpected/duplicate feature row')
        seen.add(key); text = texts[key]
        if canonical_sha256(text) != row['text_sha256']: raise ValueError('Feature text identity changed')
        slots = [row['feature_ids'].index(i) for i in ids]
        sums = {region: sum(stats['feature_sum'][i] for i in slots) for region, stats in row['region_statistics'].items()}
        total = sum(sums.values()); token_count = len(row['token_ids'])
        spans = [m.span() for m in re.finditer('Answer', text)]
        literal_mass = 0.
        for event in row['events']:
            if event['feature_id'] not in ids: continue
            left, right = row['token_offsets'][event['position']]
            if any(left < b and right > a for a, b in spans): literal_mass += event['activation']
        event_mass = sum(e['activation'] for e in row['events'] if e['feature_id'] in ids)
        if not np.isclose(event_mass, total, rtol=1e-6, atol=1e-5): raise ValueError('Feature events do not reconstruct regional mass')
        record = {'problem_id': row['problem_id'], 'variant': row['variant'], 'tokens': token_count,
            'short_feature_mass': total, 'short_mean_per_feature_token': total/(len(ids)*token_count),
            'literal_Answer_mass': literal_mass, **{region+'_mass': value for region, value in sums.items()}}
        for region, stats in row['region_statistics'].items():
            record[region+'_mean'] = sums[region]/(len(ids)*stats['tokens']) if stats['tokens'] else None
        per_question.append(record)
    if seen != set(texts): raise ValueError('Missing optional/reference feature variants')
    for variant in sorted({r['variant'] for r in features}):
        subset = [r for r in per_question if r['variant'] == variant]
        total = sum(r['short_feature_mass'] for r in subset)
        entry = {'variant': variant, 'questions': len(subset), 'mean_tokens': statistics.fmean(r['tokens'] for r in subset),
            'mean_short_activation': statistics.fmean(r['short_mean_per_feature_token'] for r in subset),
            'literal_Answer_mass_fraction': sum(r['literal_Answer_mass'] for r in subset)/total if total else None}
        for region in ('reasoning_body', 'answer_marker', 'answer_suffix'):
            values = [r[region+'_mean'] for r in subset if r[region+'_mean'] is not None]
            entry[region+'_mean'] = statistics.fmean(values) if values else None
            entry[region+'_questions'] = len(values)
            entry[region+'_mass_fraction'] = sum(r[region+'_mass'] for r in subset)/total if total else None
        region_summary.append(entry)
    contrasts = []
    for left_name, right_name in cfg['discovery_contrasts']:
        for metric in ('short_mean_per_feature_token', 'reasoning_body_mean'):
            left = {r['problem_id']: r[metric] for r in per_question if r['variant'] == left_name and r[metric] is not None}
            right = {r['problem_id']: r[metric] for r in per_question if r['variant'] == right_name and r[metric] is not None}
            common = sorted(set(left)&set(right))
            result = paired_question_bootstrap({k:left[k] for k in common}, {k:right[k] for k in common},
                samples=cfg['bootstrap_samples'], seed=cfg['bootstrap_seed'])
            contrasts.append({'left': left_name, 'right': right_name, 'metric': metric,
                'missing_from_pair': len(discovery)-len(common), **result})
    adjusted = holm_adjust([r['bootstrap_p_value'] for r in contrasts])
    for row, value in zip(contrasts, adjusted): row['holm_p_value'] = value
    zero = {(r['problem_id'], r['position_name']): r for r in probes if r['direction'] == 'zero'}
    grouped = defaultdict(list)
    for row in probes:
        if row['direction'] == 'zero': continue
        baseline = zero[(row['problem_id'], row['position_name'])]
        grouped[(row['position_name'], row['direction'], row['requested_rho'])].append({
            'target_sum_change': row['target_sum_after']-row['target_sum_before'],
            'answer_probability_change': row['answer_next_probability']-baseline['answer_next_probability'],
            'eos_probability_change': row['eos_next_probability']-baseline['eos_next_probability'],
            **{k:row[k] for k in ('forward_kl', 'actual_delta_fraction', 'nontarget_l1_change', 'active_jaccard')}})
    cells = [{'position': pos, 'direction': direction, 'rho': rho, 'questions': len(values),
        **{key:statistics.fmean(r[key] for r in values) for key in values[0]}}
        for (pos, direction, rho), values in sorted(grouped.items())]
    summary = {'formal_claim_allowed': False, 'discovery_questions': len(discovery), 'development_questions': len(dev),
        'feature_records': len(features), 'probe_measurements': len(probes),
        'variant_summary': region_summary, 'discovery_contrasts': contrasts,
        'local_causal_cells': cells, 'directions': read_json(root/'directions/definition.json'),
        'minimum_recorded_kl': min(r['forward_kl'] for r in probes),
        'local_probe_statistics': 'Descriptive means of registered complete dev cells. No rollout accuracy/length or inferential claim.',
        'confirmation_feature_measurements_inspected': False, 'claim_boundary': cfg['claim_boundary']}
    out.mkdir(parents=True, exist_ok=False); save(out/'summary.json', summary)
    write_jsonl(out/'discovery_question_metrics.jsonl', per_question); write_jsonl(out/'local_causal_cells.jsonl', cells)
    plot(out, region_summary, cells, cfg)
    report = ['# SAE 新题区域与同状态读回', '', '![区域读回](discovery_regions.png)', '',
        '400 道 discovery 题用于参考变体测量和 dense/答案格式方向拟合；300 道 dev 题用于固定原始前缀上的局部因果读回。300 道 confirmation 的特征未读取。', '',
        '所有非零条件均从独立复制的同一前缀 cache 开始，逐分支核验相同 incoming hidden state；零剂量特征和输出概率保持不变。隐藏状态用 BF16，SAE 采用原层 17 / TopK-64，decoder 先转 BF16 再以 FP32 求和归一化。', '',
        '| 参考变体 | 题数 | 平均 short 激活 / feature / token | 字面 Answer 占 short 激活质量 |',
        '| --- | ---: | ---: | ---: |']
    for r in region_summary:
        share = f"{100*r['literal_Answer_mass_fraction']:.2f}%" if r['literal_Answer_mass_fraction'] is not None else 'undefined'
        report.append(f"| {r['variant']} | {r['questions']} | {r['mean_short_activation']:.5f} | {share} |")
    report += ['', '![局部剂量读回](development_local_effects.png)', '',
        f"共 {len(probes):,} 条局部测量；全词表 forward KL 最小值 {summary['minimum_recorded_kl']:.3g}（FP32 舍入可能出现约 1e-7 的负数，原值保留）。Answer 概率只汇总登记的单 token 形式；EOS 使用模型生成配置中的终止 token。", '',
        'dense 为每题官方参考平均响应状态减原始生成平均响应状态，再等题权重平均；它仍可能混入内容、风格和长度差异。答案格式方向在同一正文后的 Answer 与 Result 标题末 token 上求差。',
        '区域汇总为逐题平均；字面 Answer 占比为跨题激活质量汇总。错误计算变体缺失题单独报告，不能当成验证了整条推理错误。原始生成文本重新分词，不冒充采样时保存的 token IDs。',
        '开发集图为描述性均值，输出完整方向/剂量/位置网格；不据此声称缩短整条生成、改善准确率或提高学生蒸馏质量。后续仍需冻结生成剂量、时间窗口、特征数和独立 confirmation 协议。', '', cfg['claim_boundary'], '']
    (out/'report_zh.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json', [Path(config_path), Path(cfg['launch_root'])/'FROZEN.json', *markers, *sorted(out.glob('*'))],
         stage='sae_discovery_and_dev_readback_analysis', formal_claim_allowed=False)


def plot(out, regions, cells, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
    order = ['reference', 'result_marker', 'no_marker', 'neutral_note', 'answer_note', 'wrong_calculation', 'unmodified_generated']
    lookup = {r['variant']: r for r in regions}; x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(x, [lookup[k]['mean_short_activation'] for k in order], color='#2166AC')
    axes[0].set_ylabel('Mean short-feature activation / feature / token')
    bottom = np.zeros(len(order))
    for region, color in [('reasoning_body', '#6c757d'), ('answer_marker', '#2166AC'), ('answer_suffix', '#e9c46a')]:
        values = np.array([lookup[k][region+'_mass_fraction'] or 0. for k in order])
        axes[1].bar(x, values, bottom=bottom, color=color, label=region.replace('_', ' ')); bottom += values
    axes[1].set_ylabel('Fraction of pooled short-feature activation'); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, [k.replace('_', '\n') for k in order], fontsize=8)
    fig.suptitle('SAE: 400 discovery questions; historical eight short features')
    for suffix in ('png', 'pdf'): fig.savefig(out/f'discovery_regions.{suffix}', dpi=180)
    plt.close(fig)
    positions = ['first_response_predictor', 'body_middle', 'before_marker', 'after_marker', 'response_end']
    metrics = [('target_sum_change', 'Change in summed short activation'), ('forward_kl', 'Next-token forward KL'),
               ('eos_probability_change', 'Change in EOS probability')]
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), constrained_layout=True)
    colors = ['#2166AC', '#b2182b', '#e69f00', '#6c757d', '#009e73', '#9467bd']
    for col, position in enumerate(positions):
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for direction, color in zip(cfg['directions'], colors):
                subset = sorted([r for r in cells if r['position'] == position and r['direction'] == direction], key=lambda r:r['rho'])
                ax.plot([0.]+[r['rho'] for r in subset], [0.]+[r[metric] for r in subset], marker='.', color=color,
                    label=direction, linewidth=1.3)
            ax.axhline(0., color='#bbbbbb', linewidth=.5)
            if col == 0: ax.set_ylabel(label)
            if row == 0: ax.set_title(position.replace('_', ' '), fontsize=9)
            if row == 2: ax.set_xlabel('Requested relative hidden-norm dose')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('SAE local readback: 300 development questions; descriptive means')
    for suffix in ('png', 'pdf'): fig.savefig(out/f'development_local_effects.{suffix}', dpi=170)
    plt.close(fig)
