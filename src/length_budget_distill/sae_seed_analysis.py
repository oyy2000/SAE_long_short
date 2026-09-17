"""Audit repeated SAE discovery, align candidates and repeat lexical diagnostics.

Reuse mutual_decoder_matches and run_diagnostic. Matching is descriptive and
does not replace each dictionary's original confirmed-feature selection.
"""
from itertools import combinations
from pathlib import Path
import csv
import logging

import numpy as np

from .experiment_io import read_json
from .records import write_jsonl
from .factorial import file_sha256
from .ncsu_reproduction import save, seal, verify
from .sae_seed_features import feature_selection
from .sae_ablation import mutual_decoder_matches
from .answer_marker_diagnostic import run_diagnostic

CODE = Path(__file__).resolve().parents[2]
IDENTITY = ('problem_id', 'corpus_index', 'analysis_length_label', 'solution_token_count')


def aligned_test_activations(rows, candidates, reference=None):
    """Align identical held-out traces; reject silent intersection or duplicates."""
    selected = [r for r in rows if r['question_split'] == 'test']
    lookup = {r['trace_id']: r for r in selected}
    if len(lookup) != len(selected) or not lookup:
        raise ValueError('Empty or duplicate held-out trace support')
    identity = {tid: {key: row[key] for key in IDENTITY} for tid, row in lookup.items()}
    if reference is not None and identity != reference:
        raise ValueError('Cross-seed trace support or metadata changed')
    matrix = np.array([[float(lookup[tid][f"feature_{c['feature_id']}_token_mean_activation"])
                        for c in candidates] for tid in sorted(lookup)], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite feature activations')
    return matrix, identity


def confirmed_subset(candidates, count):
    ids = feature_selection(candidates, count)['features']['short']
    if len(ids) != count:
        raise ValueError('Insufficient confirmed short features; no fallback')
    return ids


def analyze(config_path):
    import torch
    from safetensors.torch import load_file
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    if CODE != Path(cfg['code_root']):
        raise ValueError('Use frozen seed-analysis source')
    if root.exists():
        raise FileExistsError(root)
    seeds = Path(cfg['seed_root']); features = Path(cfg['features_root'])
    markers = [Path(cfg['launch_root'])/'FROZEN.json', seeds/'protocol/FROZEN.json',
               features/'protocol/FROZEN.json', features/'protocol/SOURCES.json']
    for seed in cfg['seeds'][1:]:
        markers += [seeds/'training'/f'seed_{seed}'/'COMPLETE.json',
                    features/'execution'/f'seed_{seed}'/'COMPLETE.json']
    for marker in markers:
        verify(marker)
    legacy = read_json(seeds/'inputs/seed17_reuse.json')
    parent = Path(read_json(seeds/'protocol/frozen_config.json')['parent_root'])
    root.mkdir(parents=True)
    reference = None; runs = {}; rows = []; directions = {}; bindings = []
    diagnostic_base = read_json(cfg['region_config_path'])
    for seed in cfg['seeds']:
        if seed == cfg['seeds'][0]:
            score = parent/'sae_features/feature_scores/layer_17_k_064'
            protocol = parent/'sae_features/protocol/frozen_protocol.json'
            metrics_path = Path(legacy['metrics_path']); model = Path(legacy['model_path'])
        else:
            score = features/'feature_scores'/f'seed_{seed}'
            protocol = features/'inputs'/f'seed_{seed}'/'feature_config.json'
            metrics_path = seeds/'training'/f'seed_{seed}'/'training_metrics.json'
            model = Path(read_json(metrics_path)['model_path'])
        summary = read_json(score/'scoring_summary.json')
        if file_sha256(model) != summary['input_evidence']['checkpoint_sha256']:
            raise ValueError('Scored SAE checkpoint changed')
        candidates = read_json(score/'discovered_features.json')['candidates']
        ids = [int(c['feature_id']) for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate candidate feature IDs')
        with (score/'selected_trace_metrics.csv').open() as handle:
            activation, reference = aligned_test_activations(list(csv.DictReader(handle)), candidates, reference)
        if len(activation) != summary['confirmation_trace_count']:
            raise ValueError('Held-out activation count differs from original scorer')
        decoder = load_file(str(model))['decoder_weight']
        candidate_decoders = decoder[ids].float().numpy()
        selected = confirmed_subset(candidates, cfg['supplement_feature_count'])
        vector = decoder[selected].to(torch.bfloat16).float().sum(0)
        if not torch.isfinite(vector).all() or float(vector.norm()) <= 0:
            raise ValueError('Invalid selected direction')
        directions[f'seed_{seed}_top_{len(selected)}'] = (vector/vector.norm()).numpy()
        if seed == cfg['seeds'][0]:
            original = confirmed_subset(candidates, cfg['original_feature_count'])
            vector = decoder[original].to(torch.bfloat16).float().sum(0)
            directions[f'seed_{seed}_top_{len(original)}'] = (vector/vector.norm()).numpy()
        del decoder
        runs[seed] = dict(candidates=candidates, decoders=candidate_decoders, activations=activation)
        region_cfg = {**diagnostic_base, 'scoring_root':str(score), 'feature_protocol_path':str(protocol),
                      'result_root':str(root/'regions'/f'seed_{seed}')}
        region_path = root/'inputs'/f'regions_seed_{seed}.json'; save(region_path, region_cfg)
        region = run_diagnostic(region_path, Path(cfg['project_root']))
        metrics = {k:v for k,v in read_json(metrics_path)['final_metrics']['test'].items()
                   if isinstance(v,(int,float))}
        rows.append({'seed':seed, 'confirmed_short':summary['short_confirmed_count'],
            'confirmed_long':summary['long_confirmed_count'],
            'original_eight_short_feasible':summary['short_confirmed_count'] >= cfg['original_feature_count'],
            'supplement_short_ids':selected, 'test_metrics':metrics,
            'short_answer_mass_share_range':region['short_answer_mass_share_range'],
            'short_top_token_all_answer':region['short_top_token_all_answer'],
            'heldout_traces':len(activation), 'heldout_questions':summary['confirmation_question_count']})
        bindings += [model, metrics_path, protocol, score/'scoring_summary.json']
        logging.info('Seed %d: %d short confirmed, supplementary IDs %s',seed,rows[-1]['confirmed_short'],selected)
    pairs = []; pair_counts = []
    for a,b in combinations(cfg['seeds'],2):
        left,right = runs[a],runs[b]
        matches = mutual_decoder_matches(left_name=f'seed_{a}',right_name=f'seed_{b}',
            left_candidates=left['candidates'],right_candidates=right['candidates'],
            left_decoders=left['decoders'],right_decoders=right['decoders'],
            left_test_activations=left['activations'],right_test_activations=right['activations'],
            minimum_decoder_cosine=cfg['descriptive_matching']['minimum_decoder_cosine'],
            minimum_activation_correlation=cfg['descriptive_matching']['minimum_activation_correlation'],
            require_same_direction=True)
        for match in matches:
            match['descriptive_threshold_match'] = match.pop('stable_match')
            match['both_originally_confirmed'] = bool(left['candidates'][match['left_slot']]['confirmed']
                                                      and right['candidates'][match['right_slot']]['confirmed'])
        pairs += matches
        pair_counts.append({'left_seed':a,'right_seed':b,'candidate_rows':len(matches),
            'confirmed_short_descriptive_matches':sum(m['descriptive_threshold_match'] and m['both_originally_confirmed']
                                                      and m['left_direction']=='short' for m in matches)})
    names=list(directions); matrix=np.stack([directions[name] for name in names])
    np.savez(root/'directions.npz',**directions)
    save(root/'direction_cosines.json',{'names':names,'cosine_matrix':(matrix@matrix.T).tolist(),
        'construction':'Original decoder BF16 rounding followed by FP32 row sum and unit normalization, here on CPU.',
        'selection':'Dictionary-specific confirmed discovery order. Top 3 is post-count exploratory adaptation, not an eight-feature replication.'})
    write_jsonl(root/'candidate_matches.jsonl',pairs); write_jsonl(root/'seed_summary.jsonl',rows)
    save(root/'summary.json',{'seeds':rows,'pair_counts':pair_counts,'trace_support_identical':True,
        'descriptive_matching':cfg['descriptive_matching'],'claim_boundary':cfg['claim_boundary'],
        'original_eight_feature_generation_feasible_for_all_seeds':all(r['original_eight_short_feasible'] for r in rows),
        'generation_stability_complete':False})
    plot(root,rows,names,matrix@matrix.T)
    report=['# SAE seeds：原规则复验与描述性匹配','',
        '三个新 SAE 均完成 1,500 步训练；原 seed 17 在输入、训练源码、运行依赖和权重验证后复用。',
        '原规则分别确认 8、3、5、7 个 short features。新增三个 seed 均不足八个，不补入未确认特征，也不按 cosine 匹配替换。',
        '下表与匹配均来自原 132 道 held-out 问题、1,036 条相同轨迹，是字典/特征复验，尚非新生成稳定性。','',
        '![SAE seed overview](seed_overview.png)','',
        '| Seed | Confirmed short | Confirmed long | Test MSE | Answer mass range |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lo,hi=row['short_answer_mass_share_range']
        report.append(f"| {row['seed']} | {row['confirmed_short']} | {row['confirmed_long']} | {row['test_metrics']['mse']:.6f} | {lo:.2%}–{hi:.2%} |")
    report += ['','全部候选的双向 decoder 最近邻与相同 held-out trace 激活相关复用 `mutual_decoder_matches`；',
        '0.5 cosine / 0.5 correlation 沿用仓库历史门槛，只作描述，完整值保留于 candidate_matches.jsonl。',
        '逐 seed 的 Answer/body/suffix 与去除 Answer 后的配对效应复用 `answer_marker_diagnostic`，并逐事件重构全序列及前 64 tokens 的激活均值。',
        '额外统一取前三个已确认特征，是看到入选数量后的探索性适配。已保存四个方向及原 seed17 八特征参考；不能称为原八特征方法跨 seed 成功复现。',
        '计数、词汇关联、decoder cosine 和重构质量都不替代生成准确率—长度实验或学生 SFT。']
    (root/'report_zh.md').write_text('\n'.join(report)+'\n')
    seal(root/'COMPLETE.json',[Path(config_path),*markers,*bindings,*[p for p in root.rglob('*') if p.is_file()]],
         stage='sae_seed_feature_analysis',generation_stability_complete=False)


def plot(root,rows,names,cosines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    x=np.arange(len(rows))
    axes[0].bar(x,[r['confirmed_short'] for r in rows],color='#2166AC')
    axes[0].axhline(8,color='#B2182B',ls='--',label='Original eight-feature requirement')
    axes[0].set(xticks=x,xticklabels=[str(r['seed']) for r in rows],ylim=(0,10),
                xlabel='SAE seed',ylabel='Originally confirmed short features')
    axes[0].legend(frameon=False,fontsize=8)
    im=axes[1].imshow(cosines,vmin=-1,vmax=1,cmap='RdBu')
    labels=[name.replace('seed_','S').replace('_top_',' / ') for name in names]
    axes[1].set(xticks=np.arange(len(names)),xticklabels=labels,yticks=np.arange(len(names)),
                yticklabels=labels,title='Selected direction cosine; descriptive')
    axes[1].tick_params(axis='x',labelrotation=35)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[1].text(j,i,f'{cosines[i,j]:.2f}',ha='center',va='center',fontsize=8,
                         color='white' if abs(cosines[i,j])>.65 else 'black')
    fig.colorbar(im,ax=axes[1],shrink=.8)
    for suffix in ('png','pdf'):fig.savefig(root/f'seed_overview.{suffix}',dpi=180)
    plt.close(fig)
