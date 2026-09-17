"""Post-hoc region diagnostics from complete, previously scored SAE events.

Extends the event-mass analysis in scripts/2_12_render_confirmed_feature_token_anatomy.py
with dynamic feature counts, literal-token removal, and aligned answer-body regions.
Uses the shared paired-feature statistics; never recomputes or changes discovery.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .experiment_io import read_json, write_json_exclusive
from .factorial import file_sha256, read_key_value_marker
from .sae_feature_analysis import paired_feature_statistics, holm_adjust


def region_masks(text, offsets, pattern):
    """Assign non-overlapping token regions using the first explicit answer heading.

    Tokens crossing a boundary belong to the later region. No heading means all
    tokens are body. This is an offline text diagnostic, not a causal text edit.
    """
    match = re.search(pattern, text)
    regions = np.zeros(len(offsets), dtype=np.int8)
    if match:
        # Historical diagnostics name the heading; pilot patterns match it whole.
        start = match.start('marker') if 'marker' in match.re.groupindex else match.start()
        end = match.end()
        for i, (left, right) in enumerate(offsets):
            if right <= start:
                regions[i] = 0
            elif left < end:
                regions[i] = 1
            else:
                regions[i] = 2
    return regions, match is not None


def event_key(event):
    return str(event['trace_id']), int(event['feature_id']), int(event['position'])


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(config_path, project):
    from tokenizers import Tokenizer
    config_path, project = Path(config_path), Path(project)
    cfg = read_json(config_path)
    source = project / cfg['source_root']
    run = project / cfg['scoring_root'] if 'scoring_root' in cfg else source / 'feature_scores' / f"layer_{cfg['layer_index']:02d}_k_{cfg['k']:03d}"
    out = project / cfg['result_root']
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    marker = read_key_value_marker(run / 'FEATURE_SCORING_COMPLETE')
    if marker.get('status') != 'complete' or marker['summary_sha256'] != file_sha256(run / 'scoring_summary.json'):
        raise ValueError('Unverified parent scoring marker')
    summary = read_json(run / 'scoring_summary.json')
    needed = ['discovered_features.json', 'selected_trace_metrics.csv', 'selected_feature_token_events.jsonl']
    hashes = {str(config_path): file_sha256(config_path)}
    for name in needed:
        p = run / name
        if file_sha256(p) != summary['artifacts'][name]['sha256']:
            raise ValueError(f'Parent artifact changed: {name}')
        hashes[str(p)] = file_sha256(p)
    protocol_path = project / cfg['feature_protocol_path'] if 'feature_protocol_path' in cfg else source / 'protocol/frozen_protocol.json'
    if file_sha256(protocol_path) != summary['config_sha256']:
        raise ValueError('Parent feature protocol changed')
    protocol = read_json(protocol_path)
    parent_path = Path(protocol['parent_sae']['config_path'])
    if file_sha256(parent_path) != protocol['parent_sae']['config_sha256']:
        raise ValueError('Parent SAE protocol changed')
    parent = read_json(parent_path)
    tokenizer_path = Path(parent['teacher']['snapshot_path']) / 'tokenizer.json'
    if file_sha256(tokenizer_path) != parent['teacher']['tokenizer_json_sha256']:
        raise ValueError('Tokenizer hash mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    candidates = [c for c in read_json(run / needed[0])['candidates'] if c['confirmed']]
    features = [int(c['feature_id']) for c in candidates]
    feature_lookup = {f: i for i, f in enumerate(features)}
    with (run / needed[1]).open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['question_split'] == cfg['split']]
    lookup = {r['trace_id']: i for i, r in enumerate(rows)}
    if len(lookup) != len(rows) or len(rows) != summary['confirmation_trace_count']:
        raise ValueError('Missing/duplicate trace metrics')
    corpus_path = Path(protocol['parent_sae']['corpus_path'])
    with corpus_path.open() as handle:
        corpus = {r['trace_id']: r for r in map(json.loads, handle) if r['trace_id'] in lookup}
    if set(corpus) != set(lookup):
        raise ValueError('Missing corpus records')
    hashes.update({str(p):file_sha256(p) for p in [protocol_path,parent_path,tokenizer_path,corpus_path,run/'FEATURE_SCORING_COMPLETE',run/'scoring_summary.json']})
    lengths = np.array([int(r['solution_token_count']) for r in rows])
    n, f = len(rows), len(features)
    region_sizes = np.zeros((n, 3), dtype=np.int64)
    ids, regions, literal_masks = [], [], []
    headings = []
    for i, row in enumerate(rows):
        record = corpus[row['trace_id']]
        for field in ['problem_id','question_split','analysis_length_label']:
            if str(record[field]) != row[field]: raise ValueError(f'Trace metadata mismatch: {field}')
        encoded = tokenizer.encode(record['solution'], add_special_tokens=False)
        if len(encoded.ids) != lengths[i]: raise ValueError('Token length mismatch')
        region, has_heading = region_masks(record['solution'], encoded.offsets, cfg['answer_heading_pattern'])
        ids.append(encoded.ids); regions.append(region); headings.append(has_heading)
        literal_masks.append(np.array([tokenizer.id_to_token(t) == cfg['answer_token_text'] for t in encoded.ids]))
        region_sizes[i] = np.bincount(region, minlength=3)
    mass = np.zeros((n, f, 3), dtype=np.float64)
    literal_mass = np.zeros((n, f), dtype=np.float64)
    early_mass = np.zeros((n, f), dtype=np.float64)
    token_mass = [Counter() for _ in features]
    seen = set()
    event_count = 0
    with (run / needed[2]).open() as handle:
        for raw in handle:
            event = json.loads(raw)
            if int(event['feature_id']) not in feature_lookup: continue
            key = event_key(event)
            if key in seen: raise ValueError(f'Duplicate event: {key}')
            seen.add(key)
            if event['split'] != cfg['split'] or key[0] not in lookup: raise ValueError('Unexpected event support')
            i, j, pos = lookup[key[0]], feature_lookup[key[1]], key[2]
            if not 0 <= pos < lengths[i] or ids[i][pos] != int(event['token_id']): raise ValueError('Misaligned token event')
            if event['problem_id'] != rows[i]['problem_id'] or event['analysis_length_label'] != rows[i]['analysis_length_label']: raise ValueError('Misaligned event metadata')
            value = float(event['activation'])
            if not np.isfinite(value) or value <= 0: raise ValueError('Invalid sparse activation')
            mass[i,j,regions[i][pos]] += value
            if literal_masks[i][pos]: literal_mass[i,j] += value
            if pos < 64: early_mass[i,j] += value
            token_mass[j][int(event['token_id'])] += value
            event_count += 1
    total = mass.sum(axis=2)
    expected = np.array([[float(r[f'feature_{fid}_token_mean_activation']) for fid in features] for r in rows])
    actual = total / lengths[:,None]
    if not np.allclose(actual, expected, atol=cfg['event_mass_atol'], rtol=cfg['event_mass_rtol']):
        raise ValueError('Sparse events do not reconstruct saved full means')
    early = early_mass / np.minimum(lengths,64)[:,None]
    expected_early = np.array([[float(r[f'feature_{fid}_first_64_token_mean_activation']) for fid in features] for r in rows])
    if not np.allclose(early, expected_early, atol=cfg['event_mass_atol'], rtol=cfg['event_mass_rtol']): raise ValueError('Sparse events do not reconstruct early means')
    literal_counts = np.array([m.sum() for m in literal_masks])
    if (region_sizes[:,0] == 0).any() or (lengths <= literal_counts).any():
        raise ValueError('Empty body/support would require an explicit paired-support policy')
    matrices = {'whole_trace': actual, 'without_answer_token': (total-literal_mass)/(lengths-literal_counts)[:,None], 'reasoning_body':mass[:,:,0]/region_sizes[:,0,None]}
    stats = {name:paired_feature_statistics(v,rows) for name,v in matrices.items()}
    # Existing features and test questions were already examined: descriptive intervals only.
    effects=[]
    for name, stat in stats.items():
        for j, candidate in enumerate(candidates):
            effects.append({'feature_id':features[j],'direction':candidate['direction'],'metric':name,'paired_questions':stat['question_count'],'effect':float(stat['effect'][j]),'ci_low':float(stat['ci_low'][j]),'ci_high':float(stat['ci_high'][j]),'paired_d':float(stat['paired_d'][j]),'descriptive_p':float(stat['p_value'][j])})
    adjusted=holm_adjust([r['descriptive_p'] for r in effects])
    for row,p in zip(effects,adjusted):row['descriptive_holm_p']=float(p)
    features_out=[]
    for j,c in enumerate(candidates):
        top_id,top_mass=token_mass[j].most_common(1)[0]
        denom=total[:,j].sum()
        features_out.append({'feature_id':features[j],'direction':c['direction'],'top_token':tokenizer.id_to_token(top_id),'top_token_mass_share':float(top_mass/denom),'answer_token_mass_share':float(literal_mass[:,j].sum()/denom),'body_mass_share':float(mass[:,j,0].sum()/denom),'marker_mass_share':float(mass[:,j,1].sum()/denom),'suffix_mass_share':float(mass[:,j,2].sum()/denom),'original_paired_d':float(stats['whole_trace']['paired_d'][j]),'body_paired_d':float(stats['reasoning_body']['paired_d'][j]),'without_answer_paired_d':float(stats['without_answer_token']['paired_d'][j])})
    trace_out=[]
    for i,r in enumerate(rows):
        trace_out.append({**{k:r[k] for k in ['trace_id','problem_id','analysis_length_label']},'tokens':int(lengths[i]),'body_tokens':int(region_sizes[i,0]),'marker_tokens':int(region_sizes[i,1]),'suffix_tokens':int(region_sizes[i,2]),'literal_answer_count':int(literal_counts[i]),'explicit_heading':bool(headings[i]),'heading_before64':bool(headings[i] and region_sizes[i,0]<64)})
    out.mkdir(parents=True,exist_ok=False)
    write_json_exclusive(out/'config_snapshot.json',cfg)
    write_csv(out/'feature_regions.csv',features_out)
    write_csv(out/'paired_effects.csv',effects)
    write_csv(out/'trace_support.csv',trace_out)
    short=[r for r in features_out if r['direction']=='short']
    short_range=[min(r['answer_token_mass_share'] for r in short),max(r['answer_token_mass_share'] for r in short)]
    manifest={'status':'complete','formal_claim_allowed':False,'claim_boundary':cfg['claim_boundary'],'traces':n,'problems':len({r['problem_id'] for r in rows}),'features':f,'short_features':len(short),'selected_nonzero_events':event_count,'heading_count':sum(headings),'heading_before64_count':sum(r['heading_before64'] for r in trace_out),'short_answer_mass_share_range':short_range,'short_top_token_all_answer':all(r['top_token']==cfg['answer_token_text'] for r in short),'full_mean_max_abs_reconstruction_error':float(np.abs(actual-expected).max()),'early_mean_max_abs_reconstruction_error':float(np.abs(early-expected_early).max()),'input_hashes':hashes,'source_hashes':{str(p):file_sha256(p) for p in [Path(__file__),project/'src/length_budget_distill/sae_feature_analysis.py',project/'scripts/13_2_analyze_answer_marker.py']}}
    report=['# Answer-marker diagnostic','',f'分析已完成：{n}条轨迹、{manifest["problems"]}道已观察问题、{f}个既有入选特征。',f'{len(short)} 个短特征中，字面 Answer token 的总激活质量占比范围为 {100*short_range[0]:.2f}%–{100*short_range[1]:.2f}%。',f'检测到答案标题的轨迹为{sum(headings)}条；在前64 token内出现标题的为{manifest["heading_before64_count"]}条。','','这是既有特征与题集的事后区域分析。正文去除后的效应是条件性描述，不能当作新发现或证明格式是干预的因果机制。','统计复用题内配对实现；区间为逐对照95%区间，Holm仅列作描述性校正，不恢复独立确认。','','| Feature | Answer mass | Whole paired d | Body paired d | Without Answer d |','| --- | ---: | ---: | ---: | ---: |']
    report += [f"| F{x['feature_id']} | {100*x['answer_token_mass_share']:.2f}% | {x['original_paired_d']:.3f} | {x['body_paired_d']:.3f} | {x['without_answer_paired_d']:.3f} |" for x in short]
    report += ['','下一步：用新问题做格式反例、同状态目标读回及SAE/dense/format/多随机生成，不能从本分析直接进入冗余推理语义结论。']
    (out/'report_zh.md').write_text('\n'.join(report)+'\n')
    plot_regions(features_out,cfg,out)
    manifest['artifacts']={p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}
    write_json_exclusive(out/'summary.json',manifest)
    write_json_exclusive(out/'COMPLETE.json',{'status':'complete','summary_sha256':file_sha256(out/'summary.json'),'formal_claim_allowed':False})
    return manifest


def plot_regions(rows,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    ordered=sorted(rows,key=lambda r:(r['direction']!='short',-r['answer_token_mass_share']))
    y=np.arange(len(ordered))
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    axes[0].barh(y,[r['answer_token_mass_share'] for r in ordered],color=[cfg['figure_colors'][r['direction']] for r in ordered])
    axes[0].set_yticks(y,[f"{r['direction'][0].upper()} F{r['feature_id']}" for r in ordered]);axes[0].invert_yaxis()
    axes[0].set(xlim=(0,1),xlabel='Activation mass on literal Answer token',title='A. Lexical concentration')
    for direction in ['short','long']:
        selected=[r for r in rows if r['direction']==direction]
        axes[1].scatter([r['original_paired_d'] for r in selected],[r['body_paired_d'] for r in selected],color=cfg['figure_colors'][direction],label=direction+'-associated')
    axes[1].axhline(0,color='#999999',lw=.8);axes[1].axvline(0,color='#999999',lw=.8)
    axes[1].set(xlabel='Whole-trace paired d',ylabel='Reasoning-body paired d',title='B. After excluding heading and suffix');axes[1].legend(frameon=False)
    fig.suptitle('Post-hoc diagnostic of previously selected SAE features')
    for suffix in ['png','pdf']:fig.savefig(out/('answer_marker_diagnostic.'+suffix),dpi=220)
    plt.close(fig)
