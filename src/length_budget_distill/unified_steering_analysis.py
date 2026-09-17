"""Select registered development operating points for unified student data."""
from collections import defaultdict
from pathlib import Path
import statistics

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import verify, save, seal
from .unified_math_candidates import audit_candidate_grid, select_candidates, require_same_grading_method
from .utility_analysis import paired_question_bootstrap

CODE = Path(__file__).resolve().parents[2]


def operating_points(metrics, reference, rules):
    """Apply only the recorded engineering filters; never invent an identity arm."""
    result = {}; assessments = []
    for name, row in metrics.items():
        failures = []
        if row['candidate_accuracy'] < reference['candidate_accuracy']-rules['maximum_candidate_accuracy_drop']-1e-12:
            failures.append('candidate_accuracy')
        if row['correct_support_fraction'] < reference['correct_support_fraction']-rules['maximum_correct_support_drop']-1e-12:
            failures.append('correct_support')
        if row['cap_hit_rate'] > rules['maximum_cap_hit_rate']+1e-12: failures.append('cap_hit')
        if not row['paired_selected_questions']: failures.append('no_paired_correct_support')
        assessments.append({**row, 'condition':name, 'admissible':not failures, 'failed_filters':failures})
    for family in ('B3','B4','B7'):
        eligible = [r for r in assessments if r['baseline']==family and r['admissible']]
        if eligible:
            result[family] = min(eligible, key=lambda r:(r['paired_selected_token_difference'],r['strength'],r['condition']))
        else: result[family] = None
    return result, assessments


def analyze(config_path):
    cfg = read_json(config_path); root = Path(cfg['steered_root']); out = Path(cfg['result_root'])
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen steering-analysis source')
    launch_marker = Path(cfg['launch_root'])/'FROZEN.json'; verify(launch_marker)
    if out.exists(): raise FileExistsError(out)
    pcfg = read_json(root/'protocol/frozen_config.json'); reference = Path(pcfg['reference_candidate_root'])
    text_root = Path(cfg['text_compression_root']); markers = [launch_marker]
    for parent, terminal in [(root,'selection/development/COMPLETE.json'),
                             (reference,'selection/development/COMPLETE.json'),
                             (text_root,'merged/development/COMPLETE.json')]:
        for relative in ('protocol/FROZEN.json','protocol/SOURCES.json',terminal):
            marker = parent/relative; verify(marker); markers.append(marker)
    tcfg = read_json(text_root/'protocol/frozen_config.json')
    if Path(tcfg['candidate_root']) != reference:
        raise ValueError('Text compression uses a different raw candidate pool')
    questions = list(read_jsonl(root/'inputs/development.jsonl')); n = len(questions)
    rows = list(read_jsonl(root/'selection/development/predictions.jsonl'))
    raw_reference = list(read_jsonl(reference/'selection/development/predictions.jsonl'))
    rcfg = read_json(reference/'protocol/frozen_config.json')
    require_same_grading_method(pcfg,rcfg,tcfg)
    for key in ('teacher','grading'):
        if tcfg[key] != rcfg[key]: raise ValueError('Text compression and reference differ: '+key)
    for key in ('teacher','grading','generation','candidate_seed_stride','candidates_per_question'):
        if pcfg[key] != rcfg[key]: raise ValueError('Reference and sweep differ: '+key)
    audit_candidate_grid(rows,questions,pcfg); audit_candidate_grid(raw_reference,questions,rcfg)
    selected, _, _ = select_candidates(rows,questions,pcfg)
    original, _, _ = select_candidates(raw_reference,questions,rcfg)
    b1 = original['B1']; by_method = defaultdict(list)
    for row in rows: by_method[row['condition']].append(row)
    b1_rows = [r for r in raw_reference if r['condition']=='B1']
    def base_metrics(values, eligible):
        return {'candidate_accuracy':statistics.mean(float(r['is_correct']) for r in values),
            'mean_generated_tokens':statistics.mean(r['generated_tokens'] for r in values),
            'cap_hit_rate':statistics.mean(float(r['hit_max_new_tokens']) for r in values),
            'correct_support_fraction':len(eligible)/n, 'correct_support_questions':len(eligible),
            'candidate_records':len(values)}
    ref_metrics = base_metrics(b1_rows,b1); metrics = {}; contrasts = {}
    for name, values in by_method.items():
        common = sorted(set(selected[name]) & set(b1))
        metrics[name] = {**base_metrics(values,selected[name]), 'baseline':name.split('__')[0],
            'strength':pcfg['methods'][name]['strength'], 'paired_selected_questions':len(common),
            'paired_selected_token_difference':statistics.mean(selected[name][pid]['generated_tokens']-b1[pid]['generated_tokens']
                for pid in common) if common else None,
            'paired_selected_mean_tokens':statistics.mean(selected[name][pid]['generated_tokens'] for pid in common) if common else None}
        contrasts[name] = {}
        for field in ('is_correct','generated_tokens'):
            def means(records):
                groups = defaultdict(list)
                for row in records: groups[row['problem_id']].append(float(row[field]))
                return {pid:statistics.mean(v) for pid,v in groups.items()}
            contrasts[name][field] = paired_question_bootstrap(means(values),means(b1_rows),
                samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
    points, assessments = operating_points(metrics,ref_metrics,pcfg['operating_point_selection'])
    selected_methods = dict(original)
    for family, row in points.items():
        if row is not None: selected_methods[family] = selected[row['condition']]
    for family, filename in [('B5','B5_selected_full_support.jsonl'),('B6','B6_assigned_examples.jsonl')]:
        values = list(read_jsonl(text_root/'merged/development'/filename))
        by_id = {row['problem_id']:row for row in values}
        if len(by_id)!=len(values): raise ValueError('Repeated text-compression selected question')
        selected_methods[family] = by_id
    ready = all(points.values())
    common = sorted(set.intersection(*(set(v) for v in selected_methods.values()))) if ready else []
    out.mkdir(parents=True)
    write_jsonl(out/'condition_metrics.jsonl',assessments)
    save(out/'paired_candidate_contrasts.json',contrasts)
    save(out/'operating_points.json',{'selected_conditions':{k:v['condition'] if v else None for k,v in points.items()},
        'selected_method_specs':{k:pcfg['methods'][v['condition']] if v else None for k,v in points.items()},
        'all_families_have_admissible_point':ready, 'rules':pcfg['operating_point_selection'],
        'objective_operationalization':'Paired means refer to per-question (selected method tokens minus B1 selected tokens); each question has equal weight.'})
    for method, values in selected_methods.items():
        write_jsonl(out/(method+'_selected_development.jsonl'),[dict(values[pid],selected_baseline=method) for pid in sorted(values)])
    save(out/'summary.json',{'questions':n,'reference':ref_metrics,'operating_points':{k:v['condition'] if v else None for k,v in points.items()},
        'selected_method_support':{k:len(v) for k,v in selected_methods.items()},'all_eight_common_ids':common,
        'all_eight_common_questions':len(common),'all_families_have_admissible_point':ready,
        'student_generation_protocol_frozen':False,'student_training_complete':False,'claim_boundary':cfg['claim_boundary']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for family, color in [('B3','#6c757d'),('B4','#2a9d8f'),('B7','#c41230')]:
        values = sorted([r for r in assessments if r['baseline']==family],key=lambda r:r['strength'])
        for ax, field, label in [(axes[0],'candidate_accuracy','Raw candidate accuracy (%)'),
                                  (axes[1],'correct_support_fraction','Questions with correct supervision (%)')]:
            ax.plot([ref_metrics['mean_generated_tokens']]+[r['mean_generated_tokens'] for r in values],
                    [100*ref_metrics[field]]+[100*r[field] for r in values],color=color,marker='o',label=family)
            ax.set_xlabel('Mean raw generated tokens');ax.set_ylabel(label);ax.set_ylim(0,105)
            ax.legend()
    fig.suptitle('MATH development: 64 questions, four candidates per condition')
    fig.savefig(out/'steering_accuracy_length.png',dpi=180);fig.savefig(out/'steering_accuracy_length.pdf');plt.close(fig)
    report=['# 统一教师 steering 开发比较','', '![曲线](steering_accuracy_length.png)','',
        '每条件四个候选；区间按题重采样，保留题内四候选，不将其当作独立问题。全部开发比较为探索性，未作非劣或独立确认推断。','',
        '| 方法 | 选择条件 | 正确监督题数 |','| --- | --- | ---: |']
    for method,row in points.items(): report.append(f"| {method} | {row['condition'] if row else '无合格点'} | {row['correct_support_questions'] if row else 0} |")
    report += ['',f'全部八方法开发共同支持：{len(common)} / {n}。是否能据此冻结学生池的生成协议仍须单独审核输入与完整方法配置。',
        '运行点目标是同题最短正确监督相对 B1 的平均 token 差；扫描规则及全部失败条件保留。B5/B6 的附回答案与正确过滤不能作为与原始候选同分母的准确率。','',cfg['claim_boundary'],'']
    (out/'report_zh.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json',[Path(config_path),*markers,*sorted(out.glob('*'))],stage='unified_steering_development_analysis',formal_training_ready=False)
