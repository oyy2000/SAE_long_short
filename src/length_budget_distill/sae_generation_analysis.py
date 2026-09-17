"""Complete-grid, reviewed-answer analysis of SAE generation controls.

Reuse the generation specifications, cohort audits, reviewed-prediction identity
checks, paired bootstrap, and the established readback plotting conventions.
"""
from collections import Counter, defaultdict
from pathlib import Path
import math
import statistics

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import verify, save, seal
from .baseline_method_analysis import audit_cohort
from .sae_generation_controls import conditions, generated_regions
from .gsm8k_answer_review import load_review_index, apply_reviewed_grade
from .utility_analysis import paired_question_bootstrap, paired_sign_randomization
from .factorial_analysis import holm_adjust

CODE = Path(__file__).resolve().parents[2]


def audit_cell(rows, questions, spec, cap, eos, tolerance):
    sources = {q['problem_id']:q for q in questions}
    audit_cohort(rows, sources)
    for row in rows:
        source = sources[row['problem_id']]
        if row['question'] != source['question'] or row['gold_answer'] != source['answer']:
            raise ValueError('Generation source identity differs')
        if row['spec'] != spec or row['condition'] != spec['name']:
            raise ValueError('Generation condition differs from frozen protocol')
        tokens = row['token_ids']
        if not tokens or len(tokens) > cap or (eos in tokens and (tokens[-1] != eos or tokens.count(eos) != 1)):
            raise ValueError('Invalid generated EOS boundary')
        ended = tokens[-1] == eos
        if row['generated_tokens'] != len(tokens)-int(ended) or row['hit_max_new_tokens'] != (not ended):
            raise ValueError('Inconsistent generated-token or cap accounting')
        if not ended and len(tokens) != cap: raise ValueError('Premature unfinished generation')
        diag = row['diagnostics']; maximum = diag['max_delta_to_hidden_norm_fraction']
        if not math.isfinite(maximum) or not 0 <= maximum <= spec['rho']+tolerance:
            raise ValueError('Invalid measured perturbation')
        if spec['rho'] == 0 and (maximum != 0 or diag['modified_positions'] != 0):
            raise ValueError('Nonzero perturbation in the unmodified control')
        if sum(row['region_token_counts'].values()) != row['retokenized_tokens']:
            raise ValueError('Retokenized region counts do not partition the text')


def audit_batches(rows, batches):
    counts = Counter((r['batch_id'], r['condition']) for r in rows)
    seen = set(); totals = defaultdict(float)
    for batch in batches:
        key = (batch['batch_id'],batch['condition'])
        seconds = batch['generation_wall_seconds']
        if key in seen or counts.get(key) != batch['questions'] or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Missing/duplicate batch or invalid timing')
        seen.add(key); totals[batch['condition']] += seconds
    if seen != set(counts): raise ValueError('Incomplete batch coverage')
    for row in rows:
        batch = next(b for b in batches if b['batch_id'] == row['batch_id'] and b['condition'] == row['condition'])
        if not math.isclose(row['amortized_generation_wall_seconds']*batch['questions'], batch['generation_wall_seconds'], rel_tol=1e-9):
            raise ValueError('Amortized batch timing differs')
    return dict(totals)


def metric(row, name):
    if name in ('reasoning_body','answer_marker','answer_suffix'): return row['region_token_counts'][name]
    return float(row[name])


def analyze_seed_supplement(config_path):
    """Audit fresh equal-count seed outputs with the existing grading/statistics."""
    from transformers import AutoTokenizer
    from .sae_seed_generation import specifications
    cfg=read_json(config_path);root=Path(cfg['generation_root']);out=Path(cfg['result_root'])
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen supplementary analysis source')
    if out.exists():raise FileExistsError(out)
    markers=[Path(cfg['launch_root'])/'FROZEN.json',root/'protocol/FROZEN.json',root/'protocol/SOURCES.json']
    for marker in markers:verify(marker)
    protocol=read_json(root/'protocol/frozen_config.json')
    seed_analysis=Path(protocol['seed_analysis_root']);verify(seed_analysis/'COMPLETE.json')
    markers.append(seed_analysis/'COMPLETE.json')
    specs=specifications(protocol,list(read_jsonl(seed_analysis/'seed_summary.jsonl')))
    questions=list(read_jsonl(root/'inputs/evaluation.jsonl'))
    reviewed=load_review_index(cfg['review_root']);markers.append(Path(cfg['review_root'])/'COMPLETE.json')
    tok=AutoTokenizer.from_pretrained(protocol['teacher']['snapshot_path'],local_files_only=True)
    predictions=[];hardware=[];cost=defaultdict(float)
    for shard in range(protocol['shards']):
        part=root/'generation/seed_stability'/f'shard_{shard:02d}';marker=part/'COMPLETE.json'
        verify(marker);markers.append(marker)
        rows=list(read_jsonl(part/'predictions.jsonl'));expected=questions[shard::protocol['shards']]
        if len(rows)!=len(expected)*len(specs):raise ValueError('Wrong supplementary shard grid size')
        for spec in specs:
            audit_cell([r for r in rows if r['condition']==spec['name']],expected,spec,
                protocol['generation']['max_new_tokens'],tok.eos_token_id,protocol['norm_rounding_tolerance'])
        for name,seconds in audit_batches(rows,list(read_jsonl(part/'batches.jsonl'))).items():cost[name]+=seconds
        for row in rows:
            if row['question_split']!='confirmation' or row['response']!=tok.decode(row['token_ids'],skip_special_tokens=True):
                raise ValueError('Supplement role or token-text identity changed')
            regions=generated_regions(tok,row['response'],protocol['answer_heading_pattern'])
            if any(row[k]!=v for k,v in regions.items()):raise ValueError('Supplement regions changed')
            key=('sae_seed_supplement/'+row['condition'],row['problem_id'])
            if key not in reviewed:raise ValueError('Fresh supplementary output lacks review')
            predictions.append(apply_reviewed_grade(row,reviewed[key],text_field='response',gold_field='gold_answer',tokens_field='generated_tokens'))
        h=read_json(part/'hardware.json')
        if 'H200' not in h['inventory_csv']:raise ValueError('Supplement generation changed GPU type')
        hardware.append({'shard':shard,'hardware':h,'summary':read_json(part/'summary.json')})
    summaries=[];by_condition={s['name']:[r for r in predictions if r['condition']==s['name']] for s in specs}
    for spec in specs:
        rows=by_condition[spec['name']];audit_cohort(rows,[q['problem_id'] for q in questions])
        summaries.append({'condition':spec['name'],'spec':spec,'questions':len(rows),
            'correct':sum(r['is_correct'] for r in rows),'accuracy':statistics.fmean(r['is_correct'] for r in rows),
            'mean_generated_tokens':statistics.fmean(r['generated_tokens'] for r in rows),
            'cap_hit_rate':statistics.fmean(r['hit_max_new_tokens'] for r in rows),
            'generation_batch_seconds':cost[spec['name']],
            **{f'mean_{name}_tokens':statistics.fmean(metric(r,name) for r in rows)
               for name in ('reasoning_body','answer_marker','answer_suffix')}})
    contrasts=[]
    for seed in protocol['seeds']:
        left_name=f'seed_{seed}_top_3'
        for right_name in ('unmodified','seed_17_top_8','answer_format','dense_reference_minus_generated'):
            for field in ('is_correct','generated_tokens','reasoning_body'):
                left={r['problem_id']:metric(r,field) for r in by_condition[left_name]}
                right={r['problem_id']:metric(r,field) for r in by_condition[right_name]}
                if set(left)!=set(right) or len(left)!=protocol['questions']:raise ValueError('Incomplete paired seed support')
                interval=paired_question_bootstrap(left,right,samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
                interval.pop('bootstrap_p_value')
                contrasts.append({'left':left_name,'right':right_name,'metric':field,**interval,
                    'scope':'Descriptive marginal interval conditional on fixed trained dictionaries; no simultaneous inference or noninferiority claim.'})
    out.mkdir(parents=True)
    write_jsonl(out/'reviewed_predictions.jsonl',predictions);write_jsonl(out/'conditions.jsonl',summaries)
    write_jsonl(out/'paired_descriptive_intervals.jsonl',contrasts)
    save(out/'hardware_and_cost.json',{'shards':hardware,'cost_scope':'Batch generation includes reference SAE readback; excludes failed training, preparation and review. Not end-to-end or deployment latency.'})
    save(out/'summary.json',{'conditions':summaries,'predictions':len(predictions),'questions':len(questions),
        'claim_boundary':cfg['claim_boundary'],'original_eight_feature_feasibility_failures_preserved':True})
    plot_seed_supplement(out,summaries)
    report=['# SAE seeds：同特征数探索性生成','',
        '64 道已经观察的问题，四个训练字典各取前三个原规则 confirmed short；同 H200、rho=0.3、每条件一个采样输出。',
        '新增 seeds 不足八个的原方法复验结果保持不变。此补充的参数在看到特征数量后确定，不是原八特征构造的成功复现或独立确认。','',
        '![Equal-count SAE supplement](seed_generation.png)','',
        '| Condition | Correct / 64 | Generated tokens | Body tokens | Cap hit |','| --- | ---: | ---: | ---: | ---: |']
    report += [f"| {r['condition']} | {r['correct']} | {r['mean_generated_tokens']:.2f} | {r['mean_reasoning_body_tokens']:.2f} | {r['cap_hit_rate']:.2%} |" for r in summaries]
    report += ['','逐题配对区间保留于 paired_descriptive_intervals.jsonl，只作固定字典下的边际描述，不以区间跨零证明非劣。',
        '新生成按完全相同文本、题目、答案与 token/cap 身份连接本次复核。项目助手审阅最终答案句不是独立人工验证。',
        '解码中的目标激活读回仍是 seed17 字典参考，不能当作其他 seed 自身的 target engagement。生成计时含该开销。',
        '学生 SFT 与全部构建/失败/训练/推理成本尚未完成；此结果不支持完整蒸馏收益或端到端效率主张。']
    (out/'report_zh.md').write_text('\n'.join(report)+'\n')
    seal(out/'COMPLETE.json',[Path(config_path),*markers,*sorted(out.glob('*'))],
        stage='exploratory_equal_count_seed_generation_analysis',formal_claim_allowed=False)


def plot_seed_supplement(out,rows):
    """Use the existing regional-bar/accuracy layout for this eight-cell grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
    y=np.arange(len(rows));left=np.zeros(len(rows))
    for region,color in [('reasoning_body','#6c757d'),('answer_marker','#2166AC'),('answer_suffix','#e9c46a')]:
        values=np.array([r[f'mean_{region}_tokens'] for r in rows]);axes[0].barh(y,values,left=left,color=color,label=region.replace('_',' '));left+=values
    labels=[r['condition'].replace('dense_reference_minus_generated','dense').replace('_',' ') for r in rows]
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Regional tokens (retokenized)')
    axes[0].legend(fontsize=8,loc='lower right')
    axes[1].plot([100*r['accuracy'] for r in rows],y,'o',color='#2166AC')
    axes[1].set(xlabel='Reviewed teacher accuracy (%)',xlim=(-2,102))
    fig.suptitle('Exploratory equal-count SAE seeds: 64 previously observed questions')
    for suffix in ('png','pdf'):fig.savefig(out/f'seed_generation.{suffix}',dpi=180)
    plt.close(fig)


def confirmation_tests(rows, plan):
    from scipy.stats import binomtest
    by_condition = defaultdict(dict)
    for row in rows:
        if row['family'] != 'dose': continue
        values = by_condition[row['condition']]
        if row['problem_id'] in values: raise ValueError('Duplicate confirmation question-condition')
        values[row['problem_id']] = row
    left_name = f"sae_short_8__rho{plan['reference_rho']:g}"
    left = by_condition[left_name]; results = []
    for right_name in plan['references']:
        right = by_condition[right_name]
        if set(left) != set(right) or len(left) != plan['questions']:
            raise ValueError('Incomplete confirmation contrast support')
        for field in plan['primary_metrics']:
            a = {pid:metric(row,field) for pid,row in left.items()}
            b = {pid:metric(row,field) for pid,row in right.items()}
            if field == 'is_correct':
                wins = sum(a[p] == 1 and b[p] == 0 for p in a)
                losses = sum(a[p] == 0 and b[p] == 1 for p in a)
                stats = {'p_value':float(binomtest(wins,wins+losses,p=.5).pvalue) if wins+losses else 1.,
                    'method':'exact_two_sided_mcnemar', 'left_only_correct':wins, 'right_only_correct':losses}
            else:
                stats = paired_sign_randomization(a,b,samples=plan['randomization_samples'],seed=plan['randomization_seed'])
            results.append({'left':left_name,'right':right_name,'metric':field,'questions':len(a),
                'mean_difference':statistics.fmean(a[p]-b[p] for p in a),**stats})
    adjusted = holm_adjust([row['p_value'] for row in results])
    for row,p in zip(results,adjusted):
        row.update(holm_adjusted_p_value=p,holm_family_size=len(results),alpha=plan['alpha'],
                   rejects_two_sided_null=p<=plan['alpha'])
    return results


def analyze(config_path):
    from transformers import AutoTokenizer
    cfg = read_json(config_path); root = Path(cfg['generation_root']); out = Path(cfg['result_root'])
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen generation-analysis source')
    if out.exists(): raise FileExistsError(out)
    markers = [Path(cfg['launch_root'])/'FROZEN.json', root/'protocol/FROZEN.json', root/'protocol/SOURCES.json']
    for marker in markers: verify(marker)
    protocol = read_json(root/'protocol/frozen_config.json')
    split = protocol.get('evaluation_split','dev'); confirmation = split == 'confirmation'
    if confirmation and cfg.get('confirmation_plan') != protocol['confirmation_plan']:
        raise ValueError('Confirmation analysis differs from its pre-generation plan')
    parent = Path(protocol['readback_root']); markers.append(parent/'directions/COMPLETE.json'); verify(markers[-1])
    review_root = Path(cfg['review_root']); reviewed = load_review_index(review_root); markers.append(review_root/'COMPLETE.json')
    tok = AutoTokenizer.from_pretrained(protocol['teacher']['snapshot_path'], local_files_only=True)
    all_rows, summaries, contrasts, hardware = [], [], [], []
    for family, source_name in (('dose',split),('ablation','ablation')):
        questions = list(read_jsonl(root/'inputs'/f'{source_name}.jsonl')); specs = conditions(protocol, family)
        family_rows = []; cost = defaultdict(float)
        for shard in range(protocol['shards']):
            part = root/'generation'/family/f'shard_{shard:02d}'; marker = part/'COMPLETE.json'
            verify(marker); markers.append(marker)
            rows = list(read_jsonl(part/'predictions.jsonl')); batches = list(read_jsonl(part/'batches.jsonl'))
            expected = [q for i,q in enumerate(questions) if i % protocol['shards'] == shard]
            if len(rows) != len(expected)*len(specs): raise ValueError('Incorrect shard record count')
            for spec in specs:
                audit_cell([r for r in rows if r['condition']==spec['name']], expected, spec,
                           protocol['generation']['max_new_tokens'], tok.eos_token_id, protocol['norm_rounding_tolerance'])
            for name, seconds in audit_batches(rows,batches).items(): cost[name] += seconds
            for row in rows:
                if confirmation and row.get('question_split') != 'confirmation':
                    raise ValueError('Confirmation output has a different question role')
                if row['response'] != tok.decode(row['token_ids'],skip_special_tokens=True):
                    raise ValueError('Stored text differs from actual sampled tokens')
                regions = generated_regions(tok,row['response'],protocol['answer_heading_pattern'])
                if any(row[key] != value for key,value in regions.items()): raise ValueError('Retokenized regions changed')
                key = (f'sae_full_{family}/'+row['condition'],row['problem_id'])
                if key not in reviewed: raise ValueError('New generated output lacks its reviewed identity')
                family_rows.append(apply_reviewed_grade(row,reviewed[key],text_field='response',gold_field='gold_answer',tokens_field='generated_tokens'))
            hardware.append({'family':family,'shard':shard,'hardware':read_json(part/'hardware.json'),
                'peak_gpu_allocated_mib':read_json(part/'summary.json')['peak_gpu_allocated_mib']})
        by_condition = {spec['name']:[r for r in family_rows if r['condition']==spec['name']] for spec in specs}
        for spec in specs:
            rows = by_condition[spec['name']]; audit_cohort(rows,[q['problem_id'] for q in questions]); n=len(rows)
            summaries.append({'family':family,'condition':spec['name'],'spec':spec,'questions':n,
                'correct':sum(r['is_correct'] for r in rows),'accuracy':statistics.fmean(r['is_correct'] for r in rows),
                'mean_generated_tokens':statistics.fmean(r['generated_tokens'] for r in rows),
                'cap_hit_rate':statistics.fmean(r['hit_max_new_tokens'] for r in rows),
                'marker_rate':statistics.fmean(r['has_explicit_answer_marker'] for r in rows),
                'instrumented_generation_batch_seconds':cost[spec['name']],
                'mean_modified_positions':statistics.fmean(r['diagnostics']['modified_positions'] for r in rows),
                'mean_measured_relative_norm':statistics.fmean(r['diagnostics']['mean_delta_to_hidden_norm_fraction'] for r in rows),
                **{f'mean_{region}_tokens':statistics.fmean(metric(r,region) for r in rows)
                   for region in ('reasoning_body','answer_marker','answer_suffix')}})
        references = ['unmodified'] if family == 'dose' else ['unmodified','sae_short_8']
        pairs = [(name,ref) for name in by_condition for ref in references if name != ref]
        if family == 'dose':
            pairs += [(f'sae_short_8__rho{dose:g}',f'{direction}__rho{dose:g}') for dose in protocol['doses']
                      for direction in protocol['dose_directions'] if direction != 'sae_short_8']
        for left,right in dict.fromkeys(pairs):
            for name in cfg['paired_metrics']:
                result = paired_question_bootstrap({r['problem_id']:metric(r,name) for r in by_condition[left]},
                    {r['problem_id']:metric(r,name) for r in by_condition[right]},samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
                contrasts.append({'family':family,'left':left,'right':right,'metric':name,**result})
        all_rows.extend(family_rows)
    out.mkdir(parents=True, exist_ok=False)
    registered_tests = confirmation_tests(all_rows,protocol['confirmation_plan']) if confirmation else []
    if confirmation: write_jsonl(out/'confirmation_tests.jsonl',registered_tests)
    save(out/'summary.json', {'conditions':summaries,'records':len(all_rows),'hardware':hardware,
        'grader_version':next(iter(reviewed.values()))['grader_version'], 'paired_contrasts':contrasts,
        'formal_claim_allowed':False,'confirmation_inspected':confirmation,'student_results':False,
        'registered_confirmation_tests':registered_tests,
        'inference_scope':('The 18 registered two-sided contrasts use one Holm family; accuracy uses exact McNemar and length uses paired-label randomization under label-exchangeability. Bootstrap intervals remain marginal. Other feature/window comparisons are secondary descriptive results. Conditional on one teacher/SAE/sample seed; no noninferiority claim.' if confirmation else
            'Exploratory paired-question intervals, conditional on one teacher/SAE/sample seed. Numerous intervals are not simultaneous tests; finite-bootstrap tail fractions are not exact p-values.'),
        'cost_scope':'Instrumented generation includes SAE readback and online marker detection on all conditions. Batch times summed once. These are not deployment latency or end-to-end costs.',
        'claim_boundary':cfg['claim_boundary']})
    write_jsonl(out/'reviewed_predictions.jsonl',all_rows); write_jsonl(out/'condition_metrics.jsonl',summaries)
    write_jsonl(out/'paired_contrasts.jsonl',contrasts); plot(out,summaries,protocol)
    cohort_description = ('300 道保留 confirmation × 7 方向条件与其中固定 64 题 × 24 消融条件；rho=0.3 来自最初参考设置。旧未干预 baseline 曾统一生成和评分，本次独立性限定为这些问题未用于方向拟合、剂量选择及干预效果分析。' if confirmation else
        '全部 300 道 dev × 25 剂量条件与 64 道 dev × 24 消融条件保留。')
    report = ['# SAE 完整生成：'+('保留题确认实验' if confirmation else '开发集控制实验'),'', '![剂量曲线](generation_dose_curves.png)','',
        cohort_description+'每题每条件一个候选。完整新文本与统一复核评分按题目、文本、gold、token 和 cap 身份关联。', '',
        '| 条件 | 题数 | 正确率 | 生成 tokens | 正文 tokens（重分词） | cap hit |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        report.append(f"| {row['family']}/{row['condition']} | {row['questions']} | {100*row['accuracy']:.2f}% | {row['mean_generated_tokens']:.2f} | {row['mean_reasoning_body_tokens']:.2f} | {100*row['cap_hit_rate']:.2f}% |")
    report += ['', '![特征与窗口](generation_ablations.png)', '',
        '无显式标题的输出保留在总体中。正文/标题/后缀是重新分词的文本区域，不能与采样 token 数直接相加比较；after-marker 未触发的行仍计入该控制。',
        ('预登记 6 对比 × 3 指标的 18 项检验见 confirmation_tests.jsonl，以同一 Holm 族校正；准确率为 exact McNemar，长度为题内标签可交换零假设下的随机化检验。配对 bootstrap 区间是边际区间，窗口/特征消融为次要描述，不以未拒绝零假设宣称非劣。' if confirmation else
         '配对差值及 95% bootstrap 区间见 paired_contrasts.jsonl。开发集多条件比较属于探索性分析，不把单个区间解释为多重比较后成立的发现；未据此选择 confirmation 设置。'),
        '计时包含 SAE 读回及在线标题检测，仅在批次级计一次。模型硬件和峰值显存逐分片保留；这里没有部署时延、完整成本或学生收益结论。', '', cfg['claim_boundary'], '']
    (out/'report_zh.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json',[Path(config_path),*markers,*sorted(out.glob('*'))],stage='reviewed_sae_generation_analysis',formal_claim_allowed=False)


def plot(out, rows, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    colors = ['#2166AC','#b2182b','#e69f00','#6c757d','#009e73','#9467bd']
    dose = [r for r in rows if r['family']=='dose']; base = next(r for r in dose if r['condition']=='unmodified')
    fig,axes = plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for direction,color in zip(protocol['dose_directions'],colors):
        cells = [base]+sorted([r for r in dose if r['spec']['direction']==direction and r['spec']['rho']>0],key=lambda r:r['spec']['rho'])
        label=direction.replace('dense_reference_minus_generated','Dense reference - generated').replace('_',' ')
        axes[0].plot([r['mean_generated_tokens'] for r in cells],[100*r['accuracy'] for r in cells],'.-',color=color,label=label)
        axes[1].plot([r['spec']['rho'] for r in cells],[r['mean_reasoning_body_tokens'] for r in cells],'.-',color=color)
        axes[2].plot([r['spec']['rho'] for r in cells],[100*r['marker_rate'] for r in cells],'.-',color=color)
    axes[0].set(xlabel='Mean generated tokens',ylabel='Reviewed teacher accuracy (%)')
    axes[1].set(xlabel='Relative hidden-norm dose',ylabel='Reasoning-body tokens (retokenized)')
    axes[2].set(xlabel='Relative hidden-norm dose',ylabel='Explicit answer-marker rate (%)',ylim=(0,102))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    split = protocol.get('evaluation_split','dev')
    cohort_label = 'confirmation' if split == 'confirmation' else 'development'
    fig.suptitle(f'Teacher generation: all 300 {cohort_label} questions; single sample per condition')
    for suffix in ('png','pdf'):fig.savefig(out/f'generation_dose_curves.{suffix}',dpi=180)
    plt.close(fig)
    cells=[r for r in rows if r['family']=='ablation'];y=np.arange(len(cells));fig,axes=plt.subplots(1,2,figsize=(12,10),sharey=True,constrained_layout=True)
    left=np.zeros(len(cells))
    for region,color in [('reasoning_body','#6c757d'),('answer_marker','#2166AC'),('answer_suffix','#e9c46a')]:
        values=np.array([r[f'mean_{region}_tokens'] for r in cells]);axes[0].barh(y,values,left=left,color=color,label=region.replace('_',' '));left+=values
    axes[0].set_yticks(y,[r['condition'].replace('sae_','').replace('_',' ') for r in cells],fontsize=8);axes[0].invert_yaxis()
    axes[0].set_xlabel('Mean regional tokens (retokenized)');axes[0].legend(loc='lower right',fontsize=8)
    axes[1].plot([100*r['accuracy'] for r in cells],y,'o',color='#2166AC');axes[1].set(xlabel='Reviewed teacher accuracy (%)',xlim=(0,100))
    fig.suptitle(f'Feature-count and timing controls: fixed 64 {cohort_label} questions')
    for suffix in ('png','pdf'):fig.savefig(out/f'generation_ablations.{suffix}',dpi=180)
    plt.close(fig)
