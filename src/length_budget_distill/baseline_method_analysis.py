"""Audited post-hoc summaries of the frozen ASC method checks.

Reuses question-paired statistics and the existing NCSU figure conventions.
Regrading is written separately and never changes registered raw predictions.
"""
from pathlib import Path
import importlib.metadata
import shutil

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import resolve, save, seal, verify
from .gsm8k_grading import grade_gsm8k_response
from .utility_analysis import paired_question_bootstrap
from .factorial_analysis import holm_adjust

CODE=Path(__file__).resolve().parents[2]


def audit_cohort(rows, expected):
    ids=[r['problem_id'] for r in rows]
    if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected):
        raise ValueError('Duplicate, missing, or unexpected evaluation problem')
    return {r['problem_id']:r for r in rows}


def audit_typed_native_prediction(row, source, parent_cfg, tokenizer):
    """Recheck native ASC tokens and the parent's declared mathematical grader."""
    from .factorial import canonical_sha256
    from .math_cap_calibration import cap_view
    from .unified_math_candidates import grade_candidate
    if source['question_role'] != 'development' or source['answer'] != row['gold_answer']:
        raise ValueError('Native mathematical ASC source identity or role differs')
    if row['seed'] != int(canonical_sha256([parent_cfg['seed'],source['problem_id']])[:8],16):
        raise ValueError('Native ASC question random seed differs')
    tokens=row['sampled_token_ids'];cap=parent_cfg['validation']['max_new_tokens'];eos=tokenizer.eos_token_id
    if not tokens or len(tokens)>cap or eos in tokens[:-1] or (tokens[-1]!=eos and len(tokens)!=cap):
        raise ValueError('Invalid native ASC token/EOS boundary')
    view=cap_view(tokens,cap,eos)
    if row['generated_tokens']!=view['generated_tokens'] or row['hit_max_new_tokens']!=view['hit_max_new_tokens']:
        raise ValueError('Native ASC token/cap accounting differs')
    if row['solution']!=tokenizer.decode(view['body_token_ids'],skip_special_tokens=True).strip():
        raise ValueError('Native ASC text does not match its sampled tokens')
    return grade_candidate({'grading_method':parent_cfg['grading']['method'],
                            'grading':parent_cfg['grading']['config']},row['solution'],source)


def analyze_asc(config_path):
    cfg=read_json(config_path);out=resolve(cfg['result_root'])
    grader=grade_gsm8k_response;review_index=None
    from .gsm8k_answer_review import VERSION as REVIEW_VERSION,load_review_index,apply_reviewed_grade
    if cfg.get('grader_version')=='gsm8k_explicit_answer_quantity_v3':
        from .gsm8k_grading_v3 import grade_gsm8k_response as grader
    elif cfg.get('grader_version')==REVIEW_VERSION:
        review_index=load_review_index(resolve(cfg['uniform_review_root']))
    elif cfg.get('grader_version', 'gsm8k_math_verify_v2') not in ('gsm8k_math_verify_v2','typed_math_v2','reviewed_math_v1'):
        raise ValueError('Unknown grader version')
    if out.exists():raise FileExistsError(out)
    groups={};bindings=[];changes=[];fits={};cohorts={};typed_context={}
    if review_index is not None:bindings.append(resolve(cfg['uniform_review_root'])/'COMPLETE.json')
    def regrade(old,group,name):
        if review_index is not None:
            row=apply_reviewed_grade(old,review_index[(group,old['problem_id'])],
                text_field='solution',gold_field='gold_answer',tokens_field='generated_tokens')
        else:
            if cfg.get('grader_version') in ('typed_math_v2','reviewed_math_v1'):
                sources,pcfg,tok=typed_context[name]
                grade=audit_typed_native_prediction(old,sources[old['problem_id']],pcfg,tok)
            else:
                grade=grader(old['solution'],old['gold_answer'],timeout_seconds=cfg['grading_timeout_seconds'])
            row={**old,**grade,'legacy_is_correct':old['is_correct'],
                 'legacy_predicted_answer':old.get('predicted_answer')}
        row['teacher']=name
        if bool(row['is_correct'])!=bool(old['is_correct']):changes.append(row)
        return row
    for name,parent_value in cfg['parents'].items():
        parent=resolve(parent_value)
        for relative in ('protocol/FROZEN.json','protocol/SOURCES.json','asc/COMPLETE.json','asc_generation/COMPLETE.json'):
            marker=parent/relative;verify(marker);bindings.append(marker)
        sources=list(read_jsonl(parent/'inputs/sources.jsonl'))
        expected=[r['problem_id'] for r in sources]
        if len(expected)!=cfg['expected_questions'] or len(set(expected))!=len(expected):
            raise ValueError('Parent development cohort changed')
        cohorts[name]=set(expected)
        parent_cfg=read_json(parent/'protocol/frozen_config.json')
        if cfg.get('grader_version') in ('typed_math_v2','reviewed_math_v1'):
            from transformers import AutoTokenizer
            if parent_cfg['grading']['method']!=cfg['grader_version']:raise ValueError('Parent mathematical grader differs')
            typed_context[name]=({r['problem_id']:r for r in sources},parent_cfg,
                AutoTokenizer.from_pretrained(parent_cfg['teacher']['snapshot_path'],local_files_only=True))
        rows=[]
        for old in read_jsonl(parent/'asc_generation/predictions.jsonl'):
            rows.append(regrade(old,f"asc_initial_{name}/{old['scale']}",name))
        if len(rows)!=len(expected)*len(parent_cfg['validation']['asc_scales']):
            raise ValueError('Unexpected number of ASC predictions')
        scales=list(parent_cfg['validation']['asc_scales'])
        if name in cfg.get('followup_roots',{}):
            follow=resolve(cfg['followup_roots'][name]);verify(follow/'COMPLETE.json');bindings.append(follow/'COMPLETE.json')
            extra=[r for r in read_jsonl(follow/'predictions.jsonl') if r['condition']['kind']=='lower_scale']
            added=sorted({r['condition']['scale'] for r in extra})
            if set(added)&set(scales):raise ValueError('Duplicated initial/follow-up scales')
            if set(added)!=set(cfg['followup_scales']):raise ValueError('Unexpected follow-up scale support')
            for old in extra:
                row=regrade(old,f"asc_followup_{name}/{old['condition']['name']}",name)
                row['scale']=old['condition']['scale'];rows.append(row)
            scales+=added
        groups[name]={str(scale):audit_cohort([r for r in rows if r['scale']==scale],expected)
                      for scale in sorted(scales)}
        fits[name]=read_json(parent/'asc/summary.json')
    if any(value!=next(iter(cohorts.values())) for value in cohorts.values()):
        raise ValueError('Teacher method checks have different problem cohorts')
    summaries={};contrasts={}
    for name,cells in groups.items():
        summaries[name]={};contrasts[name]={}
        for scale,by_id in cells.items():
            rows=list(by_id.values());n=len(rows)
            summaries[name][scale]={'n':n,'correct':sum(r['is_correct'] for r in rows),
                'accuracy':sum(r['is_correct'] for r in rows)/n,
                'legacy_accuracy':sum(r['legacy_is_correct'] for r in rows)/n,
                'mean_generated_tokens':sum(r['generated_tokens'] for r in rows)/n,
                'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rows)/n,
                'mean_latency_seconds':sum(r['latency_seconds'] for r in rows)/n}
            if float(scale)==0:continue
            contrasts[name][scale]={}
            for field in ('is_correct','generated_tokens'):
                contrasts[name][scale][field]=paired_question_bootstrap(
                    {pid:float(row[field]) for pid,row in by_id.items()},
                    {pid:float(row[field]) for pid,row in cells['0.0'].items()},
                    samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
    all_contrasts=[r for cells in contrasts.values() for fields in cells.values() for r in fields.values()]
    for row,p in zip(all_contrasts,holm_adjust([r['bootstrap_p_value'] for r in all_contrasts])):
        row['holm_p_value']=float(p)
    prompts={};prompt_contrasts={}
    for probe in cfg.get('prompt_sources',[]):
        source=resolve(probe['root']);verify(source/'COMPLETE.json');bindings.append(source/'COMPLETE.json')
        selected=[r for r in read_jsonl(source/'predictions.jsonl') if r['condition']['kind']=='prompt_probe']
        for old in selected:
            label=probe.get('labels',{}).get(old['condition']['name'],old['condition']['name'])
            group=probe['review_prefix']+'/'+label
            row=regrade(old,group,'r1');prompts.setdefault(label,[]).append(row)
    if prompts:
        ids=[r['problem_id'] for r in next(iter(prompts.values()))]
        if len(ids)!=cfg['prompt_questions']:raise ValueError('Prompt cohort size changed')
        for values in prompts.values():audit_cohort(values,ids)
        prompts['project_prompt']=[groups['r1']['0.0'][pid] for pid in ids]
        for label,values in prompts.items():
            if label=='project_prompt':continue
            prompt_contrasts[label]={field:paired_question_bootstrap(
                {r['problem_id']:float(r[field]) for r in values},
                {r['problem_id']:float(r[field]) for r in prompts['project_prompt']},
                samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
                for field in ('is_correct','generated_tokens')}
        values=[r for fields in prompt_contrasts.values() for r in fields.values()]
        for row,p in zip(values,holm_adjust([r['bootstrap_p_value'] for r in values])):row['holm_p_value']=float(p)
    prompt_summaries={label:{'n':len(values),'correct':sum(r['is_correct'] for r in values),
        'accuracy':sum(r['is_correct'] for r in values)/len(values),
        'mean_generated_tokens':sum(r['generated_tokens'] for r in values)/len(values),
        'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in values)/len(values)} for label,values in prompts.items()}
    out.mkdir(parents=True)
    for name,cells in groups.items():
        write_jsonl(out/(name+'_regraded.jsonl'),[r for by_id in cells.values() for r in by_id.values()])
    write_jsonl(out/'grading_changes.jsonl',changes)
    if prompts:write_jsonl(out/'prompt_regraded.jsonl',[dict(r,prompt_label=label) for label,values in prompts.items() for r in values])
    summary={'status':'complete','summaries':summaries,'paired_contrasts':contrasts,
             'fit_diagnostics':fits,'changed_grades':len(changes),'formal_claim_allowed':False,
             'grader_version':cfg.get('grader_version', 'gsm8k_math_verify_v2'),
             'claim_boundary':cfg['claim_boundary'],'post_hoc_regrading':True,
             'prompt_summaries':prompt_summaries,'prompt_contrasts':prompt_contrasts,
             'uniform_review_root':cfg.get('uniform_review_root'),
             'contrast_families':{'scale_accuracy_and_length':len(all_contrasts),'prompt_accuracy_and_length':len(prompt_contrasts)*2},
             'bootstrap_unit':'paired_problem; no training/decoding-seed variability estimate'}
    save(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,len(summaries),figsize=(11,4),squeeze=False,constrained_layout=True)
    for ax,(name,cells) in zip(axes[0],summaries.items()):
        ordered=sorted(cells,key=float)
        ax.plot([cells[s]['mean_generated_tokens'] for s in ordered],
                [100*cells[s]['accuracy'] for s in ordered],color='#2a9d8f',marker='o')
        for s in ordered:
            ax.annotate('scale '+s,(cells[s]['mean_generated_tokens'],100*cells[s]['accuracy']),
                        xytext=(4,5),textcoords='offset points',fontsize=9)
        ax.set_title(cfg['labels'][name]);ax.set_xlabel('Mean generated tokens')
        ax.set_ylabel(cfg.get('dataset_label','GSM8K')+' accuracy (%)');ax.set_ylim(0,105)
    fig.suptitle(f"ASC-CES: {cfg['expected_questions']} previously observed development questions")
    fig.savefig(out/'asc_accuracy_length.png',dpi=180)
    fig.savefig(out/'asc_accuracy_length.pdf');plt.close(fig)
    if cfg.get('followup_roots'):
        fig,axes=plt.subplots(1,len(summaries),figsize=(11,4),squeeze=False,constrained_layout=True)
        for ax,(name,cells) in zip(axes[0],summaries.items()):
            ordered=[s for s in sorted(cells,key=float) if float(s)<=0.5]
            ax.plot([cells[s]['mean_generated_tokens'] for s in ordered],
                    [100*cells[s]['accuracy'] for s in ordered],color='#2a9d8f',marker='o')
            for s in ordered:ax.annotate(s,(cells[s]['mean_generated_tokens'],100*cells[s]['accuracy']),
                xytext=(4,5),textcoords='offset points',fontsize=9)
            ax.set_title(cfg['labels'][name]);ax.set_xlabel('Mean generated tokens');ax.set_ylabel(cfg.get('dataset_label','GSM8K')+' accuracy (%)')
            ax.set_ylim(80,102)
        fig.suptitle('ASC-CES: lower-scale development diagnostic (zoomed accuracy axis)')
        fig.savefig(out/'asc_lower_scale_accuracy_length.png',dpi=180)
        fig.savefig(out/'asc_lower_scale_accuracy_length.pdf');plt.close(fig)
    report=['# ASC-CES 方法检查结果','','![准确率与长度](asc_accuracy_length.png)','',
        f"同一版本 {cfg.get('dataset_label','GSM8K')} grader 对本配置的 {len(summaries)} 个教师设置的完整原始输出重新评分；原始记录及旧分数保留。",
        '若使用统一复核，疑难项由项目助手审查隐藏方法与参考答案的队列；缺失最终答案的截断记录按格式规则判分，不属于独立人工验证。',
        f'共 {len(changes)} 条评分发生变化。开发集为已观察的 64 题，本报告属于事后方法诊断。','',
        '| 教师 | 强度 | 正确数 / 64 | 平均 tokens | Cap hit |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,cells in summaries.items():
        for s,v in cells.items():report.append(f"| {cfg['labels'][name]} | {s} | {v['correct']} / {v['n']} | {v['mean_generated_tokens']:.2f} | {100*v['cap_hit_rate']:.2f}% |")
    if cfg.get('followup_roots'):report+=['','![低强度诊断](asc_lower_scale_accuracy_length.png)','']
    if prompts:
        report+=['','同一固定 16 题的 R1 提示检查（均为未干预生成）：','',
            '| 提示 | 正确数 | 平均 tokens | Cap hit |','| --- | ---: | ---: | ---: |']
        for label,v in prompt_summaries.items():report.append(f"| {label} | {v['correct']} / {v['n']} | {v['mean_generated_tokens']:.2f} | {100*v['cap_hit_rate']:.2f}% |")
        report+=['','raw_question_without_bos 是早期 add_special_tokens=False 的诊断；legacy_raw_question_with_bos 才保留上游 raw tokenizer 默认 BOS。二者分别保留。native_boxed_instruction 使用原生 chat 加逐步推理和 boxed 答案要求。此检查不能证明正式论文提示已被完全复现。','']
    report.extend(['','以下模型设置的向量拟合已完成。通用文本 KL 的独立检查结果：',''])
    for name,v in fits.items():
        report.append(f"- {cfg['labels'][name]}：KL={v['heldout_generic_kl_mean']:.6f}，向量范数={v['final']['vector_norm']:.3f}，拟合 {v['final']['elapsed_seconds']:.1f} 秒。")
    report.extend(['','目标 KL 为 0.02；有限 hinge 惩罚不保证最终满足该预算。WikiText 的低 KL 也不等于数学生成行为受到同等约束。',
        '配对问题 bootstrap 结果保存于 summary.json；强度与提示分别作为两个 Holm 校正族，均包含准确率和长度比较。它不估计训练或解码随机种子的方差，事后开发比较也不构成独立确认。','',cfg['claim_boundary'],''])
    (out/'report_zh.md').write_text('\n'.join(report))
    provenance=out/'provenance';provenance.mkdir()
    shutil.copy2(config_path,provenance/'analysis_config.json')
    # Snapshot dependencies so future source maintenance cannot invalidate this report.
    shutil.copytree(CODE/'src/length_budget_distill',provenance/'length_budget_distill',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(CODE/'scripts/13_9_analyze_baseline_methods.py',provenance)
    save(provenance/'versions.json',{n:importlib.metadata.version(n) for n in
        ('math-verify','latex2sympy2_extended','sympy','matplotlib')})
    seal(out/'COMPLETE.json',bindings+[p for p in out.rglob('*') if p.is_file()],
         formal_claim_allowed=False,stage='asc_method_check_post_hoc_analysis')
    return summary
