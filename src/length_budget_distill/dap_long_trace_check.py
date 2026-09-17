"""Rewrite verified long released traces with the published DAP framework."""
from pathlib import Path
import json
import logging
import os
import subprocess
from collections import Counter
import shutil

from .experiment_io import read_json
from .records import read_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save,seal,verify,admission,teacher_bundle
from .baseline_method_analysis import audit_cohort
from .baseline_reproduction import generate_text,summarize_generations
from .compression_baselines import dap_messages,dap_structure
from .typed_math_grading import grade_typed_response
from .utility_analysis import paired_question_bootstrap
from .records import write_jsonl

CODE=Path(__file__).resolve().parents[2]


def run(config_path,*,smoke):
    import torch
    cfg=read_json(config_path);root=Path(cfg['result_root']);parent=Path(cfg['paired_source_root']);out=root/('smoke' if smoke else 'main')
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen DAP check snapshot')
    verify(Path(cfg['launch_root'])/'FROZEN.json');verify(parent/'COMPLETE.json')
    if not read_json(parent/'summary.json')['ready_for_rewriting']:raise ValueError('Paired source cohort is incomplete')
    if not smoke:verify(root/'smoke/COMPLETE.json')
    sources=list(read_jsonl(parent/'selected_sources.jsonl'))
    if len(sources)!=cfg['expected_questions']:raise ValueError('DAP source cohort changed')
    if smoke:sources=sources[:cfg['smoke_questions']]
    prompt=read_json(cfg['prompt_config']);grading=read_json(cfg['grading_config'])
    out.mkdir(parents=True,exist_ok=False);admission(cfg)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv'],check=True,text=True,capture_output=True)
    save(out/'hardware.json',{'job_id':os.environ['SLURM_JOB_ID'],'csv':hardware.stdout,
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))})
    model,tok=teacher_bundle(cfg);torch.cuda.reset_peak_memory_stats();rows=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for source in sources:
            seed=int(canonical_sha256([cfg['seed'],source['problem_id']])[:8],16)
            result=generate_text(model,tok,dap_messages(source['question'],source['long_response'],prompt),cfg['generation'],seed)
            grade=grade_typed_response(result['solution'],source['math_record'],grading)
            row={'problem_id':source['problem_id'],'question':source['math_record']['question'],'gold_answer':source['math_record']['answer'],
                **result,**grade,**dap_structure(result['solution']),
                'long_source_tokens':source['long_tokens'],'author_short_tokens':source['author_short_tokens'],
                'author_short_is_correct':source['author_short_grade']['is_correct'],
                'source_row_ids':{k:source[k] for k in ('long_file','long_row_index','short_file','short_row_index')},
                'full_thought_and_solution_retained':True}
            rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
            logging.info('DAP long-source %s input=%d output=%d correct=%s difficulty=%s',row['problem_id'],result['prompt_tokens'],result['generated_tokens'],row['is_correct'],row['difficulty'])
    audit_cohort(rows,[r['problem_id'] for r in sources])
    summary={**summarize_generations(rows),'smoke_only':smoke,
        'mean_long_source_tokens':sum(r['long_source_tokens'] for r in rows)/len(rows),
        'mean_author_short_tokens':sum(r['author_short_tokens'] for r in rows)/len(rows),
        'author_short_accuracy':sum(r['author_short_is_correct'] for r in rows)/len(rows),
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'original_teacher_generation_cost':'Unavailable from released data; excluded from measured rewrite-stage cost.',
        'claim_boundary':cfg['claim_boundary'],'formal_claim_allowed':False}
    save(out/'summary.json',summary)
    seal(out/'COMPLETE.json',[parent/'COMPLETE.json',parent/'selected_sources.jsonl',Path(config_path),
        Path(cfg['launch_root'])/'FROZEN.json']+[p for p in out.rglob('*') if p.is_file()],
        stage='dap_long_trace_rewrite_method_check',smoke_only=smoke,formal_claim_allowed=False)


def analyze(config_path):
    """Compare complete released long/short pairs and actual DAP rewrites."""
    cfg=read_json(config_path);parent=Path(cfg['rewrite_root']);source=Path(cfg['paired_source_root']);out=Path(cfg['result_root'])
    verify(parent/'COMPLETE.json');verify(source/'COMPLETE.json')
    rows=list(read_jsonl(parent/'predictions.jsonl'))
    sources=list(read_jsonl(source/'selected_sources.jsonl'))
    expected=[r['problem_id'] for r in sources]
    if len(expected)!=cfg['expected_questions']:raise ValueError('DAP analysis cohort changed')
    source_by_id=audit_cohort(sources,expected);audit_cohort(rows,expected)
    for row in rows:
        original=source_by_id[row['problem_id']]
        if not original['eligible_long_input'] or not original['long_grade']['is_correct']:
            raise ValueError('DAP original source failed the registered eligibility rule')
        if row['author_short_is_correct']!=original['author_short_grade']['is_correct']:
            raise ValueError('Released short grade changed')
        if row['question']!=original['math_record']['question'] or str(row['gold_answer'])!=str(original['math_record']['answer']):
            raise ValueError('DAP rewrite/reference identity changed')
        if row['long_source_tokens']!=original['long_tokens'] or row['author_short_tokens']!=original['author_short_tokens']:
            raise ValueError('DAP source token accounting changed')
        if not row['full_thought_and_solution_retained']:raise ValueError('DAP output was clipped')
    metrics={}
    definitions={'released_long':('long_source_tokens',None),
                 'released_dap_short':('author_short_tokens','author_short_is_correct'),
                 'qwen7b_dap_rewrite':('generated_tokens','is_correct')}
    for name,(tokens,correct) in definitions.items():
        metrics[name]={'n':len(rows),'mean_tokens':sum(r[tokens] for r in rows)/len(rows),
            'correct':sum(r[correct] if correct else True for r in rows),
            'accuracy':sum(r[correct] if correct else True for r in rows)/len(rows)}
    contrasts={}
    for name,field in [('released_dap_short','author_short_tokens'),('qwen7b_dap_rewrite','generated_tokens')]:
        contrasts[name]=paired_question_bootstrap({r['problem_id']:r[field] for r in rows},
            {r['problem_id']:r['long_source_tokens'] for r in rows},
            samples=cfg['bootstrap_samples'],seed=cfg['bootstrap_seed'])
        metrics[name]['reduction_of_mean_tokens']=1-metrics[name]['mean_tokens']/metrics['released_long']['mean_tokens']
    summary={'status':'complete','formal_claim_allowed':False,'metrics':metrics,
        'paired_token_differences_from_long':contrasts,
        'grading_status_counts':dict(Counter(r['status'] for r in rows)),
        'difficulty_counts':dict(Counter(r['difficulty'] for r in rows)),
        'legacy_structure_flag_counts':{k:sum(bool(r[k]) for r in rows) for k in ('has_analysis','has_reflection','has_decomposition','has_solution')},
        'cap_hit_count':sum(r['hit_max_new_tokens'] for r in rows),
        'rewrite_seconds':sum(r['latency_seconds'] for r in rows),
        'input_tokens':sum(r['prompt_tokens'] for r in rows),
        'output_tokens':sum(r['generated_tokens'] for r in rows),
        'peak_gpu_allocated_mib':read_json(parent/'summary.json')['peak_gpu_allocated_mib'],
        'all_original_sources_correct_by_selection':True,'source_long_token_filter':cfg['source_long_token_filter'],
        'original_teacher_generation_cost':'Not available from released data.',
        'source_math_ids_reserved_from_sft':str(source/'reserved_math_development_ids.jsonl'),
        'claim_boundary':cfg['claim_boundary']}
    out.mkdir(parents=True,exist_ok=False);save(out/'summary.json',summary)
    write_jsonl(out/'paired_predictions.jsonl',rows)
    write_jsonl(out/'incorrect_rewrites.jsonl',[r for r in rows if not r['is_correct']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name,field,color,label in [('released_dap_short','author_short_tokens','#6c757d','Released DAP short'),
                                  ('qwen7b_dap_rewrite','generated_tokens','#2a9d8f','Qwen-7B DAP rewrite')]:
        axes[0].scatter([r['long_source_tokens'] for r in rows],[r[field] for r in rows],
            color=color,alpha=.65,s=22,label=label)
    axes[0].plot([100,20000],[100,20000],color='#b5b5b5',linestyle='--',linewidth=1,label='Equal length')
    axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xlim(900,18000);axes[0].set_ylim(100,18000)
    axes[0].set_xlabel('Released long trace tokens');axes[0].set_ylabel('Full compressed output tokens');axes[0].legend(fontsize=8)
    labels=['Released\nlong','Released\nDAP short','Qwen-7B\nDAP rewrite'];values=list(metrics.values())
    axes[1].bar(labels,[r['mean_tokens'] for r in values],color=['#e9c46a','#6c757d','#2a9d8f'])
    for i,row in enumerate(values):axes[1].text(i,row['mean_tokens']+70,f"{row['mean_tokens']:.0f} tokens\n{row['correct']}/{row['n']} correct",ha='center',fontsize=9)
    axes[1].set_ylim(0,metrics['released_long']['mean_tokens']*1.22);axes[1].set_ylabel('Mean tokens')
    fig.suptitle('DAP: 64 reserved MATH development questions; correct long sources')
    fig.savefig(out/'dap_long_trace_comparison.png',dpi=180);fig.savefig(out/'dap_long_trace_comparison.pdf');plt.close(fig)
    report=['# DAP 长轨迹改写检查','','![完整输出长度对比](dap_long_trace_comparison.png)','',
        '固定 64 道 MATH 开发题；原长轨迹通过独立答案验证，长度在 1,000–16,384 Qwen-7B tokens 范围。原始长轨迹的 100% 正确率由筛选条件保证，不能作为全题池教师能力。',
        '保留全部 Thought 和 Solution 输出，包括不规范段落、错误和失败；没有只截取较短 Solution。作者发布的压缩轨迹作为同题参考单独列出。','',
        '| 数据 | 正确数 / 64 | 平均 tokens | 相对原长轨迹缩短 |','| --- | ---: | ---: | ---: |']
    for name,row in metrics.items():report.append(f"| {name} | {row['correct']} / {row['n']} | {row['mean_tokens']:.2f} | {100*row.get('reduction_of_mean_tokens',0):.2f}% |")
    report+=['',f"改写共 {summary['input_tokens']:,} input tokens、{summary['output_tokens']:,} output tokens，生成计时合计 {summary['rewrite_seconds']:.2f} 秒；cap hit={summary['cap_hit_count']}，评分状态={summary['grading_status_counts']}。",
        '这里只测改写阶段；公开数据没有完整原教师生成成本，不能据此宣称端到端成本占优。',
        '旧段落检测器仅匹配带冒号的标题，has_solution=False 不必然意味着没有答案段；例如 begin_of_solution 标记、Markdown 标题会漏检。原始检测标记保留，不用它筛掉样本。',
        '本次改写模型为 Qwen2.5-7B-Instruct，原论文使用完整 DeepSeek-R1；属于忠实保留 DAP 难度框架的模型适配检查。作者原始 25K 样本 ID 未恢复。',
        '所用 64 个 MATH ID 已登记为开发保留题，后续学生 SFT 必须排除。','',cfg['claim_boundary'],'']
    (out/'report_zh.md').write_text('\n'.join(report))
    provenance=out/'provenance';provenance.mkdir()
    shutil.copy2(config_path,provenance/'analysis_config.json')
    shutil.copytree(CODE/'src/length_budget_distill',provenance/'length_budget_distill',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(CODE/'scripts/13_21_analyze_dap_long_traces.py',provenance)
    seal(out/'COMPLETE.json',[parent/'COMPLETE.json',source/'COMPLETE.json']+[p for p in out.rglob('*') if p.is_file()],
        stage='dap_long_trace_method_analysis',formal_claim_allowed=False,questions=len(rows))
