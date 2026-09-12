"""Finalize the exploratory experiment only after audited teacher decisions."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .experiment_io import read_json, write_json_exclusive
from .factorial import file_sha256
from .sae_local_data import ROOT, paths, evidence, verify
from .sae_local_report import audited_records, plots


def learning_curves(config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,_=paths(config)
    output=ROOT/config['figure_root']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    for condition,color in zip(config['sampling']['conditions'],['#777777','#2166AC','#B2182B']):
        metrics=read_json(root/'sae_training'/condition/'training_metrics.json')
        rows=metrics['training_log'];x=[r['step'] for r in rows]
        label={'full_token_uniform':'Token-uniform','full_trace_balanced':'Trace-balanced','prefix64_trace_balanced':'Prefix64-balanced'}[condition]
        axes[0].plot(x,[r['train_reconstruction_loss'] for r in rows],marker='o',label=label,color=color)
        axes[1].plot(x,[r['dev_explained_variance'] for r in rows],marker='o',label=label,color=color)
    axes[0].set(title='Reconstruction loss on training batches',ylabel='MSE',xlabel='Optimizer step',yscale='log')
    axes[1].set(title='Same full-trace dev sample',ylabel='Explained variance',xlabel='Optimizer step')
    axes[1].legend(frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same architecture, seed, optimizer and step budget; different training positions')
    fig.tight_layout()
    files=[]
    for suffix in ['png','pdf']:
        p=output/f'03_sae_training_curves.{suffix}';fig.savefig(p,dpi=240,bbox_inches='tight');files.append(p)
    plt.close(fig)
    return files


def outcome_plots(config, student):
    import numpy as np
    import matplotlib.pyplot as plt
    root,_=paths(config);output=ROOT/config['figure_root'];files=[]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ['png','pdf']:
            p=output/f'{name}.{suffix}';fig.savefig(p,dpi=240,bbox_inches='tight');files.append(p)
        plt.close(fig)
    replay=read_json(root/'matched_joint_control/analysis.json')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
    x=np.arange(2)
    for i,(name,label,color) in enumerate([
        ('no_steering_replay','No steering','#777777'),
        ('selected_target_replay','Selected joint intervention','#B2182B'),
        ('matched_joint_random','Random 16 positive + 16 negative','#2166AC')]):
        cells=[replay['summaries'][s]['metrics'][name] for s in ['dev','test']]
        ys=[c['mean_tokens'] for c in cells]
        bars=axes[0].bar(x+(i-1)*.25,ys,.24,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=8)
        axes[1].plot(x,[100*c['accuracy'] for c in cells],marker=['o','s','^'][i],
                     markersize=8,label=label,color=color,alpha=.8)
    axes[0].set(xticks=x,xticklabels=['Dev','Test'],ylabel='Mean output tokens',ylim=(0,325))
    axes[1].set(xticks=x,xticklabels=['Dev','Test'],ylabel='Correct (%)',ylim=(94,100))
    axes[1].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Supplement: fixed dev-selected target versus structurally matched random control')
    save(fig,'04_structural_random_control')
    if student is None:return files
    labels=['No steering','SAE joint','Matched random','Historical short']
    conditions=config['student_followup']['conditions'];colors=['#777777','#B2182B','#2166AC','#4393C3']
    metrics={r['name']:r for r in student['metrics']}
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6))
    for j,regime in enumerate(config['student_followup']['budget_regimes']):
        values=[100*metrics[f'{regime}__{c}__seed_17']['accuracy'] for c in conditions]
        bars=axes[0].bar(np.arange(4)+(j-.5)*.35,values,.34,color=colors,alpha=[1,.5][j],
                         label=['Equal examples','Equal target tokens'][j],edgecolor='white')
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8)
        rows=[r for r in student['contrasts'] if r['target'].startswith(regime+'__')]
        means=np.array([100*r['estimate'] for r in rows]);lo=np.array([100*r['ci_low'] for r in rows]);hi=np.array([100*r['ci_high'] for r in rows])
        axes[1].errorbar(means,np.arange(3)+(j-.5)*.18,xerr=[means-lo,hi-means],fmt=['o','s'][j],
                         color=['#B2182B','#2166AC'][j],capsize=3,label=['Equal examples','Equal target tokens'][j])
    axes[0].axhline(100*student['reused_base']['accuracy'],color='black',ls='--',lw=1,label='Base (verified E1 predictions)')
    all_acc=[100*r['accuracy'] for r in student['metrics']]+[100*student['reused_base']['accuracy']]
    axes[0].set(xticks=np.arange(4),xticklabels=labels,ylabel='GSM8K accuracy (%)',ylim=(min(all_acc)-5,max(all_acc)+4))
    axes[0].tick_params(axis='x',rotation=12);axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    axes[1].axvline(0,color='black',lw=.8)
    axes[1].set(yticks=np.arange(3),yticklabels=['SAE − no steering','SAE − matched random','SAE − historical short'],
                xlabel='Paired accuracy difference (percentage points)',title='95% question-bootstrap intervals')
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Eight student runs: same question support, seed 17, full 1,269-question evaluation')
    save(fig,'05_student_accuracy')
    data=read_json(root/'student_followup/DATA_COMPLETE.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for j,regime in enumerate(config['student_followup']['budget_regimes']):
        cells=[next(r for r in data['budget'] if r['regime']==regime and r['condition']==c) for c in conditions]
        for ax,key,scale in [(axes[0],'completion_tokens',1000),(axes[1],'steps',1)]:
            bars=ax.bar(np.arange(4)+(j-.5)*.35,[r[key]/scale for r in cells],.34,
                        color=colors,alpha=[1,.5][j],label=['Equal examples','Equal target tokens'][j])
            ax.bar_label(bars,fmt='%.1f' if scale==1000 else '%d',fontsize=8)
            ax.set(xticks=np.arange(4),xticklabels=labels);ax.tick_params(axis='x',rotation=12)
    axes[0].set(ylabel='Completion tokens (thousands)',title='Whole-trace supervision budget')
    axes[1].set(ylabel='Optimizer steps',title='Equal token counts do not equalize steps')
    axes[1].legend(frameon=False,fontsize=8)
    save(fig,'06_student_training_budgets')
    return files


def teacher_results(config):
    root,_=paths(config)
    decision=read_json(root/'TEACHER_DECISION.json')
    test=read_json(root/'analysis/test/teacher_summary.json')
    metrics={r['condition']:r for r in test['summaries']};base=metrics['no_steering']
    selected=metrics[decision['selected_condition']]
    replay=read_json(root/'matched_joint_control/analysis.json')['summaries']
    def get(mode,count,rho,start=0,dictionary='full_trace_balanced'):
        return metrics[f'{dictionary}__{mode}{count}__rho{rho:g}__start{start}']
    section=['## 已完成的教师实验','',
        f"完成 3 个 SAE、原定 27 条件 × 64 dev 题与 64 test 题（3,456 条输出），以及结构匹配随机对照的 384 条配对重放。dev 固定选择 `{decision['selected_condition']}`：使用 full-trace-balanced 字典，增强 16 个 short 方向并减去 16 个 long 方向，从第一个输出 logit 开始注入，实际扰动范数为隐藏状态的 30%。",'',
        f"原定 test 中，平均长度从 **{base['mean_output_tokens']:.2f} 降至 {selected['mean_output_tokens']:.2f} tokens（缩短 {selected['relative_length_reduction']:.2%}）**；准确率从 {base['correct']}/64 变为 {selected['correct']}/64。配对长度差为 {selected['length_minus_base']['estimate']:.2f} tokens，95% 区间 [{selected['length_minus_base']['ci_low']:.2f}, {selected['length_minus_base']['ci_high']:.2f}]。准确率差的区间为 [{100*selected['accuracy_minus_base']['ci_low']:.2f}, {100*selected['accuracy_minus_base']['ci_high']:.2f}] 个百分点，**不能据此证明准确率无损或非劣效**。",'',
        '![测试题上的实际范数剂量响应](../figures/phase6_local_length_controlled_strength_v1/02_teacher_dose_response.png)','',
        '图 2：固定字典/方向下的剂量响应、正确率、实际范数及完整 TopK 目标激活读回。区间是点态探索性区间；图中的 random16 对 joint16 仅匹配总范数，没有匹配其总计 32 个方向的正负结构，故另做下面的补充对照。','',
        '| 原定 test 对照 | 答对/64 | 平均 tokens | 相对无干预长度差 [95% CI] |',
        '|---|---:|---:|---:|']
    cells=[('无干预',base),('Full short4 / 1.3%',get('short',4,.013)),('Full short4 / 15%',get('short',4,.15)),
           ('Full short4 / 30%',get('short',4,.30)),('Full short16 / 15%',get('short',16,.15)),
           ('Full short16 / 30%',get('short',16,.30)),('Full joint16 / 15%',get('joint',16,.15)),
           ('Full joint16 / 30%（dev 选定）',selected),('Full long16 / 30%',get('long',16,.30)),
           ('Full random16 / 30%',get('random',16,.30)),
           ('Prefix64 short16 / 15%',get('short',16,.15,dictionary='prefix64_trace_balanced')),
           ('Prefix64 short16 / 30%',get('short',16,.30,dictionary='prefix64_trace_balanced')),
           ('Full short4 / 15% / start64',get('short',4,.15,64))]
    for name,s in cells:
        d=s['length_minus_base'];section.append(f"| {name} | {s['correct']} | {s['mean_output_tokens']:.2f} | {d['estimate']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] |")
    diag=selected['diagnostics']
    r=replay['test'];m=r['metrics'];delta=r['length_minus_structural_random']
    section += ['',
        f"**加大力度并非越大越好。** 同样 30% 范数，short16 单独增强只答对 {get('short',16,.30)['correct']}/64，联合方向答对 {selected['correct']}/64；prefix64 short16 则只有 {get('short',16,.30,dictionary='prefix64_trace_balanced')['correct']}/64。因此本轮支持的是一个经 dev 选定的具体联合方向，不能推广成‘增强所有 short feature 都有效’。token-uniform SAE 只做训练/重构消融，没有与其字典进行完整干预对照，不能归因说 trace-balanced 训练本身造成了全部干预改善。",'',
        f"**干预确实送达。** 选定条件实际平均范数为 {diag['mean_delta_to_hidden_norm_fraction']:.4%}；目标 short 激活总和平均从 {diag['mean_short_activation_before']:.3f} 变为 {diag['mean_short_activation_after']:.3f}，long 从 {diag['mean_long_activation_before']:.3f} 变为 {diag['mean_long_activation_after']:.3f}。这些是干预前后完整 TopK 编码的读回，包含稀疏支持变化；不说明其他 feature 没有变化，也不为目标赋予单一语义。",'',
        '### 结构匹配随机对照补充','',
        '原分析器用 `count=16` 匹配 joint 与 random；joint 实际包含 16 个正向和 16 个负向特征，原 `random_count_matched=true` 只反映配置参数相同，不能解释为总方向数匹配。发现后保留原协议和输出，固定已经选定的 target，另行冻结补充协议：抽取不重叠于目标的 16 个正向和 16 个负向随机 decoder 方向，各组归一化后相减，再匹配总范数。未重新选择 target，也未按 test 结果更换随机集合。', '',
        f"补充 test 的无干预、目标、随机均答对 63/64；平均长度分别为 {m['no_steering_replay']['mean_tokens']:.2f}、{m['selected_target_replay']['mean_tokens']:.2f}、{m['matched_joint_random']['mean_tokens']:.2f} tokens。目标比无干预短 **{r['relative_length_reduction']:.2%}**，比结构匹配随机短 {abs(delta['estimate']):.2f} tokens（配对区间 [{delta['ci_low']:.2f}, {delta['ci_high']:.2f}]）。原 test 门槛与补充 dev/test 门槛均通过，因此进入学生阶段。",'',
        '![结构匹配对照](../figures/phase6_local_length_controlled_strength_v1/04_structural_random_control.png)','',
        '图 4：每组 64 题，补充重放在 C31 完成。原 test 分片还使用 C32；模型权重、随机数及题目相同，但 BF16 推理不保证跨 GPU 架构逐 token 一致。因此原 test 与补充重放分开报告，不合并成 128 道独立题，也不挑选准确率更高的一次作为唯一结果。每次实验内部同题各条件均在同一卡、同一批次布局运行。','',
        '![三个 SAE 的实际训练曲线](../figures/phase6_local_length_controlled_strength_v1/03_sae_training_curves.png)','',
        '图 3：左图是各自抽样分布上的重构损失，不能直接比较同分布优劣；右图是在相同完整 dev 样本上的 explained variance。','',
        '[逐题实例：接近中位数的共同正确例子及首个受损例子](../results/phase6_local_length_controlled_strength_v1/exploratory/analysis/test/paired_examples.md)。选择规则固定且在实例文件中说明，不用例子替代全体统计。“正确”指最终答案验证通过，不代表每一步推导都正确；例如第一个例子的工作收入叙述仍有混乱。受损例子中，干预后把 $4\\times6\\times3$ 错算为 36，说明缩短仍可能伴随算术错误。','']
    return '\n'.join(section)


def student_results(config, student):
    if student is None:return '## 学生阶段\n\n教师完整门槛未通过，登记的学生后续未触发。\n'
    root,_=paths(config);data=read_json(root/'student_followup/DATA_COMPLETE.json')
    generation=read_json(root/'student_followup/teacher_data_summary.json')
    section=['## 已完成的学生蒸馏与完整测分','',
        f"为 881 道训练题生成每条件 4 个候选，三个生成条件共 **{data['all_generation_rows']:,} 条输出**。每条件先合并重复文本，再选最短的正确且非截断候选；三组正确题交集为 **{data['support_count']} 题**。历史自然 short 使用这一相同题集。4 候选与历史 short 的原 16 候选选择预算不同，历史 short 是一个实际数据基线，不能据此单独估计纯干预效应。纯干预的主要比较是相同四候选预算下的 target 与 no-steering/random。",'',
        '下面先报告全量生成、尚未按正确性筛选的教师输出。它覆盖 SAE train/dev/test 的全部 881 题，不能作为另一个独立确认集；共同正确题集也会改变难度分布，必须同时报告覆盖率。','',
        '| 全量教师数据条件 | 正确候选率 | 平均 tokens | 截断率 | 有正确且完整候选的题数/881 |',
        '|---|---:|---:|---:|---:|']
    for row in generation['metrics']:
        section.append(f"| {row['condition']} | {row['candidate_accuracy']:.2%} | {row['mean_output_tokens']:.2f} | {row['hit_cap_fraction']:.2%} | {row['correct_complete_support']} |")
    acc_effect=next(r for r in generation['contrasts'] if r['comparator']=='no_steering' and r['metric']=='accuracy')
    section+=['',f"**全量生成的正确率代价已经可见。** 相对无干预，目标的正确候选率差为 {100*acc_effect['estimate']:+.2f} 个百分点，按题重采样、先平均四候选的 95% 区间为 [{100*acc_effect['ci_low']:+.2f}, {100*acc_effect['ci_high']:+.2f}]。不能把 64 题补充重放中的相同正确率推广成全量无损压缩。后续共同正确题集用于公平 SFT 比较，但不会消除教师生成阶段已经发生的错误。",'',
              '[全量教师候选统计与按题区间](../results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/teacher_data_summary.json)。']
    section += ['',
        '固定 Qwen2.5-1.5B-Instruct 学生、seed 17、LoRA r=4/alpha=16/dropout=0.05、学习率 2e-5、batch 4、梯度累积 1、1 epoch、最大序列 2048、completion-only loss；沿用 E1 验证的 TRL 0.9.6 实现。训练提示词统一使用学生标准提示，不把不同教师指令作为学生条件。8 个 adapter 均完成训练并在 GSM8K `test[50:1319]` 全部 1,269 题上完成 greedy 评估。', '',
        '![学生准确率和配对比较](../figures/phase6_local_length_controlled_strength_v1/05_student_accuracy.png)','',
        '图 5：左图是单个训练 seed 的准确率；右图是相同评估题的配对 bootstrap 区间，不估计训练 seed 变异。六个计划对比的 p 值另外做 Holm 校正。图中 base 复用已验证的 E1 同模型字节、同固定题集和同评估实现预测，不计作本次新评估。','',
        '| 训练预算 | 数据条件 | 答对/1269 | 准确率 | 学生平均输出 tokens |',
        '|---|---|---:|---:|---:|']
    for r in student['metrics']:
        regime,condition,_=r['name'].split('__')
        section.append(f"| {regime} | {condition} | {r['correct']} | {r['accuracy']:.2%} | {r['mean_output_tokens']:.2f} |")
    repeat=read_json(root/'student_followup/repeated_control_audit.json')
    section+=['',
        f"**同数据、同 seed 的执行差异。** matched_random 两组的数据哈希、全部科学设置、219 步预算与初始 LoRA 哈希相同，但分别答对 {repeat['correct_counts'][0]} 和 {repeat['correct_counts'][1]} 题（差 {100*repeat['accuracy_difference_second_minus_first']['estimate']:.2f} 个百分点），有 {repeat['correctness_disagreement_questions']} 题的正确/错误状态不同。最终 LoRA 参数的相对 L2 差为 {100*repeat['relative_l2_parameter_difference']:.4f}%，早期训练日志已出现差异，所以不能只归因于评估输出文件或把两行差值解释成预算收益。",'',
        '原配方未请求严格 `full_determinism`。本轮定位了差异在训练执行阶段出现，但没有隔离具体底层算子原因，也没有事后替换任一结果。固定 seed 不等于逐字节可复现；下面按题 bootstrap 的区间不能覆盖这种训练执行变异或跨 seed 变异。产物完整性审计通过也不代表逐字节复现通过。[重复对照审计](../results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/repeated_control_audit.json)。']
    section+=['',f"可复用 base 为 {student['reused_base']['accuracy']:.2%}；历史 E1 全 881 题 short 为 70.84%。后者与本轮共同题集的 historical-short adapter 不是同一训练数据量，不能直接替代本轮同题集比较。",'',
        '| 预算 | SAE target 相对 | 正确率差（百分点） | 95% 配对 CI | Holm p |',
        '|---|---|---:|---:|---:|']
    for r in student['contrasts']:
        regime=r['target'].split('__')[0];comp=r['comparator'].split('__')[1]
        section.append(f"| {regime} | {comp} | {100*r['estimate']:+.2f} | [{100*r['ci_low']:+.2f}, {100*r['ci_high']:+.2f}] | {r['holm_p']:.4f} |")
    effects=[r for r in student['contrasts'] if '__no_steering__' in r['comparator']]
    supported=[r for r in effects if r['ci_low']>0 and r['holm_p']<.05]
    if len(supported)==len(effects):
        conclusion='两种预算下，SAE 数据学生均优于无干预学生，并通过本轮配对区间和六对比 Holm 检查；这是单种子、已观察 GSM8K 上的探索性收益，仍不证明跨 seed 稳定。'
    elif supported:
        conclusion='学生相对无干预的收益只在部分预算下得到本轮配对统计支持，不能写成对训练预算稳健的提升。'
    else:
        conclusion='本轮没有确立 SAE 数据学生相对无干预学生的准确率提升；教师输出明显缩短，并不自动转化为学生涨点。表中点估计仍须与配对区间和随机/历史 short 对照一起解释。'
    section+=['',f'**学生判定：{conclusion}**','',
        '![监督量与步数](../figures/phase6_local_length_controlled_strength_v1/06_student_training_budgets.png)','',
        '图 6：equal_examples 每题使用一次；equal_target_tokens 复用历史完整轨迹重复算法，重复较短数据以接近四组最大的 completion-token 总量，差距不超过登记的 512 tokens。不裁剪回答。等 token 不等于等优化器步数、等计算或等每题权重，prompt/EOS 开销也未包含在该 token 预算中。最大 token 总量的 matched_random 组无需重复，两种预算的数据文件哈希相同；两次运行不构成独立 seed 复现。','',
        '[共同支持与训练预算](../results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/DATA_COMPLETE.json)、[学生逐模型指标与配对统计](../results/phase6_local_length_controlled_strength_v1/exploratory/student_followup/STUDENT_COMPLETE.json)。','']
    return '\n'.join(section)


def teacher_data_summary(config, generated):
    import numpy as np
    from .utility_analysis import paired_question_bootstrap
    root,_=paths(config);by_condition={};metrics=[]
    for condition in ['no_steering','selected_target','matched_random']:
        rows=[r for r in generated if r['condition']==condition]
        by_question={}
        for row in rows:by_question.setdefault(row['problem_id'],[]).append(row)
        by_condition[condition]={p:{'length':float(np.mean([r['output_token_count'] for r in rs])),
                                    'accuracy':float(np.mean([r['is_correct'] for r in rs]))} for p,rs in by_question.items()}
        metrics.append({'condition':condition,'n':len(rows),'question_count':len(by_question),
            'candidate_accuracy':float(np.mean([r['is_correct'] for r in rows])),
            'mean_output_tokens':float(np.mean([r['output_token_count'] for r in rows])),
            'hit_cap_fraction':float(np.mean([r['hit_max_new_tokens'] for r in rows])),
            'correct_complete_support':len({r['problem_id'] for r in rows if r['is_correct'] and not r['hit_max_new_tokens']})})
    contrasts=[]
    for other in ['no_steering','matched_random']:
        for metric in ['length','accuracy']:
            effect=paired_question_bootstrap({p:r[metric] for p,r in by_condition['selected_target'].items()},
                {p:r[metric] for p,r in by_condition[other].items()},samples=10000,seed=61517)
            contrasts.append({'target':'selected_target','comparator':other,'metric':metric,**effect})
    output=root/'student_followup/teacher_data_summary.json'
    report={'status':'complete','metrics':metrics,'contrasts':contrasts,
            'resampling_unit':'question, averaging four candidates before resampling',
            'scope':'descriptive full training-data generation, not independent confirmation','formal_claim_allowed':False}
    if output.exists():
        if read_json(output)!=report:raise ValueError('Changed full teacher data summary')
    else:write_json_exclusive(output,report)
    return output


def training_exposure_audit(config):
    """Replay the frozen trainer's CPU permutations against sampled trace IDs."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from scipy.stats import spearmanr
    from .sae_local_data import jsonl
    root,_=paths(config);corpus=jsonl(root/'corpus.jsonl')
    lengths=np.array([r['solution_token_count'] for r in corpus]);train=np.array([r['question_split']=='train' for r in corpus])
    results=[];batch=config['sae']['batch_size'];steps=config['sae']['max_steps']
    for item in read_json(root/'SAMPLING_COMPLETE.json')['conditions']:
        manifest=read_json(item['path'])
        filename=next(r['path'] for r in manifest['layers'][0]['samples'] if r['split']=='train')
        with safe_open(filename,framework='pt',device='cpu') as handle:trace=handle.get_tensor('trace_indices')
        generator=torch.Generator(device='cpu').manual_seed(config['sae']['seed']+17)
        permutation=torch.randperm(len(trace),generator=generator);cursor=0
        counts=np.zeros(len(corpus),dtype=np.int64)
        for _ in range(steps):
            if cursor+batch>len(permutation):permutation=torch.randperm(len(trace),generator=generator);cursor=0
            chosen=trace[permutation[cursor:cursor+batch]].numpy();cursor+=batch
            counts+=np.bincount(chosen,minlength=len(corpus))
        results.append({'condition':item['condition'],'draw_pool_tokens':len(trace),'optimizer_token_presentations':int(counts.sum()),
                        'pool_length_exposure_spearman':manifest['length_exposure_spearman'],
                        'actual_optimizer_length_exposure_spearman':float(spearmanr(lengths[train],counts[train]).statistic),
                        'per_trace_presentations':counts.tolist()})
    report={'status':'complete','method':'exact replay of seed+layer CPU randperm, batch cursor and dropped remainder in frozen trainer',
            'source':evidence(ROOT/'scripts/2_4_train_topk_sae.py'),'conditions':results,'formal_claim_allowed':False}
    output=root/'training_exposure_audit.json'
    if output.exists():
        if read_json(output)!=report:raise ValueError('Changed actual training exposure audit')
    else:write_json_exclusive(output,report)
    return output


def paired_examples(config, rows):
    """Deterministic representative and failure examples from the completed test."""
    import numpy as np
    from .sae_local_data import jsonl
    root,_=paths(config)
    selected=read_json(root/'DEV_SELECTION.json')['selected_condition']
    baseline={r['problem_id']:r for r in rows if r['condition']=='no_steering'}
    target={r['problem_id']:r for r in rows if r['condition']==selected}
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl')}
    both=[p for p in baseline if baseline[p]['is_correct'] and target[p]['is_correct']]
    median=float(np.median([target[p]['output_token_count']-baseline[p]['output_token_count'] for p in both]))
    representative=min(both,key=lambda p:(abs(target[p]['output_token_count']-baseline[p]['output_token_count']-median),p))
    failures=sorted(p for p in baseline if baseline[p]['is_correct'] and not target[p]['is_correct'])
    cases=[('共同正确题中最接近长度差中位数的例子',representative)]
    if failures:cases.append(('按题号排序的第一个正确率受损例子',failures[0]))
    output=root/'analysis/test/paired_examples.md'
    lines=['# 选定联合干预的逐题实例','',
        '来自原定 64 道 test 题，同题条件在同 GPU、同批次布局执行。第一个例子按共同正确题的长度差中位数确定；第二个按题号选择 baseline 正确而 target 错误的首例。它们描述行为，不把 feature 命名为某种确定语义。','',
        f'固定条件：`{selected}`。所有 64 题的总体统计见相邻 `teacher_report.md`。','']
    for title,pid in cases:
        base=baseline[pid];changed=target[pid]
        lines.extend([f'## {title}：{pid}','',f"Gold：`{base['gold_answer']}`。",'',
            '```text',corpus[pid]['student_prompt'],'```','',
            f"无干预：{base['output_token_count']} tokens，正确={base['is_correct']}。",'',
            '```text',base['response'],'```','',
            f"联合干预：{changed['output_token_count']} tokens，正确={changed['is_correct']}。",'',
            '```text',changed['response'],'```',''])
    text='\n'.join(lines)
    if output.exists():
        if output.read_text()!=text:raise ValueError('Changed paired examples')
    else:output.write_text(text)
    return output


def repeated_control_audit(config):
    """Measure observed same-data, same-seed variability without replacing any run."""
    import torch
    from safetensors.torch import load_file
    from transformers import TrainingArguments
    from .sae_local_data import jsonl
    from .utility_analysis import paired_question_bootstrap
    root,_=paths(config);sft=root/'student_followup/sft'
    names=[f'{regime}__matched_random__seed_17' for regime in ['equal_examples','equal_target_tokens']]
    runs=[read_json(sft/'replication/configs'/f'{name}.json') for name in names]
    # Strip only experiment/output identifiers and input paths; retain all settings.
    import copy
    normalized=[]
    for run in runs:
        item=copy.deepcopy(run);item.pop('experiment_name');item['data'].pop('train_path');item['training'].pop('output_dir');normalized.append(item)
    if normalized[0]!=normalized[1]:raise ValueError('Repeated-control configurations differ scientifically')
    data=[evidence(Path(r['data']['train_path'])) for r in runs]
    if data[0]['sha256']!=data[1]['sha256']:raise ValueError('Repeated controls do not use identical data')
    directories=[sft/'replication/checkpoints'/n for n in names]
    training=[read_json(p/'training_metrics.json') for p in directories]
    weights=[load_file(str(p/'adapter_model.safetensors'),device='cpu') for p in directories]
    if weights[0].keys()!=weights[1].keys():raise ValueError('Different LoRA parameter keys')
    norm=0.;delta=0.;maximum=0.;changed=0;total=0
    for key in weights[0]:
        a=weights[0][key].double();b=weights[1][key].double();difference=b-a
        norm+=float(a.square().sum());delta+=float(difference.square().sum())
        maximum=max(maximum,float(difference.abs().max()));changed+=int((a!=b).sum());total+=a.numel()
    predictions=[{r['problem_id']:r for r in jsonl(sft/'evaluation'/n/'predictions.jsonl')} for n in names]
    effect=paired_question_bootstrap({p:float(r['is_correct']) for p,r in predictions[1].items()},
        {p:float(r['is_correct']) for p,r in predictions[0].items()},samples=10000,seed=61617)
    report={'status':'audited','runs':names,'training_data':data,'scientific_configs_identical':True,
        'initial_adapter_sha256':[r['initial_adapter_sha256'] for r in training],
        'first_logged_training_records':[r['log_history'][0] for r in training],
        'final_adapter_files':[evidence(p/'adapter_model.safetensors') for p in directories],
        'parameter_count':total,'changed_parameter_count':changed,'max_absolute_parameter_difference':maximum,
        'relative_l2_parameter_difference':(delta/max(norm,1e-30))**.5,
        'default_full_determinism':TrainingArguments.__dataclass_fields__['full_determinism'].default,
        'explicit_full_determinism_in_registered_training':runs[0]['training'].get('full_determinism'),
        'correct_counts':[sum(r['is_correct'] for r in p.values()) for p in predictions],
        'correctness_disagreement_questions':sum(predictions[0][p]['is_correct']!=predictions[1][p]['is_correct'] for p in predictions[0]),
        'accuracy_difference_second_minus_first':effect,'bitwise_training_reproducible':changed==0,
        'interpretation':'Observed same-data, same-initialization, same-seed training executions diverged before evaluation. Exact low-level cause was not isolated; strict full determinism was not configured. Keep both registered runs and do not interpret them as independent seeds.',
        'formal_claim_allowed':False}
    output=root/'student_followup/repeated_control_audit.json'
    if output.exists():
        if read_json(output)!=report:raise ValueError('Changed repeated-control audit')
    else:write_json_exclusive(output,report)
    return output


def preview(config):
    """Publish completed teacher findings without claiming the student stage finished."""
    plots(config);learning_curves(config);outcome_plots(config,None)
    doc=ROOT/'docs/phase6_local_length_controlled_strength_report_zh.md'
    text=doc.read_text();methods='## 问题与范围'+text.split('## 问题与范围',1)[1]
    doc.write_text('# 当前数据上的长度控制 SAE 与加强干预实验\n\n'
        '实验：`phase6_local_length_controlled_strength_v1`。开始日期：2026-09-10。'
        '状态：三个 SAE、教师网格与结构匹配补充已完成；学生阶段正在运行，最终审计尚未完成。\n\n'
        +teacher_results(config)+'\n'+methods)


def finish(config):
    from .sae_local_data import jsonl
    from . import legacy_replication as legacy
    from . import local_short_gain_check as gain
    from .verifiers import extract_final_answer,verify_answer
    root,_=paths(config);checks=[]
    def check(item):verify(item);checks.append(item)
    original=read_json(root/'TEACHER_DECISION.json')
    corrected=read_json(root/'MATCHED_CONTROL_DECISION.json')
    triggered=original['student_authorized_by_protocol'] and corrected['matched_control_gate_passed']
    student=read_json(root/'student_followup/STUDENT_COMPLETE.json') if triggered else None
    student_initializations=[];student_parameter_counts=[]
    if student and (student['training_runs']!=8 or student['new_evaluations']!=8):raise ValueError('Incomplete student matrix')
    recovery=read_json(root/'RECOVERY_COMPLETE.json')
    for item in [recovery['corpus'],recovery['config'],recovery['source'],*recovery['teacher_files'],*recovery['ranks'].values()]:check(item)
    from .factorial import canonical_sha256
    import numpy as np
    from safetensors import safe_open
    source_corpus=jsonl(root/'corpus.jsonl');trace_indices=[];coordinates=[];token_count=0
    lengths=np.array([r['solution_token_count'] for r in source_corpus]);offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    expected_tokens=np.array([t for r in source_corpus for t in r['token_ids']])
    for shard in range(config['activation_extraction']['trajectory_shards']):
        extraction=read_json(root/f'extraction_shard_{shard}.json')
        if extraction['config_hash']!=canonical_sha256(config) or extraction['corpus_sha256']!=recovery['corpus']['sha256']:raise ValueError('Changed extraction protocol')
        trace_indices+=extraction['trace_indices'];check(extraction['source'])
        for item in extraction['chunks']:
            check(item)
            with safe_open(item['path'],framework='pt',device='cpu') as handle:
                trace=handle.get_tensor('trace_indices').numpy().astype(np.int64)
                position=handle.get_tensor('positions').numpy().astype(np.int64)
                tokens=handle.get_tensor('token_ids').numpy()
            if np.any(trace<0) or np.any(trace>=len(source_corpus)) or np.any(position<0) or np.any(position>=lengths[trace]):raise ValueError('Invalid extracted token coordinates')
            flattened=offsets[trace]+position
            if len(tokens)!=item['token_count'] or not np.array_equal(tokens,expected_tokens[flattened]):raise ValueError('Extracted token identity mismatch')
            coordinates.append(flattened);token_count+=len(tokens)
    if sorted(trace_indices)!=list(range(len(source_corpus))) or not np.array_equal(np.sort(np.concatenate(coordinates)),np.arange(len(expected_tokens))):raise ValueError('Missing or duplicate extracted traces/tokens')
    for filename,digest in read_json(root/'sae_launch.json')['source_hashes'].items():check({'path':str(ROOT/filename),'sha256':digest})
    for item in read_json(root/'protocol/dependency_sources.json')['files']:check(item)
    for condition in config['sampling']['conditions']:
        summary=read_json(root/'feature_screen'/condition/'SCREEN_COMPLETE.json')
        for key in ['checkpoint','source','raw_means']:check(summary[key])
        metrics=read_json(root/'sae_training'/condition/'training_metrics.json')
        if metrics['max_steps']!=config['sae']['max_steps']:raise ValueError('Incomplete SAE training')
        check({'path':metrics['sample_manifest_path'],'sha256':metrics['sample_manifest_sha256']})
        layer=read_json(metrics['sample_manifest_path'])['layers'][0]
        for item in layer['samples']:check(item)
        check({'path':layer['normalizer_path'],'sha256':layer['normalizer_sha256']})
    exposure_output=training_exposure_audit(config)
    protocol=read_json(root/'protocol/generation_protocol.json')
    for key in ['source_code','config_hash']:check(protocol[key])
    check(read_json(root/'protocol/analysis_protocol.json')['source'])
    for item in protocol['sources'].values():check(item)
    counts={}
    for split in ['dev','test']:
        rows=audited_records(config,split);counts[split]=len(rows)
        if split=='test':examples_output=paired_examples(config,rows)
        for shard in range(config['generation']['shards']):
            check(read_json(root/'generation'/split/f'shard_{shard}'/'GENERATION_COMPLETE.json')['predictions'])
    addon=read_json(root/'matched_joint_control/protocol.json')
    for key in ['config','source','selection','checkpoint','parent_generation_protocol']:check(addon[key])
    replay=[]
    for shard in range(addon['addon']['shards']):
        marker=read_json(root/'matched_joint_control'/f'shard_{shard}'/'COMPLETE.json')
        check(marker['predictions']);check(marker['protocol']);replay+=jsonl(marker['predictions']['path'])
    expected={(t['split'],p,s['name']) for t in addon['tasks'] for p in t['problem_ids'] for s in addon['specs']}
    observed=[(r['question_split'],r['problem_id'],r['condition']) for r in replay]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError('Incomplete structural control')
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl')}
    from .student_prompts import build_student_math_prompt
    normalize=lambda value:re.sub(r'\s+',' ',value).strip()
    training_prompts={normalize(r['student_prompt']) for r in corpus.values()}
    eval_fixture=ROOT/'results/phase1_local_short_gain_check_v1/imported/locked_evaluation_questions.jsonl'
    evaluation_prompts={normalize(build_student_math_prompt(r['question'])) for r in jsonl(eval_fixture)}
    question_overlap=len(training_prompts & evaluation_prompts)
    if question_overlap:raise ValueError('Training/evaluation exact question overlap')
    for row in replay:
        if row['gold_answer']!=extract_final_answer(corpus[row['problem_id']]['gold_answer']) or row['is_correct']!=verify_answer(extract_final_answer(row['response']),row['gold_answer']):raise ValueError('Invalid replay correctness')
    counts['structural_replay']=len(replay)
    if student:
        output=root/'student_followup';main=read_json(output/'main_protocol.json')
        for key in ['source','control','parent_config','teacher_gate','structural_gate']:check(main[key])
        generated=[]
        for shard in range(main['shards']):
            marker=read_json(output/'generation'/f'shard_{shard}'/'COMPLETE.json')
            check(marker['predictions']);check(marker['protocol']);generated+=jsonl(marker['predictions']['path'])
        expected={(p,s['name'],c) for p in main['question_ids'] for s in main['specs'] for c in range(main['candidates_per_question'])}
        observed=[(r['problem_id'],r['condition'],r['candidate_index']) for r in generated]
        if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError('Incomplete main teacher generation')
        specs={s['name']:s for s in main['specs']};seeds={}
        for row in generated:
            key=(row['problem_id'],row['candidate_index']);spec=specs[row['condition']]
            if row['spec']!=spec or row['output_token_count']!=len(row['token_ids']):raise ValueError('Changed main generation spec or length')
            if key in seeds and row['seed']!=seeds[key]:raise ValueError('Unpaired main RNG')
            seeds[key]=row['seed']
            if row['gold_answer']!=extract_final_answer(corpus[row['problem_id']]['gold_answer']) or row['is_correct']!=verify_answer(extract_final_answer(row['response']),row['gold_answer']):raise ValueError('Incorrect main generation gold')
            if row['diagnostics']['max_delta_to_hidden_norm_fraction']>spec['rho']+.01:raise ValueError('Main norm exceeds tolerance')
        counts['student_data_generation']=len(generated)
        teacher_data_summary(config,generated)
        student_config=legacy.config_load(output/'sft/student_config.json')
        manifest=legacy.source_hashes(student_config)
        for item in manifest['files']:check(item)
        admission=read_json(root/'protocol/student_launch_admission_addendum.json')
        for item in admission['files'].values():check(item)
        check(admission['core'])
        registered_metrics={r['name']:r for r in student['metrics']}
        for entry in manifest['runs']:
            name=entry['name'];trained=legacy.validate_training(student_config,name);evaluated=gain.validate_evaluation(student_config,name)
            student_initializations.append(trained['initial_adapter_sha256']);student_parameter_counts.append(trained['trainable_parameter_count'])
            if trained['optimizer_steps']!=registered_metrics[name]['training_steps'] or sum(r['is_correct'] for r in evaluated)!=registered_metrics[name]['correct']:raise ValueError('Student summary does not match artifacts')
            for p in [(output/'sft/replication/checkpoints'/name/filename) for filename in ['adapter_config.json','adapter_model.safetensors','training_metrics.json','TRAIN_COMPLETE']]:checks.append(evidence(p))
            for p in [(output/'sft/evaluation'/name/filename) for filename in ['predictions.jsonl','summary.json','EVALUATION_COMPLETE']]:checks.append(evidence(p))
        check(student['reused_base']['predictions'])
        if len(set(student_initializations))!=1 or len(set(student_parameter_counts))!=1:raise ValueError('Student initialization or trainable parameter count differs across cells')
        repeated_control_audit(config)
    tests=[(root/'logs/intervention_tests_v2.log',3),(root/'logs/c32_runtime_tests.log',3),(root/'logs/joint_control_tests.log',1)]
    for filename,count in tests:
        contents=filename.read_text()
        if f'Ran {count} test' not in contents or '\nOK\n' not in contents:raise ValueError(f'Required tests failed: {filename}')
    figures=plots(config)+learning_curves(config)+outcome_plots(config,student)
    doc=ROOT/'docs/phase6_local_length_controlled_strength_report_zh.md'
    current=doc.read_text()
    # Keep the written method/history sections; replace the pending result prefix.
    methods='## 问题与范围'+current.split('## 问题与范围',1)[1]
    methods=methods.split('\n## 完整产物与审计',1)[0].rstrip()
    methods=methods.replace('运行中不宣称完成。','本轮已达到上述完整性要求，见最终审计与完成标记。')
    if student:
        methods=methods.replace('如教师门槛通过，才启动登记的四候选、881 题、单 seed=17 学生后续：','本轮教师门槛通过，已经完成登记的四候选、881 题、单 seed=17 学生后续：')
    intro=['# 当前数据上的长度控制 SAE 与加强干预实验','',
        '实验：`phase6_local_length_controlled_strength_v1`。开始日期：2026-09-10。状态：完整实验、逐题评估与最终审计已完成；单种子探索性证据。','',
        '**核心结论：** 长度均衡抽样可以去除训练曝光偏差；单纯加强 short 方向可能损伤正确率，而 dev 选定的 short/long 联合干预在本轮留出题上缩短输出约 22%–24%。学生准确率是否改善必须依据下面独立的学生对照，不能由教师缩短推断。','',
        teacher_results(config),student_results(config,student),methods,'',
        '## 完整产物与审计','',
        '[Dev 全条件与选定依据](../results/phase6_local_length_controlled_strength_v1/exploratory/analysis/dev/teacher_report.md)、[Test 全 27 条件](../results/phase6_local_length_controlled_strength_v1/exploratory/analysis/test/teacher_report.md)、[结构匹配补充](../results/phase6_local_length_controlled_strength_v1/exploratory/matched_joint_control/analysis.json)、[补充协议](../configs/phase6_local_matched_joint_control_v1.json)、[原教师决策](../results/phase6_local_length_controlled_strength_v1/exploratory/TEACHER_DECISION.json)、[补充教师决策](../results/phase6_local_length_controlled_strength_v1/exploratory/MATCHED_CONTROL_DECISION.json)、[最终审计](../results/phase6_local_length_controlled_strength_v1/exploratory/FINAL_AUDIT.json)、[完成标记](../results/phase6_local_length_controlled_strength_v1/exploratory/EXPERIMENT_COMPLETE.json)。','',
        '原教师决策文件是在门槛判断时冻结的，里面 `student_triggered=false` 表示当时尚未启动；后续是否完成以 `student_followup/STUDENT_COMPLETE.json` 与总完成标记为准，不改写已被后续协议哈希绑定的旧状态。','',
        '完成标记仅证明本轮登记的探索性工作完成，不等于正式论文结论。主要限制包括已观察的单一 GSM8K 数据、仅有正确 rank 轨迹、一个 SAE/学生 seed、一个随机方向集合、教师门槛的正确率点估计规则，以及 equal-token 重复造成的步数和题目权重差异。未做独立 OOD 或训练 seed 复现。','']
    if student:
        m={r['name']:r for r in student['metrics']};sentences=[];specific=[]
        gm={r['condition']:r for r in read_json(root/'student_followup/teacher_data_summary.json')['metrics']}
        compression=1-gm['selected_target']['mean_output_tokens']/gm['no_steering']['mean_output_tokens']
        intro[4:4]=[f"**全量教师取舍：** 联合干预把 881 题四候选的平均输出缩短 {compression:.2%}，同时把正确候选率从 {gm['no_steering']['candidate_accuracy']:.2%} 降至 {gm['selected_target']['candidate_accuracy']:.2%}；这是有正确率代价的压缩。",'']
        for regime,label in [('equal_examples','等样本'),('equal_target_tokens','等目标 token')]:
            target=m[f'{regime}__selected_target__seed_17'];base=m[f'{regime}__no_steering__seed_17']
            sentences.append(f"{label}下，SAE 学生 {target['accuracy']:.2%}，无干预学生 {base['accuracy']:.2%}，差 {100*(target['accuracy']-base['accuracy']):+.2f} 个百分点")
            contrasts=[r for r in student['contrasts'] if r['target'].startswith(regime+'__') and any(f'__{c}__' in r['comparator'] for c in ['no_steering','matched_random'])]
            specific.append(len(contrasts)==2 and all(r['ci_low']>0 and r['holm_p']<.05 for r in contrasts))
        if all(specific):boundary='两种预算下都得到超越无干预和匹配随机对照的配对统计支持，但仍只代表单训练种子的探索性结果。'
        elif any(specific):boundary='仅部分预算下得到同时超越无干预和匹配随机对照的配对统计支持，尚不能声称跨预算稳健收益。'
        else:boundary='本轮尚未确立同时超越无干预和匹配随机对照的学生收益；教师缩短与学生涨点是两个不同结论。'
        intro[4:4]=['**学生结果：** '+'；'.join(sentences)+'。'+boundary,'']
        repeat=read_json(root/'student_followup/repeated_control_audit.json')
        intro[6:6]=[f"**复现限制：** 两次相同数据、相同 seed 的随机对照训练出现 {100*repeat['accuracy_difference_second_minus_first']['estimate']:.2f} 个百分点的分数差；模型权重和早期训练日志已不同。本轮未启用严格确定性训练，因此不能把这些单次执行的细小差值当作稳定收益。",'']
    doc.write_text('\n'.join(intro))
    checked_links=[]
    for link in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
        if link.startswith(('http:','https:','#')):continue
        target=(doc.parent/link.split('#')[0]).resolve()
        if target in [root/'FINAL_AUDIT.json',root/'EXPERIMENT_COMPLETE.json']:continue
        if not target.exists():raise ValueError(f'Broken final report link: {link}')
        checked_links.append(str(target))
    source_bundle=root/'protocol/source_bundle';source_bundle.mkdir(exist_ok=False)
    outputs=[doc,ROOT/'docs/sae_representation_one_slide_en.md',exposure_output,examples_output,*figures,*[t[0] for t in tests],root/'TEACHER_DECISION.json',root/'MATCHED_CONTROL_DECISION.json',
             root/'DEV_SELECTION.json',root/'analysis/dev/teacher_summary.json',root/'analysis/test/teacher_summary.json',
             root/'matched_joint_control/analysis.json']
    if student:outputs.extend([root/'student_followup/STUDENT_COMPLETE.json',root/'student_followup/DATA_COMPLETE.json',root/'student_followup/teacher_data_summary.json',root/'student_followup/repeated_control_audit.json'])
    if student:
        outputs.append(root/'protocol/student_launch_admission_addendum.json')
        for row in student['metrics']:
            log=root/'logs'/f"student_{row['name']}_admission.log"
            if not log.is_file() or 'Selected stable memory-fit GPUs' not in log.read_text():raise ValueError('Missing per-student GPU admission evidence')
            outputs.append(log)
    sources={ROOT/'src/length_budget_distill'/f'{name}.py' for name in ['sae_local_data','sae_local_screen','sae_norm_intervention','sae_joint_control','sae_local_report','sae_local_student','sae_local_finalize']}
    for item in checks:
        source=Path(item['path'])
        try:relative=source.relative_to(ROOT)
        except ValueError:continue
        if relative.parts[0] in ['src','scripts','configs'] and source.suffix in ['.py','.sh','.json']:sources.add(source)
    sources.update((ROOT/'scripts').glob('6_[1-7]_*.py'))
    sources.update((ROOT/'scripts/slurm').glob('6_[1-8]_*.sh'))
    for source in sorted(sources):
        destination=source_bundle/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes());outputs.append(destination)
    from datetime import datetime,timezone
    completed_at=datetime.now(timezone.utc).isoformat()
    audit={'status':'passed','completed_at_utc':completed_at,'scope':'exploratory length-controlled SAE, teacher intervention, structural control and conditional student distillation',
           'three_sae_training_complete':True,'teacher_prediction_counts':counts,
           'activation_extraction':{'shards':4,'traces':len(source_corpus),'tokens':token_count,'token_identity_and_duplicate_missing_audit':'passed'},
           'normalized_exact_training_eval_question_overlap':question_overlap,
           'student_initial_adapter_hashes':student_initializations,'student_trainable_parameter_counts':student_parameter_counts,
           'bitwise_student_training_reproducibility':read_json(root/'student_followup/repeated_control_audit.json')['bitwise_training_reproducible'] if student else None,
           'duplicate_missing_gold_prefix_norm_audit':'passed','hash_checks':checks,
           'report_link_checks':checked_links,'outputs':[evidence(p) for p in outputs],
           'student_followup':'complete' if student else 'not_triggered_teacher_gate_failed',
           'student_training_runs':student['training_runs'] if student else 0,'formal_claim_allowed':False}
    write_json_exclusive(root/'FINAL_AUDIT.json',audit)
    write_json_exclusive(root/'EXPERIMENT_COMPLETE.json',{'status':'complete','completed_at_utc':completed_at,'scope':audit['scope'],
        'audit':evidence(root/'FINAL_AUDIT.json'),'report':evidence(doc),
        'student_training_runs':audit['student_training_runs'],
        'bitwise_student_training_reproducible':audit['bitwise_student_training_reproducibility'],'formal_claim_allowed':False})
    print(json.dumps({'status':'complete','report':str(doc),'student_runs':audit['student_training_runs']}),flush=True)
