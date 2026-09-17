"""Audit complete base/author/replica TokenSkip evaluation and publish curves."""
from pathlib import Path
import math
import shutil

from .experiment_io import read_json
from .factorial import file_sha256
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import resolve,save,seal,verify
from .baseline_method_analysis import audit_cohort
from .tokenskip_reproduction import evaluation_cells,expected_training_steps
from .utility_analysis import paired_question_bootstrap
from .factorial_analysis import holm_adjust

CODE=Path(__file__).resolve().parents[2]


def analyze(config_path):
    spec=read_json(config_path);parent=resolve(spec['parent_root']);out=resolve(spec['result_root'])
    grader=None;review_index=None;grading_changes=[];regraded_records=[];legacy_metrics={}
    from .gsm8k_answer_review import VERSION as REVIEW_VERSION,load_review_index,apply_reviewed_grade
    if spec.get('grader_version')=='gsm8k_explicit_answer_quantity_v3':
        from .gsm8k_grading_v3 import grade_gsm8k_response as grader
    elif spec.get('grader_version')==REVIEW_VERSION:
        review_index=load_review_index(resolve(spec['uniform_review_root']))
    elif 'grader_version' in spec:
        raise ValueError('Unknown regrading version')
    if out.exists():raise FileExistsError(out)
    markers=[parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json']
    if review_index is not None:markers.append(resolve(spec['uniform_review_root'])/'COMPLETE.json')
    for path in markers:verify(path)
    cfg=read_json(parent/'protocol/frozen_config.json')
    model_files=read_json(parent/'inputs/model_hashes.json');model_hash_checks=0
    for files in model_files.values():
        for path,digest in files.items():
            if file_sha256(path)!=digest:raise ValueError('Changed model file: '+path)
            model_hash_checks+=1
    adapter=Path(cfg['checkpoint_root'])/'replica'
    verify(adapter/'TRAIN_COMPLETE.json');markers.append(adapter/'TRAIN_COMPLETE.json')
    training=read_json(adapter/'training_metrics.json');data=read_json(parent/'inputs/data_audit.json')
    if training['smoke_only'] or training['records']!=data['train_records'] or training['optimizer_steps']!=expected_training_steps(data['train_records'],cfg):
        raise ValueError('Replica is not the complete registered SFT run')
    expected=[r['problem_id'] for r in read_jsonl(parent/'inputs/evaluation.jsonl')]
    if len(expected)!=1269 or len(set(expected))!=1269:raise ValueError('Locked evaluation cohort changed')
    rows_by_model={};metrics={};record_count=0
    for name in ('base','author','replica'):
        root=parent/'evaluation'/name
        verify(root/'COMPLETE.json');markers.append(root/'COMPLETE.json')
        rows_by_model[name]={};metrics[name]={}
        legacy_metrics[name]={}
        for cell in evaluation_cells(cfg,name):
            directory=root/cell['name'];verify(directory/'COMPLETE.json');markers.append(directory/'COMPLETE.json')
            rows=list(read_jsonl(directory/'predictions.jsonl'));by_id=audit_cohort(rows,expected)
            if any(r['cell']!=cell for r in rows):raise ValueError('Ratio or cap changed')
            if any(r['output_tokens']!=len(r['token_ids']) or r['output_tokens']>cell['max_new_tokens'] for r in rows):
                raise ValueError('Invalid generated token accounting')
            stored=read_json(directory/'metrics.json')
            recomputed={'n':len(rows),'accuracy':sum(r['is_correct'] for r in rows)/len(rows),
                'mean_output_tokens':sum(r['output_tokens'] for r in rows)/len(rows),
                'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rows)/len(rows)}
            if any(not math.isclose(stored[k],v,abs_tol=1e-12) for k,v in recomputed.items()):
                raise ValueError('Aggregate metrics do not match per-example records')
            legacy_metrics[name][cell['name']]=stored.copy()
            if grader is not None or review_index is not None:
                updated=[]
                for row in rows:
                    if review_index is not None:
                        revised=apply_reviewed_grade(row,review_index[(f"tokenskip_{name}/{cell['name']}",row['problem_id'])],
                            text_field='prediction_text',gold_field='answer',tokens_field='output_tokens')
                        revised['model']=name
                    else:
                        grade=grader(row['prediction_text'],row['answer'],timeout_seconds=spec['grading_timeout_seconds'])
                        revised={**row,**grade,'legacy_is_correct':row['is_correct'],
                                 'legacy_predicted_answer':row['predicted_answer'],'model':name}
                    updated.append(revised)
                    if bool(revised['is_correct'])!=bool(row['is_correct']):grading_changes.append(revised)
                regraded_records.extend(updated);by_id=audit_cohort(updated,expected)
                stored={**stored,'accuracy':sum(r['is_correct'] for r in updated)/len(updated)}
            metrics[name][cell['name']]=stored;rows_by_model[name][cell['name']]=by_id
            record_count+=len(rows)
    base=rows_by_model['base']['ratio_1.0__fixed'];contrasts=[]
    for name in ('author','replica'):
        for cell,rows in rows_by_model[name].items():
            result=paired_question_bootstrap({p:float(r['is_correct']) for p,r in rows.items()},
                {p:float(r['is_correct']) for p,r in base.items()},
                samples=spec['bootstrap_samples'],seed=spec['bootstrap_seed'])
            contrasts.append({'model':name,'cell':cell,'comparison':'minus base at fixed 512 cap',**result})
    for result,p in zip(contrasts,holm_adjust([r['bootstrap_p_value'] for r in contrasts])):
        result['holm_p_value']=float(p)
    out.mkdir(parents=True)
    summary={'status':'complete','formal_claim_allowed':False,'metrics':metrics,'training':training,
        'data_audit':data,'model_file_hash_checks':model_hash_checks,'audited_predictions':record_count,
        'paired_accuracy_contrasts':contrasts,'training_seed_count':1,
        'author_checkpoint_state':read_json(parent/'inputs/author_checkpoint_state.json'),
        'claim_boundary':spec['claim_boundary'],'adaptations':cfg['adaptations'],
        'post_hoc_regrading':grader is not None or review_index is not None,
        'uniform_review_root':spec.get('uniform_review_root'),
        'grader_version':spec.get('grader_version','gsm8k_math_verify_v2'),
        'changed_grades':len(grading_changes),'legacy_metrics':legacy_metrics}
    save(out/'summary.json',summary)
    if grader is not None or review_index is not None:
        write_jsonl(out/'regraded_predictions.jsonl',regraded_records)
        write_jsonl(out/'grading_changes.jsonl',grading_changes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    baseline=metrics['base']['ratio_1.0__fixed']
    for ax,policy in zip(axes,('fixed','scaled')):
        for name,color in (('author','#6c757d'),('replica','#2a9d8f')):
            cells=[metrics[name][f"ratio_{r:.1f}__{'fixed' if r==1. else policy}"] for r in cfg['ratios']]
            ax.plot([r['mean_output_tokens'] for r in cells],[100*r['accuracy'] for r in cells],
                    marker='o',color=color,label='Published adapter' if name=='author' else 'SFT replica (seed 42)')
            for ratio,row in zip(cfg['ratios'],cells):
                ax.annotate(f'{ratio:.1f}',(row['mean_output_tokens'],100*row['accuracy']),
                            xytext=(3,4),textcoords='offset points',fontsize=8,color=color)
        ax.scatter([baseline['mean_output_tokens']],[100*baseline['accuracy']],marker='*',s=110,
                   color='#e76f51',label='Base (512 cap)',zorder=4)
        ax.set_xlabel('Mean generated tokens');ax.set_ylabel('GSM8K accuracy (%)')
        ax.set_title('Fixed 512-token cap' if policy=='fixed' else 'Cap = floor(512 x ratio)')
        ax.legend(fontsize=8)
    fig.suptitle('TokenSkip: locked 1,269-question evaluation')
    fig.savefig(out/'tokenskip_accuracy_length.png',dpi=180)
    fig.savefig(out/'tokenskip_accuracy_length.pdf');plt.close(fig)
    report=['# TokenSkip 原设置训练复现','','![准确率与长度曲线](tokenskip_accuracy_length.png)','',
        f"作者公开数据共 {data['author_records']:,} 道唯一题，训练 {data['train_records']:,} 条，验证 {data['validation_records']:,} 条；与测试题零重合，无 token 截断。",
        f"Qwen2.5-3B-Instruct / LoRA rank 8、alpha 16 / seed 42；本次完成 {training['optimizer_steps']:,} 个优化步，配置 3 epochs，Trainer 实际 epoch={training['actual_epoch']:.6f}。",
        f"训练用时 {training['elapsed_seconds']/60:.2f} 分钟，峰值 GPU allocated memory {training['peak_gpu_allocated_mib']:.1f} MiB。此处为训练阶段成本，不含数据构造和评估。",
        f"共审计 {record_count:,} 条预测：base 1 个条件、作者及复现 adapter 各 11 个不同条件。ratio=1 的两种 cap 策略相同，只运行一次。",'',
        '| 模型 | 比例 | Cap 策略 | 准确率 | 平均 tokens | Cap hit |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for name,cells in metrics.items():
        for cell,row in cells.items():
            report.append(f"| {name} | {row['cell']['ratio']:.1f} | {row['cell']['cap_policy']} | {100*row['accuracy']:.2f}% | {row['mean_output_tokens']:.2f} | {100*row['cap_hit_rate']:.2f}% |")
    report.extend(['',
        f"评分版本：{summary['grader_version']}；单独重评分={summary['post_hoc_regrading']}，改变 {len(grading_changes)} 条正误标签。原始预测和评分保持不变，legacy_metrics 保存原分数。",
        '如使用统一答案复核，疑难项由项目助手在隐藏方法与参考答案的队列中审查；截断且缺少最终答案声明的记录按格式规则判分。这不是独立人工验证，仍保留原评分作为敏感性对照。',
        '作者公开 adapter 附带 trainer_state 为 global_step=600、max_steps=2238；该元数据不能证明其完成了三轮训练。因此作者 checkpoint 和本次完整训练 replica 分别报告。',
        '固定 cap 曲线用于区分比例条件化与输出上限的作用；scaled 曲线保留上游推理设置。表内均与固定 512 cap 的 base 比较，不能称为所有行均匹配 cap。',
        'summary.json 保存逐题配对 bootstrap 区间及 22 项对 base 比较的 Holm 校正；单训练 seed 不估计训练种子方差。',
        '训练保留作者输入和输出、Qwen 默认 chat system；评估保留作者显式 system prompt。使用项目 TRL、BF16 SDPA 和锁定测试子集等适配已记录，不直接对照论文的完整测试分数。','',spec['claim_boundary'],''])
    (out/'report_zh.md').write_text('\n'.join(report))
    provenance=out/'provenance';provenance.mkdir()
    shutil.copy2(config_path,provenance/'analysis_config.json')
    shutil.copytree(CODE/'src/length_budget_distill',provenance/'length_budget_distill',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path in (CODE/'scripts/13_10_analyze_tokenskip.py',CODE/'scripts/slurm/13_4_analyze_tokenskip.sh'):
        shutil.copy2(path,provenance)
    seal(out/'COMPLETE.json',markers+[p for p in out.rglob('*') if p.is_file()],
         formal_claim_allowed=False,stage='tokenskip_original_setting_sft_reproduction',audited_predictions=record_count)
    return summary
