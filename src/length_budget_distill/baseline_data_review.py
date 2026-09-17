"""Publish explicit source-data corrections without changing the initial audit."""
from collections import Counter
from pathlib import Path
import re
import shutil

from .experiment_io import read_json
from .factorial import canonical_sha256
from .records import read_jsonl,write_jsonl
from .ncsu_reproduction import resolve,save,seal,verify

CODE=Path(__file__).resolve().parents[2]


def number_masked_question(text):
    return re.sub(r'\s+',' ',re.sub(r'\d+(?:\.\d+)?','<NUM>',text)).strip()


def review(config_path):
    cfg=read_json(config_path);parent=resolve(cfg['parent_root']);out=resolve(cfg['result_root'])
    if out.exists():raise FileExistsError(out)
    verify(parent/'COMPLETE.json')
    def read(relative):return list(read_jsonl(parent/relative))
    near=read('audits/near_train_eval_review.jsonl');duplicates=read('audits/exact_training_duplicates.jsonl')
    failed=read('audits/unparsed_training_gold.jsonl')
    if {r['problem_id'] for r in failed}!=set(cfg['gold_repairs']) or len(failed)!=cfg['expected_gold_repairs']:
        raise ValueError('Reviewed gold-repair cohort changed')
    exclusions={r['query_id']:{'problem_id':r['query_id'],'reason':'near_evaluation_question',
        'related_evaluation_id':r['reference_id'],'policy':cfg['near_match_policy']} for r in near}
    if len(exclusions)!=cfg['expected_near_match_exclusions'] or len(duplicates)!=cfg['expected_duplicate_exclusions']:
        raise ValueError('Reviewed decontamination cohort changed')
    for row in duplicates:
        exclusions[row['duplicate_id']]={'problem_id':row['duplicate_id'],'reason':'exact_training_duplicate',
                                         'retained_id':row['retained_id']}
    source=read('sources/math_train.jsonl');eligible=[];repairs=[]
    for row in source:
        if row['problem_id'] in exclusions:continue
        correction=cfg['gold_repairs'].get(row['problem_id'])
        updated={**row,'preflight_row_sha256':canonical_sha256(row),'reviewed_gold_override':correction is not None}
        if correction:
            updated.update(source_answer=row['answer'],answer=correction['answer'],answer_kind=correction['kind'],
                           gold_review=correction)
            repairs.append({'problem_id':row['problem_id'],'source_answer':row['answer'],
                'source_reference_sha256':canonical_sha256(row['reference_solution']),**correction})
        eligible.append(updated)
    gsm={r['problem_id']:r for r in read('sources/gsm8k.jsonl')}
    hard={r['problem_id']:r for r in read('sources/gsm8k_hard.jsonl')}
    mappings=read('audits/gsmhard_parent_mapping.jsonl');corrected=[];manual=[]
    for row in mappings:
        if row['hard_id'] in cfg['gsmhard_parent_overrides']:
            pid=cfg['gsmhard_parent_overrides'][row['hard_id']]
            question=hard[row['hard_id']]['question']
            matches=[key for key,value in gsm.items() if number_masked_question(value['question'])==number_masked_question(question)]
            if matches!=[pid]:raise ValueError('Manual parent is not the unique number-masked match')
            row={**row,'parent_ids':[pid],'status':'reviewed_unique_number_masked_match',
                 'question_unchanged':question==gsm[pid]['question']}
            manual.append(row)
        if len(row['parent_ids'])!=1:raise ValueError('Unresolved GSM-Hard parent')
        corrected.append(row)
    parents=Counter(row['parent_ids'][0] for row in corrected)
    if len(corrected)!=1319:raise ValueError('GSM-Hard source count changed')
    locked=[row for row in corrected if gsm[row['parent_ids'][0]]['source_index']>=50]
    out.mkdir(parents=True)
    write_jsonl(out/'math_train_eligible_for_split.jsonl',eligible)
    write_jsonl(out/'training_exclusions.jsonl',list(exclusions.values()))
    write_jsonl(out/'gold_repairs.jsonl',repairs)
    write_jsonl(out/'gsmhard_parent_mapping.jsonl',corrected)
    write_jsonl(out/'gsmhard_manual_parent_evidence.jsonl',[{**row,'hard_question':hard[row['hard_id']]['question'],
        'parent_question':gsm[row['parent_ids'][0]]['question']} for row in manual])
    write_jsonl(out/'gsmhard_locked_parent_cohort.jsonl',locked)
    summary={'status':'source_review_complete_not_frozen_for_training','source_training_questions':len(source),
        'near_match_exclusions':len(near),'duplicate_exclusions':len(duplicates),
        'eligible_before_development_split':len(eligible),'gold_repairs':len(repairs),
        'hard_parent_count':len(corrected),'hard_unique_parent_count':len(parents),
        'hard_parents_with_multiple_children':{p:n for p,n in parents.items() if n>1},
        'hard_locked_parent_questions':len(locked),'manual_parent_repairs':len(manual),
        'formal_training_ready':False,'source_reference_solutions_unchanged':True,
        'remaining':['Validate context-aware mathematical, radix, choice, and multi-answer grading.',
            'Register GSM-Hard numerical tolerance for rounded/scientific-notation reference answers.',
            'Freeze development/calibration holdouts, generation settings, and the SAE mechanism gate before main generation.']}
    save(out/'summary.json',summary)
    report=['# MATH 数据来源审查','',
        f"7,500 题来源池排除 12 个近似评测题和 1 个训练重复，剩余 {len(eligible):,} 道题供后续划分。此数尚未扣除开发集或教师正确性筛选。",
        '12 个近似题包含共享图形/题干但提问不同的题目；为控制结构性重合，均排除训练侧，评测集保持原样。','',
        '17 个参考答案的处理已逐项登记；保留原始答案、原始解答及哈希。两条空 box 根据原解答中的合数证明补为 0，其余为显式格式规范化或 parser 提取问题。',
        '43_5 必须按五进制答案处理；不能使用将其错误解析为十进制 43 的通用数学 parser。','',
        f"GSM8K-Hard 的 3 个待匹配 parent 已通过完整非数字文本唯一匹配核对。共 {len(corrected):,} 条映射、{len(parents):,} 个唯一 parent；按 GSM8K parent 的锁定 test[50:1319] 划分保留 {len(locked):,} 条。",
        '其中 hard-01079 与 parent-00993 的题目完全相同，不能把每一道 GSM8K-Hard 都描述为数字扰动后的新题。','',
        '此阶段完成来源审查，不构成正式训练冻结。多类型 grader、GSM-Hard 浮点容差、开发/校准划分及大规模生成前的 SAE 机制检查仍需完成。','']
    (out/'report_zh.md').write_text('\n'.join(report))
    provenance=out/'provenance';provenance.mkdir()
    shutil.copy2(config_path,provenance/'review_config.json')
    shutil.copy2(__file__,provenance/'baseline_data_review.py')
    shutil.copy2(CODE/'scripts/13_11_review_math_data.py',provenance)
    seal(out/'COMPLETE.json',[parent/'COMPLETE.json']+[p for p in out.rglob('*') if p.is_file()],
         stage='source_data_review',formal_claim_allowed=False)
    return summary
