"""Check the typed grader against complete frozen reference-answer cohorts."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import logging
import shutil
import subprocess
import sys

from .experiment_io import read_json
from .records import read_jsonl,write_jsonl
from .ncsu_reproduction import resolve,save,seal,verify
from .typed_math_grading import compile_answer_spec,grade_typed_response

CODE=Path(__file__).resolve().parents[2]


def audit(config_path):
    cfg=read_json(config_path);out=resolve(cfg['result_root'])
    if out.exists():raise FileExistsError(out)
    parent=resolve(cfg['preflight_root']);review=resolve(cfg['review_root'])
    verify(parent/'COMPLETE.json');verify(review/'COMPLETE.json')
    grading=read_json(resolve(cfg['grading_config']))
    selected_hard={r['hard_id'] for r in read_jsonl(review/'gsmhard_locked_parent_cohort.jsonl')}
    cohorts={'math_train':list(read_jsonl(review/'math_train_eligible_for_split.jsonl'))}
    for dataset in ('math500','gsm8k','gsm8k_hard','aqua_rat','olympiadbench'):
        rows=list(read_jsonl(parent/'sources'/(dataset+'.jsonl')))
        if dataset=='gsm8k':rows=[r for r in rows if r['source_index']>=50]
        if dataset=='gsm8k_hard':rows=[r for r in rows if r['problem_id'] in selected_hard]
        cohorts[dataset]=rows
    for name,rows in cohorts.items():
        if len(rows)!=cfg['expected_counts'][name] or len({r['problem_id'] for r in rows})!=len(rows):
            raise ValueError('Reference cohort changed: '+name)
    out.mkdir(parents=True)
    command=[sys.executable,'-m','unittest','discover','-s',str(CODE/'tests'),'-p','test_typed_math_grading.py','-v']
    checked=subprocess.run(command,capture_output=True,text=True)
    (out/'regression_tests.txt').write_text(checked.stdout+checked.stderr)
    save(out/'regression_tests.json',{'command':command,'returncode':checked.returncode})
    if checked.returncode:raise RuntimeError('Grader regression checks failed')
    failures=[];diagnostics=[];summary={}
    from math_verify.errors import TimeoutException
    for dataset,rows in cohorts.items():
        counts=Counter();kinds=Counter()
        for index,row in enumerate(rows):
            raw_gold=row['answer'];gold=','.join(raw_gold) if isinstance(raw_gold,list) else raw_gold
            try:
                spec=compile_answer_spec(row,grading);kinds[spec['kind']]+=1
                result=grade_typed_response(r'\boxed{'+gold+'}',row,grading)
            except (Exception,TimeoutException) as error:
                result={'is_correct':False,'status':'spec_error','error_type':type(error).__name__,'error':str(error)}
            diagnostic={'problem_id':row['problem_id'],'dataset':dataset,**result}
            diagnostics.append(diagnostic);counts[result['status']]+=1
            if not result['is_correct']:failures.append({**row,'grader_diagnostic':result})
            if index%500==0:logging.info('Grader audit %s %d/%d',dataset,index+1,len(rows))
        failed=sum(r['dataset']==dataset for r in failures)
        summary[dataset]={'n':len(rows),'reference_self_grade_correct':len(rows)-failed,
                          'failures':failed,'status_counts':dict(counts),'answer_kinds':dict(kinds)}
    write_jsonl(out/'reference_checks.jsonl',diagnostics);write_jsonl(out/'reference_failures.jsonl',failures)
    result={'status':'reference_audit_complete','datasets':summary,'reference_failures':len(failures),
        'regression_checks_passed':True,'formal_protocol_ready':False,
        'limitation':'Reference self-comparison detects parsing gaps, not full semantic grader correctness; inspect failures and preserve independent negative/equivalence fixtures.'}
    save(out/'summary.json',result)
    save(out/'versions.json',{n:importlib.metadata.version(n) for n in
        ('math-verify','latex2sympy2_extended','antlr4-python3-runtime','sympy')})
    provenance=out/'provenance';provenance.mkdir()
    shutil.copy2(config_path,provenance/'audit_config.json')
    shutil.copy2(resolve(cfg['grading_config']),provenance/'grading_config.json')
    for path in (Path(__file__),CODE/'src/length_budget_distill/typed_math_grading.py',
                 CODE/'tests/test_typed_math_grading.py',CODE/'scripts/13_12_audit_typed_grading.py'):
        shutil.copy2(path,provenance)
    report=['# 数学评分器真实答案审计','',
        '已登记的正例、反例及超时回归检查通过，完整测试记录单独保存。以下审计将来源答案按约定放入 final box，再逐题评分，用于发现解析缺口；自比较通过不等于完整语义正确性证明。','',
        '| 数据 | 题数 | 来源答案可判为正确 | 待审查 |','| --- | ---: | ---: | ---: |']
    for name,row in summary.items():report.append(f"| {name} | {row['n']} | {row['reference_self_grade_correct']} | {row['failures']} |")
    report.extend(['','完整失败记录位于 reference_failures.jsonl；所有评测问题保持在原 cohort 中，未因解析失败删除。',
        'GSM8K-Hard 的整数参考严格精确匹配，非整数使用登记的相对容差 1e-4 / 绝对容差 1e-12；OlympiadBench 使用来源给定的绝对容差，缺省为 1e-8。',
        '多答案逐项一对一匹配并保留重复次数；元组内顺序、区间端点和进制必须一致。作者 scorer 的百分比倍数宽松接受规则不沿用，此处属于明确的评分器适配。',''])
    (out/'report_zh.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json',[parent/'COMPLETE.json',review/'COMPLETE.json']+[p for p in out.rglob('*') if p.is_file()],
         formal_claim_allowed=False,stage='typed_grader_reference_audit',reference_failures=len(failures))
    return result
