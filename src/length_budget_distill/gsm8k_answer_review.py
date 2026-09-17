"""Conservative automatic extraction with explicit adjudication of prose answers.

No gold-aware numeric selection is allowed. Ambiguous multi-quantity prose and
unmarked conclusions are reviewed from a method/label-masked queue. Finalization
refuses missing, duplicate, stale, or unused decisions; old grades stay intact.
"""
from collections import Counter,defaultdict
from pathlib import Path
from fractions import Fraction
import json
import logging
import re
import shutil

from .experiment_io import read_json
from .records import read_jsonl,write_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save,seal,verify
from .math_grading import extract_boxed
from .gsm8k_grading_v3 import HEADING,QUANTITY,extract_explicit_quantity,grade_gsm8k_response

CODE=Path(__file__).resolve().parents[2]
VERSION='gsm8k_explicit_quantity_with_masked_prose_review_v4'


def inspect_answer(text):
    boxed=extract_boxed(text)
    if boxed is not None:return {'status':'automatic','answer':boxed,'rule':'final_box'}
    headings=list(HEADING.finditer(text))
    if not headings:
        if not text.strip():return {'status':'invalid','answer':None,'rule':'empty_response'}
        return {'status':'needs_review','answer':None,'rule':'unmarked_conclusion','answer_section':text[-1200:]}
    suffix=text[headings[-1].end():].lstrip()
    if not suffix:return {'status':'invalid','answer':None,'rule':'empty_answer_line'}
    line=suffix.splitlines()[0].replace('−','-').replace(r'\$','$')
    line=re.sub(r'\\[,!;:]','',line).replace('{,}',',')
    first,rule=extract_explicit_quantity(text)
    if rule in ('explicit_numeric_alternatives','non_scalar_answer_expression'):
        return {'status':'needs_review','answer':None,'rule':rule,'answer_section':line}
    matches=list(QUANTITY.finditer(line))
    if not matches:
        return {'status':'needs_review','answer':None,'rule':'nonnumeric_answer_sentence','answer_section':line}
    try:
        values={Fraction(m.group().replace(',','').replace(' ','')) for m in matches}
    except (ValueError,ZeroDivisionError):
        values=set()
    if len(values)==1 and first is not None and not re.search(r'(?i)\b(?:not|unknown|undefined|impossible|either)\b',line):
        return {'status':'automatic','answer':first,'rule':'single_distinct_quantity_in_answer_sentence'}
    return {'status':'needs_review','answer':None,'rule':'prose_with_multiple_quantities',
            'answer_section':line,'candidate_quantities':[m.group() for m in matches]}


def case_key(question,text):
    return canonical_sha256([VERSION,question,text])


def validate_decisions(cases,decisions):
    expected={r['case_id'] for r in cases};lookup={}
    for row in decisions:
        key=row['case_id']
        if key in lookup:raise ValueError('Duplicate review decision')
        if key not in expected:raise ValueError('Unknown or stale review decision')
        if row['decision'] not in ('answer','invalid'):raise ValueError('Unsupported review decision')
        if row['decision']=='answer' and (not isinstance(row.get('answer'),str) or not row['answer'].strip()):
            raise ValueError('Reviewed answer must be an explicit nonempty string')
        if row['decision']=='invalid' and row.get('answer') is not None:raise ValueError('Invalid response cannot have an answer')
        if not row.get('reason'):raise ValueError('Every review decision needs a rationale')
        lookup[key]=row
    if set(lookup)!=expected:raise ValueError('Missing review decisions')
    return lookup


def score_answer(answer,gold,*,reviewed,timeout_seconds):
    """Invalid automatic output is wrong; a malformed reviewed scalar is an error."""
    if answer is None:
        return {'is_correct':False,'predicted_answer':None,'status':'invalid_final_answer'}
    grade=grade_gsm8k_response(r'\boxed{'+answer+'}',gold,timeout_seconds=timeout_seconds)
    if grade['status']=='unparsed_gold' or (reviewed and grade['status']!='graded'):
        raise ValueError('Reference or adjudicated answer failed grading: '+str(grade))
    return grade


def load_review_index(root):
    """Load sealed adjudications for reuse by analyses of the original outputs."""
    root=Path(root);verify(root/'COMPLETE.json')
    summary=read_json(root/'summary.json')
    if summary['grader_version']!=VERSION or summary['unresolved_cases']!=0:
        raise ValueError('Uniform answer review is incomplete or has a different version')
    index={}
    for row in read_jsonl(root/'graded_predictions.jsonl'):
        key=(row['group'],row['problem_id'])
        if key in index:raise ValueError('Duplicate reviewed group/problem')
        if row['grader_version']!=VERSION:raise ValueError('Mixed review versions')
        index[key]=row
    if len(index)!=summary['records']:raise ValueError('Reviewed record coverage changed')
    return index


def apply_reviewed_grade(original,reviewed,*,text_field,gold_field,tokens_field):
    """Join only an identical prediction; never transfer scores by question alone."""
    expected={'problem_id':original['problem_id'],'prediction_text':original[text_field],
        'gold_answer':str(original[gold_field]),'output_tokens':original[tokens_field],
        'hit_max_new_tokens':original['hit_max_new_tokens']}
    if original.get('question') is not None:expected['question']=original['question']
    if any(reviewed[k]!=value for k,value in expected.items()):
        raise ValueError('Reviewed grade does not match original prediction, reference or token accounting')
    if reviewed['case_id']!=case_key(reviewed['question'],reviewed['prediction_text']):
        raise ValueError('Stale reviewed case identity')
    if reviewed['grader_version']!=VERSION:raise ValueError('Unexpected reviewed grader version')
    return {**original,'is_correct':reviewed['is_correct'],'predicted_answer':reviewed['predicted_answer'],
        'status':reviewed['status'],'grader_version':VERSION,
        'legacy_is_correct':original['is_correct'],'legacy_predicted_answer':original.get('predicted_answer'),
        'review_case_id':reviewed['case_id'],'review_decision':reviewed['review_decision']}


def truncated_without_final_answer(record):
    """Recognize an exhausted budget without any final-answer declaration.

    This is a format-failure rule. It does not infer an answer from unfinished
    reasoning, and never applies to a box, an Answer heading or an 'answer is'
    declaration, even if that response subsequently hits its token cap.
    """
    return bool(record['hit_max_new_tokens'] and record['inspection']['rule']=='unmarked_conclusion'
        and not re.search(r'(?i)\b(?:final\s+)?answer\s*(?:is|=|:)',record['prediction_text']))


def triage(prepared_root,out_root):
    prepared=Path(prepared_root);out=Path(out_root);verify(prepared/'COMPLETE.json')
    flags=defaultdict(list)
    for row in read_jsonl(prepared/'records_with_labels.jsonl'):
        if row['inspection']['status']=='needs_review':flags[row['case_id']].append(truncated_without_final_answer(row))
    automatic=[];manual=[]
    for case in read_jsonl(prepared/'review_cases_masked.jsonl'):
        if all(flags[case['case_id']]):
            automatic.append({'case_id':case['case_id'],'decision':'invalid','answer':None,
                'decision_method':'automatic_missing_final_answer_at_cap',
                'reason':'Reached the token cap without a recognized final-answer declaration; the output is scored as missing its final answer, without inferring one from unfinished reasoning.'})
        else:manual.append(case)
    out.mkdir(parents=True,exist_ok=False)
    write_jsonl(out/'automatic_decisions.jsonl',automatic);write_jsonl(out/'remaining_cases_masked.jsonl',manual)
    save(out/'summary.json',{'automatic_format_failures':len(automatic),'remaining_manual_cases':len(manual),
        'all_original_cases_retained':True,'policy_uses_gold_or_method':False,
        'rule':'All records sharing a case must hit their cap, lack a box/Answer heading, and lack an answer-is/answer-equals declaration.'})
    shutil.copy2(__file__,out/'triage_source.py')
    seal(out/'COMPLETE.json',[prepared/'COMPLETE.json']+[p for p in out.rglob('*') if p.is_file()],
         stage='uniform_answer_review_format_triage',manual_review_complete=not manual)


def load_config(config_path):
    cfg=read_json(config_path)
    if CODE!=Path(cfg['code_root']):raise ValueError('Use the frozen review snapshot')
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    return cfg


def prepare(config_path):
    cfg=load_config(config_path);root=Path(cfg['result_root']);out=root/'prepared'
    if out.exists():raise FileExistsError(out)
    records=[];cases={};bindings=[];group_counts=Counter();rules=Counter()
    for spec in cfg['sources']:
        for marker in spec['markers']:
            verify(marker);bindings.append(Path(marker))
        questions={}
        if spec.get('questions_file'):
            questions={r['problem_id']:r['question'] for r in read_jsonl(spec['questions_file'])};bindings.append(Path(spec['questions_file']))
        rows=list(read_jsonl(spec['path']))
        if len(rows)!=spec['expected_rows']:raise ValueError('Source row count changed: '+spec['name'])
        if spec.get('expected_unique_questions') and len({r['problem_id'] for r in rows})!=spec['expected_unique_questions']:
            raise ValueError('Source question support changed: '+spec['name'])
        bindings.append(Path(spec['path']))
        for index,row in enumerate(rows):
            text=row[spec['text_field']];question=row.get('question') or questions[row['problem_id']]
            inspection=inspect_answer(text);key=case_key(question,text)
            if spec.get('cell_field'):
                value=row
                for field in spec['cell_field'].split('.'):value=value[field]
                group=spec['name']+'/'+str(value)
            else:group=spec['name']
            original_group=group
            group=cfg.get('group_labels',{}).get(group,group)
            group_counts[group]+=1;rules[inspection['rule']]+=1
            records.append({'source_name':spec['name'],'group':group,'source_row':index,'problem_id':row['problem_id'],
                'question':question,'prediction_text':text,'gold_answer':str(row[spec['gold_field']]),
                'legacy_is_correct':row['is_correct'],'legacy_predicted_answer':row.get('predicted_answer'),
                'output_tokens':row[spec['tokens_field']], 'hit_max_new_tokens':row['hit_max_new_tokens'],
                'case_id':key,'inspection':inspection,'original_group':original_group})
            if inspection['status']=='needs_review' and key not in cases:
                # No source/model/method identifier, reference answer or old score.
                cases[key]={'case_id':key,'question':question,'prediction_text':text,
                            'answer_section':inspection['answer_section'],'reason_for_review':inspection['rule']}
        logging.info('Prepared answer extraction audit %s records=%d',spec['name'],len(rows))
    if len(records)!=cfg['expected_total_records']:raise ValueError('Incomplete aggregate source manifest')
    out.mkdir(parents=True)
    write_jsonl(out/'records_with_labels.jsonl',records)
    write_jsonl(out/'review_cases_masked.jsonl',[cases[k] for k in sorted(cases)])
    save(out/'summary.json',{'records':len(records),'unique_review_cases':len(cases),
        'records_needing_review':sum(r['inspection']['status']=='needs_review' for r in records),
        'extraction_rules':dict(rules),'group_counts':dict(group_counts),'grading_complete':False,
        'masking_scope':'Queue omits method/model/source names, gold answers and old scores. Review is by the project assistant, not an independent blinded human annotation; some earlier examples were already observed.'})
    seal(out/'COMPLETE.json',bindings+[Path(config_path),Path(cfg['launch_root'])/'FROZEN.json']+
         [p for p in out.rglob('*') if p.is_file()],stage='uniform_answer_review_preparation',grading_complete=False)


def finalize(config_path,decisions_path):
    cfg=load_config(config_path);root=Path(cfg['result_root']);prepared=root/'prepared';out=root/'finalized'
    verify(prepared/'COMPLETE.json')
    review_bindings=[]
    if cfg.get('review_provenance_marker'):
        verify(cfg['review_provenance_marker']);review_bindings.append(Path(cfg['review_provenance_marker']))
    cases=list(read_jsonl(prepared/'review_cases_masked.jsonl'))
    decisions=validate_decisions(cases,list(read_jsonl(decisions_path)))
    decision_methods=Counter(r.get('decision_method','unspecified') for r in decisions.values())
    rows=[];changes=[];grouped=defaultdict(list)
    for record in read_jsonl(prepared/'records_with_labels.jsonl'):
        inspection=record['inspection'];answer=inspection['answer'];decision=None
        if inspection['status']=='needs_review':
            decision=decisions[record['case_id']];answer=decision['answer']
        grade=score_answer(answer,record['gold_answer'],reviewed=decision is not None,timeout_seconds=cfg['timeout_seconds'])
        row={**record,**grade,'grader_version':VERSION,'review_decision':decision}
        rows.append(row);grouped[row['group']].append(row)
        if bool(row['is_correct'])!=bool(row['legacy_is_correct']):changes.append(row)
        if len(rows)%5000==0:logging.info('Final-answer audit graded %d records',len(rows))
    if len(rows)!=cfg['expected_total_records']:raise ValueError('Finalized source coverage changed')
    summary={group:{'n':len(values),'correct':sum(r['is_correct'] for r in values),
                   'accuracy':sum(r['is_correct'] for r in values)/len(values),
                   'legacy_accuracy':sum(r['legacy_is_correct'] for r in values)/len(values),
                   'mean_output_tokens':sum(r['output_tokens'] for r in values)/len(values),
                   'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in values)/len(values),
                   'reviewed_records':sum(r['review_decision'] is not None for r in values),
                   'decision_methods':dict(Counter(r['review_decision'].get('decision_method','unspecified')
                       for r in values if r['review_decision'] is not None))} for group,values in grouped.items()}
    out.mkdir(parents=True,exist_ok=False)
    write_jsonl(out/'graded_predictions.jsonl',rows);write_jsonl(out/'grading_changes.jsonl',changes)
    save(out/'summary.json',{'status':'complete','records':len(rows),'groups':summary,'changed_scores':len(changes),
        'unique_review_decisions':len(decisions),'decision_methods':dict(decision_methods),
        'unresolved_cases':0,'grader_version':VERSION,
        'claim_boundary':cfg['claim_boundary'],'formal_claim_allowed':False})
    report=['# Uniform final-answer audit','',
        f'{len(rows):,} predictions; {len(decisions)} adjudication decisions; {len(changes)} correctness labels changed. Original text and scores remain unchanged.',
        f"Decision methods: {dict(decision_methods)}. Automatic missing-final-answer decisions are format rules, not manually validated semantic failures.",'',
        '| Group | N | Accuracy | Previous accuracy | Mean tokens | Cap hit | Reviewed |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group,m in summary.items():report.append(f"| {group} | {m['n']} | {100*m['accuracy']:.2f}% | {100*m['legacy_accuracy']:.2f}% | {m['mean_output_tokens']:.2f} | {100*m['cap_hit_rate']:.2f}% | {m['reviewed_records']} |")
    report+=['','The review queue masks labels and methods; the project assistant performs the review. This is a documented post-hoc measurement correction, not independent human validation or a test-set hyperparameter search.',
             '',cfg['claim_boundary'],'']
    (out/'report.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json',[prepared/'COMPLETE.json',Path(decisions_path),Path(config_path),
         Path(cfg['launch_root'])/'FROZEN.json',*review_bindings]+[p for p in out.rglob('*') if p.is_file()],
         stage='uniform_answer_review_finalization',formal_claim_allowed=False,unresolved_cases=0)
