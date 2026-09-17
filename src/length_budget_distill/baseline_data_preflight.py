"""Audit expanded baseline source identities, math answers, and contamination.

This is a data preflight, not a frozen training/evaluation protocol. Near matches,
unparsed reference answers, and derived-question mapping failures remain visible.
"""
from __future__ import annotations
import ast
from collections import defaultdict, Counter
from pathlib import Path
import logging
import re
import unicodedata

from .factorial import canonical_sha256, file_sha256
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import resolve, save, seal
from .experiment_io import read_json
from .math_grading import extract_boxed, parse_math_answer
from .verifiers import extract_final_answer


def normalize_question(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def shingles(text, n=5):
    tokens = re.findall(r'\\[a-z]+|[a-z]+|\d+(?:\.\d+)?|[^\w\s]', normalize_question(text))
    if len(tokens)<n: return {tuple(tokens)}
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


def near_matches(queries, references, *, n, threshold, exclude_identical=False):
    """Exact Jaccard on lexical n-grams using an inverted candidate index.

    Retain operators and numbers. A high score flags a review candidate; it is
    not automatically proof that two numerical variants are the same problem.
    """
    index=defaultdict(list);ref_sets=[]
    for i,row in enumerate(references):
        values=shingles(row['question'],n);ref_sets.append(values)
        for value in values:index[value].append(i)
    result=[]
    for query in queries:
        values=shingles(query['question'],n)
        counts=Counter(i for value in values for i in index.get(value,()))
        for i,intersection in counts.items():
            ref=references[i]
            if query['problem_id']==ref['problem_id']:continue
            score=intersection/(len(values)+len(ref_sets[i])-intersection)
            if score>=threshold:
                if exclude_identical and normalize_question(query['question'])==normalize_question(ref['question']):continue
                result.append({'query_id':query['problem_id'],'reference_id':ref['problem_id'],
                    'jaccard':score,'query_question':query['question'],'reference_question':ref['question']})
    return result


def load_source(spec):
    path=Path(spec['path'])
    if path.suffix=='.jsonl':return list(read_jsonl(path))
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix=='.arrow':
        from datasets import Dataset
        return list(Dataset.from_file(str(path)))
    raise ValueError(f'Unsupported source format: {path}')


def source_record(name,index,row,spec):
    if name=='gsm8k':
        question=row['question'];answer=extract_final_answer(row['answer']);reference=row['answer']
    elif name=='gsm8k_hard':
        question=row['input'];answer=str(row['target']);reference=None
    elif name=='aqua_rat':
        question=row['question']+'\n'+'\n'.join(row['options']);answer=row['correct'];reference=row['rationale']
    elif name=='olympiadbench':
        question=row['question'];answer=row['final_answer'];reference=row['solution']
        if (row['modality'],row['language'],row['subject'],row['question_type']) != ('Text-only','English','Math','Open-ended'):
            raise ValueError('OlympiadBench source is outside registered preflight scope')
        if any(row.get(f'image_{i}') for i in range(1,10)):
            raise ValueError('Image payload in text-only source')
    else:
        question=row['problem'];reference=row['solution'];answer=row.get('answer') or extract_boxed(reference)
    result={'problem_id':f'{name}-{index:05d}','source_index':index,'dataset':name,
        'question':question,'answer':answer,'reference_solution':reference,
        'source_repo':spec['repo_id'],'source_revision':spec['revision']}
    if name=='math500':result['source_unique_id']=row['unique_id']
    if name=='olympiadbench':
        result.update(source_unique_id=row['id'],answer_type=row['answer_type'],
                      is_multiple_answer=row['is_multiple_answer'],error_tolerance=row['error'])
    return result


def code_question_docstring(code):
    """Extract a GSM-Hard parent description using AST; never execute dataset code."""
    try:
        tree=ast.parse(code)
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef))
        return ast.get_docstring(function)
    except (SyntaxError,StopIteration):return None


def preflight(config_path):
    cfg=read_json(config_path);root=resolve(cfg['result_root'])
    if root.exists():raise FileExistsError(root)
    tables={};raw={};bindings=[Path(__file__),Path(__file__).with_name('math_grading.py')]
    for name,spec in cfg['sources'].items():
        rows=load_source(spec)
        if len(rows)!=spec['expected_count']:raise ValueError(f'Source count changed: {name}: {len(rows)}')
        raw[name]=rows
        tables[name]=[source_record(name,i,row,spec) for i,row in enumerate(rows)]
        bindings.append(Path(spec['path']))
        logging.info('Loaded %s rows=%d revision=%s',name,len(rows),spec['revision'])
    official=defaultdict(list)
    for row in tables['gsm8k']:official[normalize_question(row['question'])].append(row['problem_id'])
    mapping=[]
    for source,row in zip(raw['gsm8k_hard'],tables['gsm8k_hard']):
        description=code_question_docstring(source['code'])
        candidates=official.get(normalize_question(description),[]) if description else []
        record={'hard_id':row['problem_id'],'parent_ids':candidates,
                'status':'unique_exact_docstring_match' if len(candidates)==1 else 'needs_review'}
        row['parent_problem_ids']=candidates;mapping.append(record)
    evals=[r for name,rows in tables.items() if name!='math_train' for r in rows]
    eval_index=defaultdict(list)
    for row in evals:eval_index[normalize_question(row['question'])].append(row['problem_id'])
    seen={};exact=[];within=[]
    for row in tables['math_train']:
        key=normalize_question(row['question'])
        if key in eval_index:exact.append({'train_id':row['problem_id'],'evaluation_ids':eval_index[key]})
        if key in seen:within.append({'duplicate_id':row['problem_id'],'retained_id':seen[key]})
        else:seen[key]=row['problem_id']
    near=near_matches(tables['math_train'],evals,n=cfg['near_duplicate']['ngram'],
                      threshold=cfg['near_duplicate']['threshold'],exclude_identical=True)
    gold_failures=[]
    for i,row in enumerate(tables['math_train']):
        try:
            parsed=parse_math_answer(row['answer'],timeout_seconds=cfg['grader_timeout_seconds']) if row['answer'] is not None else []
            row['gold_parse_status']='parsed' if parsed else 'missing_or_unparsed'
            row['gold_parsed']=[str(x) for x in parsed]
        except Exception as error:
            row['gold_parse_status']='error';row['gold_parse_error']=type(error).__name__+': '+str(error)
        if row['gold_parse_status']!='parsed':gold_failures.append(row)
        if i%500==0:logging.info('Parsed MATH gold %d/%d',i,len(tables['math_train']))
    for name,rows in tables.items():write_jsonl(root/'sources'/f'{name}.jsonl',rows)
    write_jsonl(root/'audits/gsmhard_parent_mapping.jsonl',mapping)
    write_jsonl(root/'audits/exact_train_eval_matches.jsonl',exact)
    write_jsonl(root/'audits/exact_training_duplicates.jsonl',within)
    write_jsonl(root/'audits/near_train_eval_review.jsonl',near)
    write_jsonl(root/'audits/unparsed_training_gold.jsonl',gold_failures)
    summary={'status':'preflight_complete_not_frozen_for_training','source_counts':{k:len(v) for k,v in tables.items()},
        'math_training_source_questions':len(tables['math_train']),
        'exact_train_eval_matches':len(exact),'exact_training_duplicates':len(within),
        'near_train_eval_review_pairs':len(near),'unparsed_training_gold':len(gold_failures),
        'gsmhard_parent_mapping':dict(Counter(r['status'] for r in mapping)),
        'formal_training_ready':False,'remaining':['Review near matches and gold parsing failures.',
            'Resolve ambiguous/missing GSM-Hard parent links.',
            'Validate AQuA choice grading and OlympiadBench multi-answer/tolerance grading.',
            'Freeze development/calibration holdouts, prompts, caps, and actual common support before main SFT.'],
        'olympiad_scope':'674 English text-only open-ended math questions; theorem proving is a separate task.',
        'gsm8k_cohort_policy':'Full test used for decontamination; retain test[:50] smoke and test[50:1319] formal evaluation.'}
    save(root/'summary.json',summary);save(root/'config_snapshot.json',cfg)
    artifacts=[p for p in root.rglob('*') if p.is_file()]
    seal(root/'COMPLETE.json',[*bindings,*artifacts],stage='data_preflight',formal_claim_allowed=False)
    logging.info('Preflight completed: %s',summary)
