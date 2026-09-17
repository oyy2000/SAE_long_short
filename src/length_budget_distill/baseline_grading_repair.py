"""Version a calibration-grader correction without regenerating or erasing traces."""
from __future__ import annotations
from pathlib import Path
import importlib.metadata
import shutil

from .experiment_io import read_json
from .records import read_jsonl,write_jsonl
from .ncsu_reproduction import resolve,save,seal,verify
from .baseline_reproduction import grade_prediction,summarize_generations

CODE=Path(__file__).resolve().parents[2]


def repair(config_path):
    repair_cfg=read_json(config_path)
    parent=resolve(repair_cfg['parent_root']);root=resolve(repair_cfg['result_root'])
    if root.exists():raise FileExistsError(root)
    verify(parent/'protocol/FROZEN.json');verify(parent/'asc_calibration/COMPLETE.json')
    cfg=read_json(parent/'protocol/frozen_config.json')
    cfg.update(experiment_name=repair_cfg['experiment_name'],result_root=str(root),
               grading=repair_cfg['grading'],code_root=str(root/'code'),
               grading_repair_parent=str(parent),grading_repair_reason=repair_cfg['reason'])
    cfg['runtime']['overlay']=repair_cfg['runtime_overlay']
    cfg['dap']['prompt_config']=str(root/'inputs/dap_prompt.json')
    shutil.copytree(parent/'inputs',root/'inputs')
    pool=list(read_jsonl(root/'inputs/pairs.jsonl'));source_by_id={r['problem_id']:r for r in pool}
    attempts=[];accepted={};changes=[]
    for old in read_jsonl(parent/'asc_calibration/attempts.jsonl'):
        source=source_by_id[old['problem_id']]
        grade=grade_prediction(cfg,old['solution'],old['gold_answer'])
        eligible=(grade['is_correct'] and not old['hit_max_new_tokens']
            and old['solution_token_count']>source['concise_tokens']
            and old['prompt_tokens']+old['generated_tokens']+1<=cfg['asc']['max_sequence_length'])
        row={**old,**grade,'eligible_pair':eligible,'legacy_is_correct':old['is_correct'],
             'legacy_predicted_answer':old['predicted_answer']}
        attempts.append(row)
        if bool(grade['is_correct'])!=bool(old['is_correct']):changes.append(row)
        if eligible and old['problem_id'] not in accepted:
            accepted[old['problem_id']]={**source,'verbose':old['solution'],
                'verbose_tokens':old['solution_token_count'],
                'source_trace_id':f"target-{old['problem_id']}-{old['candidate_index']}",
                'verbose_teacher':cfg['teacher']['model_name'],'verbose_teacher_revision':cfg['teacher']['revision']}
    pairs=[accepted[r['problem_id']] for r in pool if r['problem_id'] in accepted][:cfg['asc']['pairs']]
    if len(pairs)!=cfg['asc']['pairs']:raise ValueError('Regrading did not recover required unique pairs')
    out=root/'asc_calibration'
    write_jsonl(out/'attempts.jsonl',attempts);write_jsonl(out/'pairs.jsonl',pairs)
    write_jsonl(out/'grading_changes.jsonl',changes)
    summary={**summarize_generations(attempts),'accepted_pairs':len(pairs),
        'available_unique_correct_pairs':len(accepted),'grading_changed_records':len(changes),
        'new_generation_calls':0,'legacy_accuracy':sum(r['legacy_is_correct'] for r in attempts)/len(attempts),
        'selection':'first eligible saved candidate per question in immutable pool order',
        'grade_method':cfg['grading'],'parent_attempts_marker':str(parent/'asc_calibration/COMPLETE.json')}
    save(out/'summary.json',summary)
    for name in ('src','scripts','configs'):
        shutil.copytree(CODE/name,root/'code'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    sources=[p for name in ('src','scripts','configs') for p in (root/'code'/name).rglob('*') if p.is_file() and not p.is_symlink()]
    seal(root/'protocol/SOURCES.json',sources)
    save(root/'protocol/frozen_config.json',cfg)
    save(root/'protocol/grading_versions.json',{n:importlib.metadata.version(n) for n in ('math-verify','latex2sympy2_extended','sympy')})
    seal(root/'protocol/FROZEN.json',[root/'protocol/SOURCES.json',root/'protocol/frozen_config.json',
        root/'protocol/grading_versions.json',*sorted((root/'inputs').glob('*')),
        parent/'protocol/FROZEN.json',parent/'asc_calibration/COMPLETE.json'],formal_claim_allowed=False)
    seal(out/'COMPLETE.json',[out/'attempts.jsonl',out/'pairs.jsonl',out/'grading_changes.jsonl',out/'summary.json',
        root/'protocol/FROZEN.json',parent/'asc_calibration/COMPLETE.json'],stage='corrected_calibration',formal_claim_allowed=False)
    return summary
