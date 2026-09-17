"""MATH development-only decoding budget checks with exact saved prefixes."""
from pathlib import Path
import json
import logging
import os
import shutil
import subprocess

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save, seal, verify, admission, teacher_bundle, evidence
from .baseline_reproduction import generate_text
from .baseline_method_analysis import audit_cohort
from .typed_math_grading import grade_typed_response

CODE = Path(__file__).resolve().parents[2]


def cap_view(token_ids, cap, eos_token_id):
    if cap <= 0: raise ValueError('Cap must be positive')
    prefix = token_ids[:cap]
    ended = eos_token_id in prefix
    body = prefix[:prefix.index(eos_token_id)] if ended else prefix
    return {'body_token_ids':body, 'sampled_token_ids':prefix, 'generated_tokens':len(body),
            'hit_max_new_tokens':not ended and len(prefix) == cap}


def merge_extended_traces(original, replacements):
    """Replace exactly the capped rows, preserving source and sampled prefix."""
    expected={r['problem_id']:r for r in original if r['hit_max_new_tokens']}
    observed={r['problem_id']:r for r in replacements}
    if len({r['problem_id'] for r in original})!=len(original) or len(observed)!=len(replacements) or set(observed)!=set(expected):
        raise ValueError('Missing, duplicate, or unexpected cap-extension rows')
    for pid,row in observed.items():
        before=expected[pid]
        for key in ('question','answer','seed','prompt_tokens','model_role'):
            if row[key]!=before[key]:raise ValueError('Cap extension changed source identity: '+key)
        if row['sampled_token_ids'][:len(before['sampled_token_ids'])]!=before['sampled_token_ids']:
            raise ValueError('Longer generation does not preserve the original sampled prefix')
    return [observed.get(r['problem_id'],r) for r in original]


def summarize_caps(rows, caps):
    result={}
    for cap in caps:
        values=[r for r in rows if r['cap']==cap];n=len(values)
        if not n:raise ValueError('Missing cap predictions')
        result[str(cap)]={'n':n,'correct':sum(r['is_correct'] for r in values),
            'accuracy':sum(r['is_correct'] for r in values)/n,
            'mean_generated_tokens':sum(r['generated_tokens'] for r in values)/n,
            'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in values)/n}
    return result


def prepare(config_path):
    cfg = read_json(config_path); project = Path(cfg['project_root']); root = project/cfg['result_root']
    parent = project/cfg['cohort_root']
    if root.exists(): raise FileExistsError(root)
    verify(parent/'COMPLETE.json')
    questions = list(read_jsonl(parent/'cohorts/development.jsonl'))
    selected = sorted(questions, key=lambda r:canonical_sha256([cfg['subset_seed'], r['problem_id']]))[:cfg['development_questions']]
    if len(selected) != cfg['development_questions'] or any(r['question_role'] != 'development' for r in selected):
        raise ValueError('Invalid cap development subset')
    write_jsonl(root/'inputs/questions.jsonl', selected)
    # This metadata is for diagnostics; selection does not use model outcomes.
    hashes = {}
    for name, spec in cfg['models'].items():
        snapshot = Path(spec['snapshot_path'])
        paths = sorted(snapshot.glob('*.safetensors'))+sorted(snapshot.glob('*.json'))
        if not any(p.suffix == '.safetensors' for p in paths): raise ValueError('Missing model weights: '+name)
        hashes[name] = evidence(paths)
    save(root/'inputs/model_hashes.json', hashes)
    cfg.update(result_root=str(root), cohort_root=str(parent), code_root=str(root/'code'))
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(CODE/folder, root/'code'/folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'), symlinks=True)
    seal(root/'protocol/SOURCES.json', [p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    save(root/'protocol/frozen_config.json', cfg)
    seal(root/'protocol/FROZEN.json', [Path(config_path), root/'protocol/SOURCES.json', root/'protocol/frozen_config.json',
         parent/'COMPLETE.json', root/'inputs/questions.jsonl', root/'inputs/model_hashes.json'],
         formal_training_ready=False, stage='math_cap_development_input_freeze')


def load(config_path):
    cfg = read_json(config_path); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    if CODE != Path(cfg['code_root']): raise ValueError('Use frozen cap-calibration source')
    return cfg


def run(cfg, name):
    import torch
    if name not in cfg['models']: raise ValueError('Unknown model role')
    root = Path(cfg['result_root']); out = root/name; out.mkdir(parents=True, exist_ok=False)
    spec = cfg['models'][name]; admission(cfg)
    inventory = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv'],
                               check=True,text=True,capture_output=True)
    save(out/'hardware.json', {'job_id':os.environ['SLURM_JOB_ID'],'inventory_csv':inventory.stdout,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))})
    model, tok = teacher_bundle({**cfg,'teacher':spec}); torch.cuda.reset_peak_memory_stats()
    if model.generation_config.forced_eos_token_id is not None:
        raise ValueError('Offline cap prefixes cannot simulate forced-EOS decoding')
    questions = list(read_jsonl(root/'inputs/questions.jsonl'))
    grading = read_json(Path(cfg['cohort_root'])/'grading_config.json'); rows = []; full = []; direct_checks = []
    generation = {**spec['generation'],'max_new_tokens':max(cfg['caps']),'retain_token_ids':True}
    with (out/'full_traces.jsonl').open('x') as traces, (out/'cap_predictions.jsonl').open('x') as predictions:
        for index, question in enumerate(questions):
            prompt = cfg['question_template'].format(question=question['question'])
            seed = int(canonical_sha256([cfg['generation_seed'],name,question['problem_id']])[:8],16)
            result = generate_text(model,tok,[{'role':'user','content':prompt}],generation,seed)
            original = {**question,'model_role':name,**result}; full.append(original)
            traces.write(json.dumps(original)+'\n'); traces.flush()
            for cap in cfg['caps']:
                view = cap_view(result['sampled_token_ids'],cap,tok.eos_token_id)
                text = tok.decode(view['body_token_ids'],skip_special_tokens=True).strip()
                row = {'problem_id':question['problem_id'],'question':question['question'],'gold_answer':question['answer'],
                    'model_role':name,'cap':cap,'solution':text,**view,
                    **grade_typed_response(text,question,grading),'source_full_token_sha256':canonical_sha256(result['sampled_token_ids']),
                    'measurement':'Exact saved-token prefix; no measured latency for this simulated cap.'}
                rows.append(row);predictions.write(json.dumps(row)+'\n')
            predictions.flush()
            if index < cfg['direct_prefix_checks']:
                cap = cfg['direct_check_cap']; actual = generate_text(model,tok,[{'role':'user','content':prompt}],{**generation,'max_new_tokens':cap},seed)
                expected = result['sampled_token_ids'][:cap]
                if actual['sampled_token_ids'] != expected: raise ValueError('Direct cap decoding differs from saved prefix')
                direct_checks.append({'problem_id':question['problem_id'],'cap':cap,'tokens':len(expected),'identical':True,
                    'latency_seconds':actual['latency_seconds'],'actually_truncated_longer_trace':len(result['sampled_token_ids'])>cap})
            logging.info('Cap calibration %s %d/%d %s tokens=%d capped=%s',name,index+1,len(questions),question['problem_id'],result['generated_tokens'],result['hit_max_new_tokens'])
    audit_cohort(full,[r['problem_id'] for r in questions])
    for cap in cfg['caps']: audit_cohort([r for r in rows if r['cap']==cap],[q['problem_id'] for q in questions])
    save(out/'direct_prefix_checks.json',{'checks':direct_checks})
    by_cap = summarize_caps(rows,cfg['caps'])
    largest = by_cap[str(max(cfg['caps']))]
    acceptable = [cap for cap in cfg['caps'] if by_cap[str(cap)]['cap_hit_rate']<=cfg['proposal_maximum_cap_hit_rate']
        and by_cap[str(cap)]['accuracy']>=largest['accuracy']-cfg['proposal_maximum_accuracy_drop']]
    save(out/'summary.json',{'model_role':name,'questions':len(questions),'by_cap':by_cap,'proposed_cap':min(acceptable) if acceptable else None,
        'proposal_only_not_final_training_protocol':True,'full_trace_generation_seconds':sum(r['latency_seconds'] for r in full),
        'direct_check_generation_seconds':sum(r['latency_seconds'] for r in direct_checks),
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'scope':'Small development subset; no statistical guarantee on tail probability, no steered-teacher/adapted-student cap guarantee.'})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*sorted(out.glob('*'))],stage='math_development_cap_calibration',formal_training_ready=False)


def extend(config_path):
    """Extend only actually capped development traces; count all repeated work."""
    import ast
    import importlib.metadata
    import torch
    from .baseline_reproduction import record_hardware
    cfg=read_json(config_path);parent=Path(cfg['parent_root']);out=Path(cfg['result_root'])
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen cap-extension source')
    if out.exists():raise FileExistsError(out)
    markers=[Path(cfg['launch_root'])/'FROZEN.json',parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json',
             parent/cfg['model_role']/'COMPLETE.json']
    for marker in markers:verify(marker)
    pcfg=read_json(parent/'protocol/frozen_config.json');spec=pcfg['models'][cfg['model_role']]
    # The changed file also contains unrelated baseline stages; bind the actual
    # generation function separately to the implementation used for the old rows.
    def function_source(path):
        text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='generate_text')
        return ast.get_source_segment(text,node)
    previous=function_source(parent/'code/src/length_budget_distill/baseline_reproduction.py')
    current=function_source(CODE/'src/length_budget_distill/baseline_reproduction.py')
    if previous!=current:raise ValueError('Extension decoder function differs from the original run')
    original=list(read_jsonl(parent/cfg['model_role']/'full_traces.jsonl'))
    pending=[r for r in original if r['hit_max_new_tokens']]
    if len(pending)!=cfg['expected_capped_questions']:raise ValueError('Unexpected cap-extension support')
    out.mkdir(parents=True);admission(cfg);record_hardware(out);model,tok=teacher_bundle({**cfg,'teacher':spec})
    if model.generation_config.forced_eos_token_id is not None:raise ValueError('Forced EOS prevents exact cap extension')
    torch.cuda.reset_peak_memory_stats();generation={**spec['generation'],'max_new_tokens':max(cfg['caps']),'retain_token_ids':True}
    replacements=[]
    with (out/'replacement_full_traces.jsonl').open('x') as handle:
        for source in pending:
            prompt=pcfg['question_template'].format(question=source['question'])
            result=generate_text(model,tok,[{'role':'user','content':prompt}],generation,source['seed'])
            row={**source,**result};replacements.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
            logging.info('Extended cap %s tokens=%d cap_hit=%s',source['problem_id'],row['generated_tokens'],row['hit_max_new_tokens'])
    merged=merge_extended_traces(original,replacements);grading=read_json(Path(pcfg['cohort_root'])/'grading_config.json')
    rows=[]
    for source in merged:
        for cap in cfg['caps']:
            view=cap_view(source['sampled_token_ids'],cap,tok.eos_token_id)
            text=tok.decode(view['body_token_ids'],skip_special_tokens=True).strip()
            rows.append({'problem_id':source['problem_id'],'question':source['question'],'gold_answer':source['answer'],
                'model_role':cfg['model_role'],'cap':cap,'solution':text,**view,**grade_typed_response(text,source,grading),
                'measurement':'Saved exact prefix; unchanged complete traces reused, originally capped traces regenerated with verified original prefixes.'})
    for cap in cfg['caps']:audit_cohort([r for r in rows if r['cap']==cap],[q['problem_id'] for q in original])
    by_cap=summarize_caps(rows,cfg['caps']);before=read_json(parent/cfg['model_role']/'summary.json')
    for cap,stats in before['by_cap'].items():
        if cap in by_cap and stats!=by_cap[cap]:raise ValueError('Extension changed an original cap summary')
    largest=by_cap[str(max(cfg['caps']))]
    eligible=[cap for cap in cfg['caps'] if by_cap[str(cap)]['cap_hit_rate']<=pcfg['proposal_maximum_cap_hit_rate']
              and by_cap[str(cap)]['accuracy']>=largest['accuracy']-pcfg['proposal_maximum_accuracy_drop']]
    summary={'questions':len(original),'regenerated_questions':len(replacements),'by_cap':by_cap,'proposed_cap':min(eligible) if eligible else None,
        'original_cap_prefixes_identical':True,'original_lower_cap_summaries_identical':True,
        'additional_generation_seconds_including_repeated_prefixes':sum(r['latency_seconds'] for r in replacements),
        'prior_full_generation_seconds':before['full_trace_generation_seconds'],'prior_direct_check_seconds':before['direct_check_generation_seconds'],
        'actual_cumulative_generation_seconds':sum(r['latency_seconds'] for r in replacements)+before['full_trace_generation_seconds']+before['direct_check_generation_seconds'],
        'additional_generated_tokens_including_repeated_prefixes':sum(r['generated_tokens'] for r in replacements),
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'generation_function_sha256':canonical_sha256(current),
        'runtime_versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers')},
        'formal_training_ready':False,'proposal_only_not_final_training_protocol':True,
        'scope':cfg['claim_boundary']}
    write_jsonl(out/'merged_full_traces.jsonl',merged);write_jsonl(out/'cap_predictions.jsonl',rows);save(out/'summary.json',summary)
    del model;torch.cuda.empty_cache()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    caps=cfg['caps'];axes[0].plot(caps,[100*by_cap[str(c)]['accuracy'] for c in caps],'o-',color='#2166AC')
    axes[1].plot(caps,[100*by_cap[str(c)]['cap_hit_rate'] for c in caps],'o-',color='#b2182b')
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(caps,[f'{c//1024}K' for c in caps]);ax.set_xlabel('Generation token cap');ax.set_ylim(0,105)
    axes[0].set_ylabel('Automatic typed-answer accuracy (%)');axes[1].set_ylabel('Cap-hit rate (%)')
    fig.suptitle('R1-7B: fixed 32 MATH development questions; adaptive cap extension')
    for suffix in ('png','pdf'):fig.savefig(out/f'cap_extension.{suffix}',dpi=180)
    plt.close(fig)
    (out/'report_zh.md').write_text('# R1 MATH 长度上限扩展\n\n![长度上限](cap_extension.png)\n\n'
        f"原 32 题中仅 {len(replacements)} 个触及 8K 上限的输出重新生成。原 token 前缀及原四档统计均一致；全部额外重复生成计入成本。规则建议 cap：{summary['proposed_cap']}。\n\n"+cfg['claim_boundary']+'\n')
    seal(out/'COMPLETE.json',[Path(config_path),*markers,*sorted(out.glob('*'))],stage='math_cap_extension',formal_training_ready=False)
