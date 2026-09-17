"""Reuse the Phase 5 fixed-support screen with a newly trained answer-free SAE."""
from pathlib import Path
import json
import subprocess
import os

import numpy as np

from .factorial import canonical_sha256,file_sha256,read_key_value_marker
from .ncsu_reproduction import save,seal,verify,isolated_gpu_preflight
from .records import read_jsonl,write_jsonl


def freeze_screen(cfg):
    root=Path(cfg['parent_result_root']);screen=Path(cfg['result_root']);source=Path(cfg['source_root'])/'sae'
    verify(root/'training/COMPLETE.json')
    checkpoint=Path(cfg['checkpoint_path'])
    if not checkpoint.is_file():raise ValueError('New SAE checkpoint missing')
    rows=[];width=cfg['clean_tokens']['count'];early=cfg['clean_tokens']['early_count']
    if width!=64 or early!=32:raise ValueError('Changed registered Phase 5 window')
    for row in read_jsonl(root/'inputs/eligible_traces.jsonl'):
        if row['question_split'] not in ('dev','test'):continue
        positions=row['eligible_positions'][:width]
        if len(positions)!=width:raise ValueError('Insufficient fixed support')
        ids=[row['token_ids'][p] for p in positions]
        tok=cfg['teacher_tokenizer']
        # Nuisance coefficients are fitted in the analysis stage on dev only.
        from tokenizers import Tokenizer
        if not hasattr(freeze_screen,'_tokenizer'):
            freeze_screen._tokenizer=Tokenizer.from_file(tok)
        pieces=[freeze_screen._tokenizer.decode([i]) for i in ids]
        nuisance=[float(np.mean([any(c.isdigit() for c in p) for p in pieces])),
                  float(np.mean(positions))/cfg['clean_tokens']['maximum_native_position'],
                  (positions[-1]-positions[0])/cfg['clean_tokens']['maximum_native_position']]
        rows.append({k:row[k] for k in ('corpus_index','trace_id','problem_id','question_split','analysis_length_label')} |
                    {'positions':positions,'token_ids':ids,'nuisance':nuisance})
    cohorts={split:{'traces':sum(r['question_split']==split for r in rows),
                    'questions':len({r['problem_id'] for r in rows if r['question_split']==split})}
             for split in ('dev','test')}
    if any(cohorts[s]['questions']<cfg['minimum_questions_per_split'] for s in cohorts):
        raise ValueError('Insufficient fixed-support question coverage')
    protocol=screen/'protocol';write_jsonl(protocol/'eligible_traces.jsonl',rows)
    from .sae_clean_features import load_clean_protocol
    score={k:cfg[k] for k in ('clean_tokens','feature_gate','scoring')}
    score.update({'experiment_name':'phase13_answer_free_sae_feature_screen_v1',
        'eligible_traces_path':str(protocol/'eligible_traces.jsonl'),
        'eligible_traces_sha256':file_sha256(protocol/'eligible_traces.jsonl'),
        'eligible_trace_count':len(rows),'cohort_audit':cohorts,
        'parent_sae':{'k':cfg['k'],'feature_count':cfg['feature_count'],'layer_index':cfg['layer_index']},
        'teacher':{'hidden_size':cfg['hidden_size']},
        'parent_evidence':{'checkpoint_path':str(checkpoint),'checkpoint_sha256':file_sha256(checkpoint)},
        'old_selected_features':{'short_feature_ids':[]},
        'outputs':{'result_root':str(screen),'figure_root':str(screen/'figures')},
        'claim_boundary':'Exploratory observed NCSU dev/test; not independent confirmation.'})
    score['activation_shards']=[]
    parent_cfg=json.loads((source/'protocol/frozen_protocol.json').read_text())
    for shard in range(parent_cfg['activation_extraction']['trajectory_shards']):
        directory=source/'activations'/f'shard_{shard:02d}_of_04';mp=directory/'activation_manifest.json'
        marker=read_key_value_marker(directory/'ACTIVATIONS_COMPLETE')
        if marker.get('manifest_sha256')!=file_sha256(mp):raise ValueError('Parent activation shard changed')
        manifest=json.loads(mp.read_text());layer=next(x for x in manifest['layers'] if x['layer_index']==cfg['layer_index'])
        score['activation_shards'].append({'manifest_path':str(mp),'manifest_sha256':file_sha256(mp),'chunks':layer['chunks']})
    save(protocol/'frozen_protocol.json',score)
    (protocol/'PROTOCOL_FROZEN').write_text(f'status=frozen\nconfig_hash={canonical_sha256(score)}\nconfig_sha256={file_sha256(protocol/"frozen_protocol.json")}\n')
    load_clean_protocol(protocol/'frozen_protocol.json')
    seal(protocol/'INPUTS_COMPLETE.json',[screen/'protocol/LAUNCH_FROZEN.json',root/'training/COMPLETE.json',
         protocol/'eligible_traces.jsonl',protocol/'frozen_protocol.json',protocol/'PROTOCOL_FROZEN',checkpoint],
         formal_claim_allowed=False)


def dispatch(cfg,stage,shard=None):
    screen=Path(cfg['result_root']);root=Path(cfg['parent_result_root'])
    verify(screen/'protocol/LAUNCH_FROZEN.json')
    if stage=='freeze':freeze_screen(cfg)
    elif stage=='score':
        verify(screen/'protocol/INPUTS_COMPLETE.json')
        if shard is None or not 0<=shard<cfg['scoring']['shards']:raise ValueError('Invalid scoring shard')
        isolated_gpu_preflight(str(screen/'protocol/launch_config.json'),screen/'preflight'/os.environ['SLURM_JOB_ID'],expected_name='L40S')
        from .sae_clean_features import score_clean_shard
        score_clean_shard(screen/'protocol/frozen_protocol.json',Path(cfg['project_root']),shard)
    elif stage=='analyze':
        verify(screen/'protocol/INPUTS_COMPLETE.json')
        from .sae_clean_analysis import analyze_clean_features
        analyze_clean_features(screen/'protocol/frozen_protocol.json',Path(cfg['project_root']))
        seal(screen/'analysis/COMPLETE.json',[screen/'protocol/INPUTS_COMPLETE.json',
             screen/'feature_gate/feature_gate_manifest.json',screen/'feature_gate/selected_features.json'],
             formal_claim_allowed=False)
    else:raise ValueError(stage)


def submit(cfg,key,stage,*,parent=(),gpu=False,shard=None):
    screen=Path(cfg['result_root']);out=screen/'launch'/key;out.mkdir(parents=True,exist_ok=False)
    route=cfg['runtime']['routes']['l40s' if gpu else 'cpu']
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],'--qos='+route['qos'],
         '--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],
         '--time='+route['time'],'--job-name=sae_body_'+key,'--output='+str(out/'%j.log')]
    if gpu:cmd.append('--gres='+route['gres'])
    if parent:cmd+=['--dependency=afterok:'+':'.join(map(str,parent)),'--kill-on-invalid-dep=yes']
    cmd += [str(screen/'code/scripts/slurm/13_6_run_frozen_python.sh'),str(screen/'code'),
            str(screen/'protocol/launch_config.json'),cfg['runtime']['python'],cfg['runtime']['overlay'],
            '13_70_run_answer_free_screen.py','--stage',stage]
    if shard is not None:cmd+=['--shard',str(shard)]
    save(out/'intent.json',{'command':cmd,'parents':list(parent)})
    job=int(subprocess.run(cmd,check=True,text=True,capture_output=True).stdout.strip().split(';')[0])
    save(out/'submission.json',{'job_id':job,'command':cmd})
    return job
