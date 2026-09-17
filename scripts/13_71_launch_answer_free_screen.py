#!/usr/bin/env python3
"""Freeze a conditional fixed-support screen after answer-free SAE training."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify
from length_budget_distill.sae_body_screen import submit

if __name__=='__main__':
    parent=ROOT/'results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2'
    original=json.loads((parent/'protocol/frozen_config.json').read_text())
    screen=parent/'screen'
    if screen.exists():raise FileExistsError(screen)
    job=json.loads((parent/'protocol/submitted_jobs.json').read_text())['train']
    info=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True)
    user=subprocess.check_output(['id','-un'],text=True).strip()
    if f'UserId={user}(' not in info or 'gres/gpu:h200' not in info.lower():
        raise ValueError('Training dependency does not belong to the registered owner/route')
    phase5=json.loads((ROOT/'configs/phase5_sae_clean_feature_causal_v1.json').read_text())
    teacher=json.loads((Path(original['source_root'])/'sae/protocol/frozen_protocol.json').read_text())['teacher']
    cfg={'project_root':str(ROOT),'parent_result_root':str(parent),'result_root':str(screen),
         'source_root':original['source_root'],'checkpoint_path':str(Path(original['checkpoint_root'])/'sae_model.safetensors'),
         'teacher_tokenizer':str(Path(teacher['snapshot_path'])/'tokenizer.json'),
         'layer_index':17,'k':64,'feature_count':28672,'hidden_size':teacher['hidden_size'],
         'clean_tokens':{'count':64,'early_count':32,'maximum_native_position':128},
         'feature_gate':phase5['feature_gate'],'scoring':{'shards':4,'batch_size':256,'readback_positions':[0,16,32,48]},
         'minimum_questions_per_split':20,
         'runtime':{'python':original['runtime']['python'],'overlay':original['runtime']['overlay'],
                    'native_preload':original['runtime']['native_preload'],'minimum_free_mib':28000,
                    'storage_policy':original['runtime']['storage_policy'],
                    'routes':{'cpu':original['runtime']['routes']['cpu'],
                              'l40s':json.loads((ROOT/'configs/phase13_token_kd_small_v2.json').read_text())['runtime']['routes']['l40s']}},
         'claim_boundary':'Previously observed NCSU dev/test, exploratory fixed-support rescreen; no independent confirmation or generation claim.'}
    observations={'qos':subprocess.check_output(['sacctmgr','-nP','show','qos','short_gpu,short','format=Name,MaxWall,GrpTRES,MaxTRESPU'],text=True),
                  'inventory':subprocess.check_output(['sinfo','-N','-p','gpu_partners','-o','%N %P %G %t'],text=True)}
    if 'gpu:l40s:4 mix' not in observations['inventory'] or 'short_gpu|02:00:00' not in observations['qos']:
        raise ValueError('L40S route unavailable')
    screen.mkdir(parents=True)
    save(screen/'protocol/scheduler_preflight.json',observations)
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(ROOT/folder,screen/'code'/folder,symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    cfg['code_root']=str(screen/'code');save(screen/'protocol/launch_config.json',cfg)
    seal(screen/'protocol/SOURCES.json',[p for p in (screen/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(screen/'protocol/LAUNCH_FROZEN.json',[parent/'protocol/FROZEN.json',
         screen/'protocol/launch_config.json',screen/'protocol/SOURCES.json',
         screen/'protocol/scheduler_preflight.json',ROOT/'configs/phase5_sae_clean_feature_causal_v1.json',
         ROOT/'configs/phase13_token_kd_small_v2.json'],formal_claim_allowed=False)
    freeze=submit(cfg,'freeze','freeze',parent=[job])
    save(screen/'protocol/initial_jobs.json',{'train_dependency':job,'freeze':freeze,
         'score_submitted':False,'intervention_submitted':False})
    print(json.dumps({'freeze':freeze,'train_dependency':job}))
