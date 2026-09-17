#!/usr/bin/env python3
"""Freeze and submit CPU checks, data preparation, GPU smoke, then gated small KD."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify
from length_budget_distill.token_kd_pilot import submit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    source=(ROOT/args.config).resolve();cfg=json.loads(source.read_text())
    for key in ('result_root','checkpoint_root','source_pilot_root'):cfg[key]=str((ROOT/cfg[key]).resolve())
    root=Path(cfg['result_root']);parent=Path(cfg['source_pilot_root'])/'execution_v7'
    if root.exists() or Path(cfg['checkpoint_root']).exists():raise FileExistsError('Preserve existing KD pilot attempts')
    verify(parent/'protocol/FROZEN.json');verify(parent/'protocol/SOURCES.json')
    root.mkdir(parents=True);user=subprocess.check_output(['id','-un'],text=True).strip()
    commands={'associations':['sacctmgr','-nP','show','assoc','user='+user,'format=Account,Partition,QOS'],
        'qos':['sacctmgr','-nP','show','qos','gpu,short_gpu,short','format=Name,MaxWall,GrpTRES,MaxTRESPU'],
        'inventory':['sinfo','-N','-p','gpu,gpu_partners','-o','%N %P %G %t']}
    observations={k:subprocess.check_output(cmd,text=True,timeout=30) for k,cmd in commands.items()}
    save(root/'protocol/scheduler_preflight.json',{'user':user,'observations':observations})
    shutil.copytree(parent/'code',root/'code',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    added=['src/length_budget_distill/token_kd.py','src/length_budget_distill/token_kd_pilot.py','tests/test_token_kd.py',
        'scripts/13_63_run_token_kd_pilot.py','scripts/13_64_launch_token_kd_pilot.py',str(source.relative_to(ROOT))]
    for rel in added:shutil.copy2(ROOT/rel,root/'code'/rel)
    cfg['code_root']=str(root/'code');save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    native=[Path(p) for p in json.loads((parent/'protocol/FROZEN.json').read_text())['hashes'] if '/envs/sft/lib/' in p]
    seal(root/'protocol/FROZEN.json',[source,root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',root/'protocol/scheduler_preflight.json',parent/'protocol/FROZEN.json',*native])
    tests=submit(cfg,'tests','test');prep=submit(cfg,'prepare','prepare',parents=[tests])
    smoke=submit(cfg,'smoke','smoke',parents=[prep],gpu=True)
    gate=submit(cfg,'launch_pilot','launch-pilot',parents=[smoke])
    save(root/'protocol/bootstrap_jobs.json',{'tests':tests,'prepare':prep,'smoke':smoke,'gate':gate,'full_scale_submitted':False})
    print(json.dumps({'tests':tests,'prepare':prep,'smoke':smoke,'gate':gate}))
