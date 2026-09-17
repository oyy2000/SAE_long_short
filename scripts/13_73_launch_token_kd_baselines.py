#!/usr/bin/env python3
"""Freeze the matched KD/SFT baseline matrix and submit checks before release."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    source=(ROOT/args.config).resolve();cfg=json.loads(source.read_text())
    for key in ('result_root','checkpoint_root','source_pilot_root'):cfg[key]=str((ROOT/cfg[key]).resolve())
    root=Path(cfg['result_root']);parent=Path(cfg['source_pilot_root'])/'execution_v7'
    if root.exists() or Path(cfg['checkpoint_root']).exists():raise FileExistsError('Preserve previous experiment attempts')
    verify(parent/'protocol/FROZEN.json');verify(parent/'protocol/SOURCES.json')
    user=subprocess.check_output(['id','-un'],text=True).strip()
    if user!='youyang7':raise ValueError('Unexpected job owner')
    commands={'associations':['sacctmgr','-nP','show','assoc','user='+user,'format=Account,Partition,QOS'],
        'qos':['sacctmgr','-nP','show','qos','gpu,short_gpu,short','format=Name,MaxWall,GrpTRES,MaxTRESPU'],
        'inventory':['sinfo','-N','-p','gpu,gpu_partners','-o','%N %P %G %t']}
    observations={k:subprocess.check_output(cmd,text=True,timeout=30) for k,cmd in commands.items()}
    route=cfg['runtime'].get('training_route','h200')
    registered={'h200':'gpu:h200:4','l40s':'gpu:l40s:'}
    if route not in registered or 'short_gpu|02:00:00' not in observations['qos'] or registered[route] not in observations['inventory']:
        raise ValueError('Registered GPU route changed')
    root.mkdir(parents=True)
    save(root/'protocol/scheduler_preflight.json',{'user':user,'observations':observations})
    shutil.copytree(parent/'code',root/'code',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    added=['src/length_budget_distill/token_kd.py','src/length_budget_distill/token_kd_pilot.py',
        'src/length_budget_distill/token_kd_baselines.py','tests/test_token_kd.py','tests/test_token_kd_baselines.py',
        'scripts/13_72_run_token_kd_baselines.py','scripts/13_73_launch_token_kd_baselines.py',str(source.relative_to(ROOT))]
    for rel in added:shutil.copy2(ROOT/rel,root/'code'/rel)
    cfg['code_root']=str(root/'code');save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    native=[Path(p) for p in json.loads((parent/'protocol/FROZEN.json').read_text())['hashes'] if '/envs/sft/lib/' in p]
    seal(root/'protocol/FROZEN.json',[source,root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
        root/'protocol/scheduler_preflight.json',parent/'protocol/FROZEN.json',*native],formal_claim_allowed=False)
    # Import in a fresh process from the frozen code, so submission paths are immutable.
    code="from length_budget_distill.token_kd_baselines import submit; from length_budget_distill.experiment_io import read_json; from length_budget_distill.ncsu_reproduction import save; from pathlib import Path; import sys; cfg=read_json(sys.argv[1]); a=submit(cfg,'tests','test'); b=submit(cfg,'prepare','prepare',parents=[a]); c=submit(cfg,'smoke','smoke',parents=[b],gpu=True,method=cfg['smoke_method']); d=submit(cfg,'launch_matrix','launch-matrix',parents=[c]); save(Path(cfg['result_root'])/'protocol/bootstrap_jobs.json',dict(tests=a,prepare=b,smoke=c,gate=d)); print(dict(tests=a,prepare=b,smoke=c,gate=d))"
    import os
    env={**os.environ,'PYTHONPATH':str(root/'code/src')}
    subprocess.run([sys.executable,'-c',code,str(root/'protocol/frozen_config.json')],check=True,env=env)
