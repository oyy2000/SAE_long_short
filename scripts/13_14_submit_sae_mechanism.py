#!/usr/bin/env python3
"""Submit one frozen SAE mechanism input stage using authorized NCSU routes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','generate','merge'])
    p.add_argument('--config',required=True);p.add_argument('--route',choices=['cpu','h100','h200','l40s'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--dependency',type=int,nargs='+')
    a=p.parse_args();path=Path(a.config).resolve();cfg=json.loads(path.read_text());project=Path(cfg['project_root'])
    if (a.stage in ('prepare','merge'))!=(a.route=='cpu'):raise ValueError('Preparation/merge use CPU; generation uses GPU')
    logs=project/cfg['launch_root'];logs.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':
        launch=logs/('prepare_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));code=launch/'code'
        for folder in ('src','scripts','configs','tests'):
            shutil.copytree(ROOT/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
        path=launch/'frozen_config.json';save(path,cfg)
        seal(launch/'FROZEN.json',[path]+[f for f in code.rglob('*') if f.is_file() and not f.is_symlink()])
    else:
        code=Path(cfg['code_root']);root=Path(cfg['result_root'])
        verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
        if a.stage=='generate':verify(root/'baseline/smoke/COMPLETE.json')
    rt=cfg['runtime'];route=rt['routes'][a.route]
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],'--qos='+route['qos'],
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],'--time='+route['time'],
        '--job-name=p13_sae_'+a.stage+'_'+str(a.shard),'--output='+str(logs/('%j_'+a.stage+'_'+str(a.shard)+'.log'))]
    if a.route!='cpu':cmd+=['--gres='+route['gres']]
    if a.dependency:cmd+=['--dependency=afterok:'+':'.join(map(str,a.dependency))]
    cmd+=[str(code/'scripts/slurm/13_5_sae_mechanism_inputs.sh'),str(code),str(path),rt['python'],rt['overlay'],a.stage,'--shard',str(a.shard)]
    result=subprocess.run(cmd,text=True,capture_output=True,check=True);job=result.stdout.strip().split(';')[0]
    record={'job_id':job,'stage':a.stage,'shard':a.shard,'route':a.route,'command':cmd,
            'submitted_utc':datetime.now(timezone.utc).isoformat(),'completed':False}
    save(logs/(job+'_submission.json'),record);print(json.dumps(record))
