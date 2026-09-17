#!/usr/bin/env python3
"""Submit one registered TokenSkip reproduction stage to NCSU Slurm."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke-train','smoke-eval','train','evaluate'])
    p.add_argument('--config',required=True);p.add_argument('--route',choices=['cpu','h100','h200','l40s'],required=True)
    p.add_argument('--model',choices=['base','author','replica'],default='replica');p.add_argument('--dependency',type=int)
    a=p.parse_args();path=Path(a.config).resolve();cfg=json.loads(path.read_text());rt=cfg['runtime']
    if (a.stage=='prepare')!=(a.route=='cpu'):raise ValueError('Preparation needs CPU; other stages need a GPU')
    code=ROOT if a.stage=='prepare' else Path(cfg['code_root'])
    if a.stage!='prepare':
        verify(Path(cfg['result_root'])/'protocol/FROZEN.json');verify(Path(cfg['result_root'])/'protocol/SOURCES.json')
    route=rt['routes'][a.route]
    logs=Path(cfg['launch_root']);logs=logs if logs.is_absolute() else ROOT/logs;logs.mkdir(parents=True,exist_ok=True)
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],
        '--qos='+route['qos'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),
        '--mem='+route['memory'],'--time='+route['time'],'--job-name=tk_'+a.stage+'_'+a.model,
        '--output='+str(logs/('%j_'+a.stage+'_'+a.model+'.log'))]
    if a.route!='cpu':cmd.append('--gres='+route['gres'])
    if a.dependency:cmd.append('--dependency=afterok:'+str(a.dependency))
    cmd.extend([str(code/'scripts/slurm/13_2_tokenskip_reproduction.sh'),str(code),str(path),
                rt['python'],rt['overlay'],a.stage,'--model',a.model])
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr)
    job=result.stdout.strip().split(';')[0]
    record={'job_id':job,'stage':a.stage,'model':a.model,'command':cmd,
            'submitted_utc':datetime.now(timezone.utc).isoformat(),'completed':False}
    with (logs/(job+'_submission.json')).open('x') as handle:json.dump(record,handle,indent=2)
    print(json.dumps({'job_id':job,'stage':a.stage,'model':a.model,'route':a.route,'log_root':str(logs)}))
