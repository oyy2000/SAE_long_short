#!/usr/bin/env python3
"""Submit one audited baseline GPU stage using a frozen NCSU configuration."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['asc-calibrate','asc-smoke','asc-train','asc-smoke-eval','asc-eval','dap','tokenskip'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--route',choices=['h100','h200','l40s'],required=True)
    parser.add_argument('--dependency',type=int)
    args=parser.parse_args()
    path=Path(args.config).resolve();cfg=json.loads(path.read_text());rt=cfg['runtime']
    root=Path(cfg['result_root']);code=Path(cfg['code_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if args.route in ('h200','l40s'): partition,qos,wall='gpu_partners','short_gpu','02:00:00'
    else: partition,qos,wall=rt['gpu_partition'],rt['gpu_qos'],rt['gpu_time']
    logs=root/'jobs';logs.mkdir(exist_ok=True)
    cmd=['sbatch','--parsable','--account='+rt['gpu_account'],'--partition='+partition,'--qos='+qos,
         '--gres=gpu:'+args.route+':1','--nodes=1','--ntasks=1','--cpus-per-task='+str(rt['cpus']),
         '--mem='+rt['memory'],'--time='+wall,'--job-name=p13_'+args.stage,
         '--output='+str(logs/('%j_'+args.stage+'.log'))]
    if args.dependency:cmd+=['--dependency=afterok:'+str(args.dependency)]
    cmd += [str(code/'scripts/slurm/13_0_ncsu_baselines.sh'),str(code),str(path),rt['python'],
            rt['overlay'],args.stage,rt['auxiliary_cache_root']]
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    jid=result.stdout.strip().split(';')[0]
    record={'job_id':jid,'stage':args.stage,'route':args.route,'command':cmd,
            'submitted_utc':datetime.now(timezone.utc).isoformat(),'status':'submitted_not_completed'}
    with (logs/(jid+'_submission.json')).open('x') as handle:json.dump(record,handle,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
