#!/usr/bin/env python3
"""Freeze a pilot bootstrap and submit CPU tests, input audit, then gated execution."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.gsm8k_student_pilot import absolute_config
from length_budget_distill.experiment_io import read_json
from length_budget_distill.ncsu_reproduction import save,seal

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--launch-root',required=True)
    args=parser.parse_args();cfg=absolute_config(read_json(args.config),ROOT);launch=(ROOT/args.launch_root).resolve()
    if launch.exists() or Path(cfg['result_root']).exists():raise FileExistsError('Preserve prior pilot/launch attempts')
    user=subprocess.check_output(['id','-un'],text=True).strip()
    observations={}
    commands={'associations':['sacctmgr','-nP','show','assoc','user='+user,'format=Account,Partition,QOS'],
              'qos':['sacctmgr','-nP','show','qos','gpu,short_gpu,normal,short','format=Name,MaxWall,GrpTRES,MaxTRESPU'],
              'inventory':['sinfo','-N','-p','gpu,gpu_partners','-o','%N %P %G %t'],
              'cpu_inventory':['sinfo','-p','compute,compute_partners','-o','%P %a %l %D %t']}
    for key,cmd in commands.items():observations[key]=subprocess.run(cmd,check=True,text=True,capture_output=True,timeout=30).stdout
    if 'jekml_cpu' not in observations['associations'] or 'jekml_gpu' not in observations['associations']:
        raise ValueError('Required account association missing')
    launch.mkdir(parents=True);save(launch/'scheduler_preflight.json',{'user':user,**observations})
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(ROOT/folder,launch/'code'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'),symlinks=True)
    cfg['code_root']=str(launch/'code');save(launch/'frozen_config.json',cfg)
    seal(launch/'SOURCES.json',[p for p in (launch/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(launch/'FROZEN.json',[launch/'frozen_config.json',launch/'SOURCES.json',launch/'scheduler_preflight.json'])
    route=cfg['runtime']['routes']['cpu'];parent=None;ids={}
    for key,stage in [('test','test'),('prepare','prepare'),('begin','advance')]:
        out=launch/key;out.mkdir();code=Path(cfg['result_root'])/'code' if key=='begin' else launch/'code'
        config=Path(cfg['result_root'])/'protocol/frozen_config.json' if key=='begin' else launch/'frozen_config.json'
        cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],'--qos='+route['qos'],
             '--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],'--time='+route['time'],
             '--job-name=gsm_pilot_'+key,'--output='+str(out/'%j.log')]
        if parent:cmd+=['--dependency=afterok:'+str(parent)]
        cmd += [str(launch/'code/scripts/slurm/13_6_run_frozen_python.sh'),str(code),str(config),cfg['runtime']['python'],
                cfg['runtime']['overlay'],'13_60_run_gsm8k_student_pilot.py','--stage',stage]
        if key=='begin':cmd+=['--group','calibration']
        save(out/'intent.json',{'command':cmd,'key':key,'parent':parent})
        response=subprocess.run(cmd,check=True,text=True,capture_output=True)
        parent=int(response.stdout.strip().split(';')[0]);ids[key]=parent
        save(out/'submission.json',{'key':key,'job_id':parent,'command':cmd,'stdout':response.stdout,'experiment_complete':False})
    save(launch/'summary.json',{'jobs':ids,'student_training_complete':False,'scope':'Tests, independent inputs and dependency-gated pilot execution'})
    seal(launch/'SUBMISSION_COMPLETE.json',[launch/'FROZEN.json',launch/'summary.json',*sorted(launch.glob('*/submission.json'))],experiment_complete=False)
    print(json.dumps(ids))
