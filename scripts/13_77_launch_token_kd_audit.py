#!/usr/bin/env python3
"""Freeze independent result verification and register it after the dynamic matrix."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify


def submit(cfg,stage,parent):
    root=Path(cfg['launch_root']);out=root/'submissions'/stage;out.mkdir(parents=True,exist_ok=False)
    route=cfg['runtime']['routes']['cpu'];launcher=Path(cfg['code_root'])/'scripts/slurm/13_6_run_frozen_python.sh'
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],
        '--qos='+route['qos'],'--cpus-per-task=2','--mem=16G','--time=02:00:00',
        '--job-name=kd_audit_'+stage,'--output='+str(out/'%j.log'),'--dependency=afterok:'+str(parent),
        '--kill-on-invalid-dep=yes',str(launcher),cfg['code_root'],str(root/'frozen_config.json'),
        cfg['runtime']['python'],cfg['runtime']['overlay'],
        '13_77_launch_token_kd_audit.py' if stage=='register' else '13_76_audit_token_kd_baselines.py']
    if stage=='register':cmd+=['--stage','register']
    save(out/'intent.json',{'command':cmd})
    job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0])
    save(out/'submission.json',{'job_id':job,'command':cmd})
    spool=out/'submitted_batch.sh';subprocess.run(['scontrol','write','batch_script',str(job),str(spool)],check=True,capture_output=True)
    if spool.read_bytes()!=launcher.read_bytes():raise ValueError('Actual audit batch changed')
    seal(out/'SUBMISSION_VERIFIED.json',[out/'intent.json',out/'submission.json',spool,launcher])
    return job


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['freeze','register'],default='freeze');a=p.parse_args()
    source=Path(a.config).resolve();cfg=json.loads(source.read_text());root=Path(cfg['launch_root'])
    experiment=json.loads(Path(cfg['execution_config']).read_text());exp_root=Path(experiment['result_root'])
    if a.stage=='freeze':
        if root.exists():raise FileExistsError('Preserve prior verification launch')
        verify(exp_root/'protocol/FROZEN.json');verify(exp_root/'protocol/SOURCES.json')
        root.mkdir(parents=True)
        shutil.copytree(Path(experiment['code_root']),root/'code',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for rel in ['src/length_budget_distill/gsm8k_pilot_summary.py','src/length_budget_distill/token_kd_audit.py',
                    'scripts/13_76_audit_token_kd_baselines.py','scripts/13_77_launch_token_kd_audit.py']:
            shutil.copy2(ROOT/rel,root/'code'/rel)
        cfg['code_root']=str(root/'code');save(root/'frozen_config.json',cfg)
        seal(root/'SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
        seal(root/'FROZEN.json',[source,root/'frozen_config.json',root/'SOURCES.json',exp_root/'protocol/FROZEN.json'])
        parent=json.loads((exp_root/'protocol/bootstrap_jobs.json').read_text())['gate']
        job=submit(cfg,'register',parent)
    else:
        verify(root/'FROZEN.json');verify(root/'SOURCES.json')
        if ROOT!=Path(cfg['code_root']):raise ValueError('Use frozen audit registration')
        parent=json.loads((exp_root/'protocol/matrix_jobs.json').read_text())['analysis']
        job=submit(cfg,'verify',parent)
    print(json.dumps({'stage':a.stage,'job_id':job,'parent':parent}))
