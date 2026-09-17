#!/usr/bin/env python3
"""Freeze CPU-only MATH source recovery without releasing historical jobs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    source=(ROOT/a.config).resolve();cfg=json.loads(source.read_text());root=Path(cfg['launch_root'])
    if root.exists() or Path(cfg['result_root']).exists():raise FileExistsError('Preserve earlier preparation attempts')
    parent=Path(cfg['parent_root'])
    if (parent/'selection/student_pool').exists():raise FileExistsError('Original selection already exists; inspect before resuming')
    user=subprocess.check_output(['id','-un'],text=True).strip()
    queue=subprocess.check_output(['squeue','-h','-u',user,'-o','%i|%T|%j|%r'],text=True)
    # Do not race an active or eligible prior merge/impact job.
    for line in queue.splitlines():
        parts=line.split('|')
        if parts[0] in [str(j) for j in cfg['historical_jobs']] and parts[3] not in ('JobHeldAdmin','JobHeldUser'):
            raise RuntimeError('Historical source job is active or eligible: '+line)
    observations={'queue':queue,
        'associations':subprocess.check_output(['sacctmgr','-nP','show','assoc','user='+user,'format=Account,Partition,QOS'],text=True),
        'qos':subprocess.check_output(['sacctmgr','-nP','show','qos','short','format=Name,MaxWall'],text=True)}
    bindings=[parent/'protocol/FROZEN.json',parent/'protocol/SOURCES.json',Path(cfg['cohort_root'])/'COMPLETE.json']
    bindings.extend(parent/f'generation/student_pool/shard_{i:02d}/COMPLETE.json' for i in range(32))
    for marker in bindings:verify(marker)
    root.mkdir(parents=True);save(root/'scheduler_preflight.json',{'user':user,'observations':observations})
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(ROOT/folder,root/'code'/folder,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    cfg['code_root']=str(root/'code');save(root/'frozen_config.json',cfg)
    seal(root/'SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'FROZEN.json',[source,root/'frozen_config.json',root/'SOURCES.json',root/'scheduler_preflight.json',*bindings],training_release=False)
    jobs=[]
    for stage in ('merge','impact'):
        out=root/'submissions'/stage;out.mkdir(parents=True)
        route=cfg['runtime']['route']
        cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],
            '--qos='+route['qos'],'--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],
            '--time='+route['time'],'--job-name=kd_math_'+stage,'--output='+str(out/'%j.log')]
        if jobs:cmd+=['--dependency=afterok:'+str(jobs[-1]),'--kill-on-invalid-dep=yes']
        launcher=root/'code/scripts/slurm/13_6_run_frozen_python.sh'
        cmd += [str(launcher),cfg['code_root'],str(root/'frozen_config.json'),cfg['runtime']['python'],
                cfg['runtime']['overlay'],'13_74_prepare_kd_math_sources.py','--stage',stage]
        save(out/'intent.json',{'command':cmd})
        job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);jobs.append(job)
        save(out/'submission.json',{'job_id':job,'command':cmd})
        spool=out/'submitted_batch.sh';subprocess.run(['scontrol','write','batch_script',str(job),str(spool)],check=True,capture_output=True)
        if spool.read_bytes()!=launcher.read_bytes():raise ValueError('Unexpected actual batch script')
        seal(out/'SUBMISSION_VERIFIED.json',[out/'intent.json',out/'submission.json',spool,launcher])
    save(root/'submitted_jobs.json',{'merge':jobs[0],'impact':jobs[1],'compression_submitted':False,'kd_training_submitted':False})
    print(json.dumps({'merge':jobs[0],'impact':jobs[1]}))
