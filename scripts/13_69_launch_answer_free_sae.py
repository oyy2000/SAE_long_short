#!/usr/bin/env python3
"""Freeze and submit the answer-free SAE preparation, sampling, and training DAG."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();source=(ROOT/args.config).resolve();cfg=json.loads(source.read_text())
    root=Path(cfg['result_root']);checkpoints=Path(cfg['checkpoint_root'])
    if root.exists() or checkpoints.exists():raise FileExistsError('Preserve existing answer-free SAE attempts')
    parent=Path(cfg['source_root'])/'sae'
    verify(Path(cfg['source_root'])/'protocol/FROZEN.json')
    user=subprocess.check_output(['id','-un'],text=True).strip()
    if user!='youyang7':raise ValueError('Unexpected Slurm owner')
    commands={'associations':['sacctmgr','-nP','show','assoc','user='+user,'format=Account,Partition,QOS'],
              'qos':['sacctmgr','-nP','show','qos','gpu,short_gpu,short','format=Name,MaxWall,GrpTRES,MaxTRESPU'],
              'inventory':['sinfo','-N','-p','gpu_partners','-o','%N %P %G %t']}
    observations={k:subprocess.check_output(v,text=True,timeout=30) for k,v in commands.items()}
    if 'gpu:h200:4' not in observations['inventory'] or 'short_gpu|02:00:00' not in observations['qos']:
        raise ValueError('Requested H200 route not verified')
    root.mkdir(parents=True)
    save(root/'protocol/scheduler_preflight.json',{'user':user,'observations':observations})
    for folder in ('src','scripts','configs','tests'):
        shutil.copytree(ROOT/folder,root/'code'/folder,symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    cfg['code_root']=str(root/'code')
    save(root/'protocol/frozen_config.json',cfg)
    seal(root/'protocol/SOURCES.json',[p for p in (root/'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(root/'protocol/FROZEN.json',[source,root/'protocol/frozen_config.json',root/'protocol/SOURCES.json',
         root/'protocol/scheduler_preflight.json',parent/'protocol/frozen_protocol.json',
         parent/'corpus/CORPUS_COMPLETE',parent/'corpus/corpus_manifest.json',
         Path(cfg['source_root'])/'protocol/FROZEN.json'],formal_claim_allowed=False)
    def submit(stage,*,parent_job=None,gpu=False):
        out=root/'launch'/stage;out.mkdir(parents=True,exist_ok=False)
        route=cfg['runtime']['routes']['gpu' if gpu else 'cpu']
        cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],
             '--qos='+route['qos'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),
             '--mem='+route['memory'],'--time='+route['time'],'--job-name=sae_body_'+stage,
             '--output='+str(out/'%j.log')]
        if gpu:cmd.append('--gres='+route['gres'])
        if parent_job:cmd.extend(['--dependency=afterok:'+str(parent_job),'--kill-on-invalid-dep=yes'])
        cmd.extend([str(root/'code/scripts/slurm/13_6_run_frozen_python.sh'),str(root/'code'),
                    str(root/'protocol/frozen_config.json'),cfg['runtime']['python'],cfg['runtime']['overlay'],
                    '13_68_run_answer_free_sae.py','--stage',stage])
        save(out/'intent.json',{'command':cmd,'parent_job':parent_job})
        reply=subprocess.run(cmd,check=True,capture_output=True,text=True)
        job=int(reply.stdout.strip().split(';')[0]);save(out/'submission.json',{'job_id':job,'command':cmd})
        return job
    prepare_job=submit('prepare');sample_job=submit('sample',parent_job=prepare_job)
    train_job=submit('train',parent_job=sample_job,gpu=True)
    save(root/'protocol/submitted_jobs.json',{'prepare':prepare_job,'sample':sample_job,'train':train_job,
         'feature_screen_submitted':False,'intervention_submitted':False})
    print(json.dumps({'prepare':prepare_job,'sample':sample_job,'train':train_job}))


if __name__=='__main__':main()
