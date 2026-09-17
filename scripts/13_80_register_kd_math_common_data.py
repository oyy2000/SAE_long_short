#!/usr/bin/env python3
"""Register common MATH student data preparation after text shard DAG exists."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save,seal,verify

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());launch=Path(cfg['launch_root']);verify(launch/'FROZEN.json');verify(launch/'SOURCES.json')
    if ROOT!=Path(cfg['controller_code_root']):raise ValueError('Use frozen common-data controller')
    text_controller=Path(cfg['text_controller_root']);verify(text_controller/'SUBMISSION_COMPLETE.json')
    text_job=json.loads((text_controller/'jobs.json').read_text())['text_merge_job']
    out=launch/'registered';out.mkdir(parents=True,exist_ok=False)
    launcher=ROOT/'scripts/slurm/13_6_run_frozen_python.sh';route=cfg['runtime']['routes']['cpu']
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],
        '--qos='+route['qos'],'--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],
        '--time='+route['time'],'--job-name=kd_math_common_data','--output='+str(out/'%j.log'),
        '--dependency=afterok:'+str(text_job),'--kill-on-invalid-dep=yes',str(launcher),
        str(ROOT),str(launch/'frozen_config.json'),cfg['runtime']['python'],cfg['runtime']['overlay'],
        '13_35_distill_unified_students.py','--stage','prepare']
    save(out/'intent.json',{'command':cmd,'text_merge_job':text_job})
    job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);save(out/'submission.json',{'job_id':job,'command':cmd})
    spool=out/'submitted_batch.sh';subprocess.run(['scontrol','write','batch_script',str(job),str(spool)],check=True,capture_output=True)
    if spool.read_bytes()!=launcher.read_bytes():raise ValueError('Submitted batch changed')
    seal(out/'SUBMISSION_VERIFIED.json',[out/'intent.json',out/'submission.json',spool,launcher,text_controller/'SUBMISSION_COMPLETE.json'],training_release=False)
    print(json.dumps({'common_data_job':job,'after_text_merge':text_job}))
