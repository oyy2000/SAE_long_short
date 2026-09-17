"""Registered terminal-job retries; independent of the read-only Luna reviewer."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from .ncsu_reproduction import save, verify
from . import gsm8k_student_pilot as data

FAILED={'FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL','PREEMPTED'}

def now():return datetime.now(timezone.utc).isoformat()

def arguments(command):
    result={}
    for key in ('stage','round','group','shard','method','seed'):
        flag='--'+key
        if flag in command:result[key]=command[command.index(flag)+1]
    return result

def retry_paths(cfg,args):
    root=Path(cfg['result_root']);stage=args['stage'];method=args.get('method','');seed=args.get('seed','17')
    if stage=='generate':
        child=data.load_child(cfg,f"round_{args['round']}/{args['group']}")
        return [Path(child['result_root'])/'generation'/method/f"shard_{int(args['shard']):02d}"]
    if stage=='compress':
        return [root/f"round_{args['round']}/text"/args['group']/method/f"shard_{int(args['shard']):02d}"]
    if stage=='evaluate-test':return [root/'evaluation'/method/f'seed_{seed}'/f"shard_{int(args['shard']):02d}"]
    if stage in ('train-dev','evaluate-dev','train'):
        paths=[]
        if stage in ('train-dev','train') and method!='base':
            sft=data.load_child(cfg,'sft');adapter=Path(sft['checkpoint_root'])/'qwen3b_student'/method/f'seed_{seed}'
            training=root/'sft/training/qwen3b_student'/method/f'seed_{seed}'
            if (adapter/'TRAIN_COMPLETE.json').exists() and (training/'COMPLETE.json').exists():
                verify(adapter/'TRAIN_COMPLETE.json');verify(training/'COMPLETE.json')
            else:paths.extend([training,adapter])
        if stage!='train':paths.append(root/'development'/method/f'seed_{seed}')
        return paths
    raise ValueError('Stage requires primary diagnosis: '+stage)

def archive_incomplete(paths,out):
    for p in paths:
        if p.exists() and any(p.rglob('*COMPLETE.json')):
            raise ValueError('Completed artifact needs audit, not automatic replacement: '+str(p))
    records=[]
    for i,p in enumerate(paths):
        if p.exists():
            target=out/'preserved_outputs'/str(i);target.parent.mkdir(parents=True,exist_ok=True)
            p.rename(target);records.append({'original':str(p),'preserved':str(target)})
    save(out/'preserved_outputs.json',{'paths':records})

def retry_route(state, current, log):
    if any(term in log for term in ('ValueError:', 'IndexError:', 'KeyError:', 'ImportError:', 'FileNotFoundError:', 'ModuleNotFoundError:', 'SyntaxError:', 'hash mismatch', 'Hash mismatch')):
        raise ValueError('Deterministic data/configuration failure requires diagnosis')
    if state=='TIMEOUT':return 'h100'
    if state=='OUT_OF_MEMORY':return 'h200'
    if any(term in log.lower() for term in ('illegal memory access','device-side assert','cuda error','gpu health')):
        return 'h100' if current=='h200' else 'h200'
    return current


def reconcile(cfg):
    root=Path(cfg['result_root']);base=root/'recovery';base.mkdir(exist_ok=True)
    with (base/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        records=[]
        for p in (root/'launch').glob('*/submission.json'):
            if not (p.parent/'SUPERSEDED.json').exists():records.append((p,json.loads(p.read_text())))
        ids=[str(r['job_id']) for _,r in records]
        if not ids:return
        output=subprocess.check_output(['sacct','-nPX','-j',','.join(ids),'-o','JobIDRaw,State,User'],text=True,timeout=30)
        user=subprocess.check_output(['id','-un'],text=True).strip();states={}
        for line in output.splitlines():
            fields=line.split('|')
            if len(fields)>=3 and fields[2]==user:states[int(fields[0])]=fields[1].split()[0].rstrip('+')
        for p,row in records:
            job=row['job_id'];state=states.get(job)
            # Only executions using this exact recovery protocol are authorized here.
            if state not in FAILED or not any(path in row['command'] for path in cfg['recovery'].get('worker_execution_configs',[cfg['execution_config_path']])):continue
            count=row.get('retry_count',0);out=base/f'job_{job}'
            if out.exists():continue  # ambiguous submissions are never duplicated
            out.mkdir()
            try:
                if count>=cfg['recovery']['maximum_retries']:raise ValueError('Registered retry limit reached')
                log_path=p.parent/(str(job)+'.log')
                if log_path.exists():
                    with log_path.open('rb') as handle:
                        handle.seek(max(0,log_path.stat().st_size-24000));failure_log=handle.read().decode('utf-8',errors='replace')
                else:failure_log=''
                selected_route=retry_route(state,row['route'],failure_log)
                args=arguments(row['command']);paths=retry_paths(cfg,args)
                archive_incomplete(paths,out)
                retry_key=row['key']+'_retry'+str(count+1);launch=root/'launch'/retry_key;launch.mkdir()
                command=[v for v in row['command'] if not v.startswith('--dependency=')]
                command=[('--output='+str(launch/'%j.log')) if v.startswith('--output=') else ('--job-name=gsm_pilot_'+retry_key) if v.startswith('--job-name=') else v for v in command]
                # TIMEOUT fallback retains all scientific settings and changes only registered hardware/time.
                if selected_route!=row['route']:
                    route=cfg['runtime']['routes'][selected_route]
                    replacements={'account':route['account'],'partition':route['partition'],'qos':route['qos'],'gres':route['gres'],'time':route['time'],'export':'ALL,LBD_PILOT_GPU_ROUTE='+selected_route}
                    command=[next(('--'+k+'='+val for k,val in replacements.items() if v.startswith('--'+k+'=')),v) for v in command]
                intent={**row,'key':retry_key,'command':command,'parents':[],'retry_count':count+1,'retry_of':job,'retry_state':state,'route':selected_route,'created_utc':now()};intent.pop('job_id',None)
                save(launch/'intent.json',intent)
                reply=subprocess.run(command,check=True,capture_output=True,text=True,timeout=60);new=int(reply.stdout.strip().split(';')[0])
                save(launch/'submission.json',{**intent,'job_id':new,'stdout':reply.stdout,'stderr':reply.stderr})
                updates=[]
                for child_path,child in records:
                    if child['job_id']==job:continue
                    live=subprocess.check_output(['scontrol','show','job',str(child['job_id']),'-o'],text=True,stderr=subprocess.DEVNULL) if states.get(child['job_id'])=='PENDING' else ''
                    if not live:continue
                    dep=next((x[len('Dependency='):] for x in live.split() if x.startswith('Dependency=')),'')
                    if str(job)+'(' not in dep and str(job) not in dep.split(':'):continue
                    if 'afterok:' not in dep or 'UserId='+user+'(' not in live:raise ValueError('Unexpected dependent ownership or dependency type')
                    import re
                    cleaned=re.sub(r'\([^)]*\)','',dep)
                    updated=re.sub(r'(?<=:)'+str(job)+r'(?=[:,]|$)',str(new),cleaned)
                    subprocess.run(['scontrol','update','JobId='+str(child['job_id']),'Dependency='+updated],check=True)
                    updates.append({'job_id':child['job_id'],'before':dep,'after':updated})
                save(out/'recovery.json',{'old_job':job,'new_job':new,'updates':updates,'utc':now()})
                save(p.parent/'SUPERSEDED.json',{'old_job':job,'replacement_job':new,'recovery':str(out/'recovery.json')})
            except Exception as error:
                save(out/'NEEDS_ATTENTION.json',{'job_id':job,'error':repr(error),'utc':now()})
        (base/'heartbeat.json').write_text(json.dumps({'utc':now(),'job_id':os.environ.get('SLURM_JOB_ID'),'states':states})+'\n')


def run(config_path,cycle):
    cfg=json.loads(Path(config_path).read_text());root=Path(cfg['result_root']);out=root/'recovery';out.mkdir(exist_ok=True)
    verify(cfg['execution_marker'])
    from .gsm8k_pilot_monitor import review
    for _ in range(55):
        if (out/'STOP').exists() or (root/'EXPERIMENT_COMPLETE.json').exists():return
        try:reconcile(cfg)
        except Exception as error:
            (out/'controller_error.json').write_text(json.dumps({'error':repr(error),'utc':now()})+'\n')
        marker=out/'last_luna.json';last=json.loads(marker.read_text()).get('unix',0) if marker.exists() else 0
        if cfg['recovery'].get('run_luna_locally',True) and time.time()-last>=3600:
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            try:
                status,_=review(root,Path(cfg['recovery']['bootstrap_launch_root']),out/'luna'/stamp,'gpt-5.6-luna','medium')
                marker.write_text(json.dumps({'unix':time.time(),'status':status,'review':str(out/'luna'/stamp/'review.md')})+'\n')
            except Exception as error:marker.write_text(json.dumps({'unix':time.time(),'error':repr(error)})+'\n')
        time.sleep(60)
    if cycle>=cfg['recovery']['maximum_controller_cycles']:
        save(out/'CONTROLLER_LIMIT.json',{'cycle':cycle,'utc':now()});return
    from .gsm8k_pilot_runtime import submit
    submit(cfg,cfg['recovery'].get('controller_prefix','recovery_controller')+'_'+str(cycle+1),'recover',route='monitor_cpu',shard=cycle+1,parents=[int(os.environ['SLURM_JOB_ID'])])
