"""Submit the bounded text-compression DAG and all-eight SFT data preparation.

Scientific workers and their input audits remain the existing frozen entrypoints.
Only scheduler dependencies and launch provenance are assembled here.
"""
from pathlib import Path
import json
import os
import pwd
import subprocess
from datetime import datetime, timezone

from .experiment_io import read_json
from .ncsu_reproduction import save, seal, verify

CODE=Path(__file__).resolve().parents[2]


def nodes(shards, dap_lanes, tokenskip_lanes):
    if not 1<=dap_lanes<=shards or not 1<=tokenskip_lanes<=shards:
        raise ValueError('Invalid compression concurrency')
    result=[{'key':'dap_smoke','stage':'dap','cohort':'smoke','shard':0,'route':'h200','parents':['text_prepare']},
            {'key':'tokenskip_smoke','stage':'tokenskip','cohort':'smoke','shard':0,'route':'l40s','parents':['text_prepare']},
            {'key':'smoke_merge','stage':'merge','cohort':'smoke','shard':0,'route':'cpu','parents':['dap_smoke','tokenskip_smoke']}]
    for method,lanes,route in (('dap',dap_lanes,'h200'),('tokenskip',tokenskip_lanes,'l40s')):
        for shard in range(shards):
            result.append({'key':f'{method}_{shard:02d}','stage':method,'cohort':'student_pool','shard':shard,
                'route':route,'parents':['smoke_merge']+([f'{method}_{shard-lanes:02d}'] if shard>=lanes else [])})
    result += [{'key':'text_merge','stage':'merge','cohort':'student_pool','shard':0,'route':'cpu',
                'parents':[f'{m}_{s:02d}' for m in ('dap','tokenskip') for s in range(shards)]},
               {'key':'sft_prepare','stage':'prepare','route':'cpu','parents':['raw_merge','steered_merge','text_merge']}]
    return result


def validate_topology(graph, external):
    known=set(external)
    for node in graph:
        if node['key'] in known or not node['parents'] or not set(node['parents'])<=known:
            raise ValueError('Repeated node, missing dependency, or non-topological graph')
        if node['stage']=='train':raise ValueError('This DAG prepares data only; SFT training is a separate stage')
        known.add(node['key'])


def scheduler_record(job):
    user=pwd.getpwuid(os.getuid()).pw_name
    cmd=['sacct','-nP','-X','-j',str(job),'-o','JobIDRaw,User,State,ExitCode']
    response=subprocess.run(cmd,check=True,text=True,capture_output=True)
    rows=[line.split('|') for line in response.stdout.splitlines() if line.strip()]
    values=[r for r in rows if r[0]==str(job)]
    if len(values)!=1 or values[0][1]!=user:
        raise ValueError('Upstream scheduler identity is missing or not owned by the current user')
    row=values[0]
    if row[2] not in ('PENDING','RUNNING','COMPLETING','COMPLETED') or (row[2]=='COMPLETED' and row[3]!='0:0'):
        raise ValueError('Upstream job is not live or successfully complete: '+str(row))
    return {'job_id':int(row[0]),'user':row[1],'state':row[2],'exit_code':row[3],
            'observed_utc':datetime.now(timezone.utc).isoformat()}


def command(cfg, node, job_ids):
    sft=node['key']=='sft_prepare'
    spec=read_json(cfg['sft_config_path'] if sft else cfg['text_config_path'])
    route=spec['runtime']['routes'][node['route']]
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],'--qos='+route['qos'],
         '--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],'--time='+route['time'],
         '--job-name=p13_cont_'+node['key'],'--output='+str(Path(cfg['launch_root'])/('%j_'+node['key']+'.log')),
         '--dependency=afterok:'+':'.join(str(job_ids[k]) for k in node['parents'])]
    if route.get('gres'):cmd+=['--gres='+route['gres']]
    if route.get('nodelist'):cmd+=['--nodelist='+route['nodelist']]
    # sbatch copies the already frozen wrapper now. Text CODE/config are created
    # by the previously submitted preparation before these jobs become eligible.
    code=Path(cfg['code_root']) if sft else Path(spec['result_root'])/'code'
    config=Path(cfg['sft_config_path']) if sft else Path(spec['result_root'])/'protocol/frozen_config.json'
    cmd += [str(Path(cfg['code_root'])/'scripts/slurm/13_6_run_frozen_python.sh'),str(code),str(config),
            spec['runtime']['python'],spec['runtime']['overlay'],
            '13_35_distill_unified_students.py' if sft else '13_31_compress_math_candidates.py','--stage',node['stage']]
    if not sft:cmd+=['--cohort',node['cohort'],'--shard',str(node['shard'])]
    return cmd


def submit(config_path):
    cfg=read_json(config_path);launch=Path(cfg['launch_root'])
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen continuation source')
    verify(launch/'FROZEN.json');verify(launch/'TEST_COMPLETE.json')
    test=read_json(launch/'TEST_COMPLETE.json');test_state=scheduler_record(test['job_id'])
    if test_state['state']!='COMPLETED':raise ValueError('Continuation tests have not completed')
    out=launch/'submission';out.mkdir(parents=True,exist_ok=False)
    state={name:scheduler_record(job) for name,job in cfg['upstream_jobs'].items()}
    save(out/'upstream_states.json',state)
    text=read_json(cfg['text_config_path']);sft=read_json(cfg['sft_config_path'])
    if text['source_cohort']!='student_pool' or text['compression_mode']!='assigned_only' or text['shards']!=cfg['shards']:
        raise ValueError('Unexpected text-compression protocol')
    if text['result_root']!=sft['text_compression_root'] or text['candidate_root']!=sft['raw_candidate_root']:
        raise ValueError('Continuation roots do not share the registered source')
    if sft['expected_source_questions']!=text['expected_questions']:
        raise ValueError('Continuation source cohort counts differ')
    graph=nodes(cfg['shards'],cfg['dap_lanes'],cfg['tokenskip_lanes']);validate_topology(graph,cfg['upstream_jobs'])
    save(out/'graph.json',{'nodes':graph,'external_jobs':cfg['upstream_jobs'],'training_submitted':False})
    ids=dict(cfg['upstream_jobs']);records=[]
    for node in graph:
        cmd=command(cfg,node,ids)
        intent={'node':node,'command':cmd,'created_utc':datetime.now(timezone.utc).isoformat()}
        save(out/(node['key']+'_intent.json'),intent)
        # An uncertain submission result leaves its intent for manual scheduler
        # reconciliation. Never retry sbatch automatically or duplicate a DAG.
        response=subprocess.run(cmd,check=True,text=True,capture_output=True)
        job=int(response.stdout.strip().split(';')[0]);ids[node['key']]=job
        record={**intent,'job_id':job,'stdout':response.stdout,'stderr':response.stderr,
                'submitted_utc':datetime.now(timezone.utc).isoformat(),'experiment_complete':False}
        save(out/(node['key']+'_submission.json'),record);records.append(record)
    save(out/'summary.json',{'jobs':{r['node']['key']:r['job_id'] for r in records},'submitted_jobs':len(records),
        'maximum_dap_shards':cfg['dap_lanes'],'maximum_tokenskip_shards':cfg['tokenskip_lanes'],
        'all_sources_complete':False,'student_training_submitted':False,
        'scope':'Compression smoke/audit, 32+32 full shards, complete text merge, then all-eight student data preparation. Submission completion is not experiment completion.'})
    seal(out/'SUBMISSION_COMPLETE.json',[Path(config_path),launch/'FROZEN.json',launch/'TEST_COMPLETE.json',*sorted(out.glob('*'))],
         stage='bounded_unified_continuation_submission',experiment_complete=False)
    print(json.dumps({'submitted_jobs':len(records),'text_merge_job':ids['text_merge'],'sft_prepare_job':ids['sft_prepare']}))
