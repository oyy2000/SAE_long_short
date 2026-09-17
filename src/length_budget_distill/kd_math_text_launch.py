"""Release reviewed MATH DAP/TokenSkip only after audited source preparation."""
from pathlib import Path
import json
import subprocess

from .experiment_io import read_json
from .ncsu_reproduction import save,seal,verify
from .unified_pipeline_launch import nodes,validate_topology,command,scheduler_record

CODE=Path(__file__).resolve().parents[2]


def submit(config_path):
    cfg=read_json(config_path);root=Path(cfg['launch_root'])
    verify(root/'FROZEN.json');verify(root/'SOURCES.json')
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen text controller')
    upstream=scheduler_record(cfg['text_prepare_job'])
    if upstream['state']!='COMPLETED':raise ValueError('Reviewed text inputs are not prepared')
    text=Path(cfg['text_result_root']);verify(text/'protocol/FROZEN.json');verify(text/'protocol/SOURCES.json')
    spec=read_json(text/'protocol/frozen_config.json')
    if spec['candidate_root']!=cfg['raw_reviewed_root'] or spec['source_cohort']!='student_pool' or spec['compression_mode']!='assigned_only':
        raise ValueError('Wrong reviewed text source or ratio assignment')
    graph=nodes(cfg['shards'],cfg['dap_lanes'],cfg['tokenskip_lanes'])[:-1]
    validate_topology(graph,{'text_prepare':cfg['text_prepare_job']})
    ids={'text_prepare':cfg['text_prepare_job']};records=[]
    # Reuse established DAG topology and command construction. Its final SFT
    # prepare node is intentionally excluded until all student-source audits pass.
    adapted={**cfg,'text_config_path':str(text/'protocol/frozen_config.json')}
    for node in graph:
        out=root/'submissions'/node['key'];out.mkdir(parents=True,exist_ok=False)
        cmd=command(adapted,node,ids)
        cmd.insert(cmd.index(str(CODE/'scripts/slurm/13_6_run_frozen_python.sh')),'--kill-on-invalid-dep=yes')
        save(out/'intent.json',{'command':cmd,'node':node})
        job=int(subprocess.check_output(cmd,text=True).strip().split(';')[0]);ids[node['key']]=job
        save(out/'submission.json',{'job_id':job,'command':cmd,'node':node})
        spool=out/'submitted_batch.sh';subprocess.run(['scontrol','write','batch_script',str(job),str(spool)],check=True,capture_output=True)
        launcher=CODE/'scripts/slurm/13_6_run_frozen_python.sh'
        if spool.read_bytes()!=launcher.read_bytes():raise ValueError('Actual text batch script changed')
        seal(out/'SUBMISSION_VERIFIED.json',[out/'intent.json',out/'submission.json',spool,launcher])
        records.append({'key':node['key'],'job_id':job,'route':node['route'],'parents':node['parents']})
    save(root/'jobs.json',{'nodes':records,'text_merge_job':ids['text_merge'],
        'student_data_preparation_submitted':False,'math_kd_training_submitted':False,
        'gpu_lanes':{'dap_h200':cfg['dap_lanes'],'tokenskip_l40s':cfg['tokenskip_lanes']}})
    seal(root/'SUBMISSION_COMPLETE.json',[root/'FROZEN.json',text/'protocol/FROZEN.json',root/'jobs.json',
        *sorted(root.glob('submissions/*/SUBMISSION_VERIFIED.json'))],training_release=False)
    print(json.dumps({'submitted':len(records),'text_merge':ids['text_merge']}))
