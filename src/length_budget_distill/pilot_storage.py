"""Workload-derived admission on administrator-designated NCSU scratch only."""
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import pwd


def estimate(cfg, stage, method=''):
    policy = cfg['runtime']['storage_policy']
    root = Path(cfg['result_root'])
    cache = policy['small_cache_bytes']
    temporary = cache
    output = cache
    details = {'small_cache_bytes': cache, 'model_weights_copied': False,
               'periodic_checkpoints': False, 'stage': stage}
    if stage in ('train', 'train-dev') and method != 'base':
        path = (root if root.name=='sft' else root/'sft')/'encoded/qwen3b_student'/(method+'.jsonl')
        size = path.stat().st_size
        model = json.loads((Path(cfg['students']['qwen3b_student']['snapshot_path'])/'config.json').read_text())
        h=model['hidden_size']; inter=model['intermediate_size']; layers=model['num_hidden_layers']
        kv=h*model['num_key_value_heads']//model['num_attention_heads']
        params=cfg['lora']['r']*layers*(2*(2*h)+2*(h+kv)+3*(h+inter))
        adapter=params*4
        temporary += policy['encoded_copies']*size + policy['adapter_copies']*adapter
        output += adapter
        details.update(encoded_bytes=size, adapter_fp32_bytes=adapter)
    elif stage=='directions':
        pairs=root/'directions/inputs/pairs.jsonl'
        count=sum(1 for _ in pairs.open())
        model=json.loads((Path(cfg['teacher']['snapshot_path'])/'config.json').read_text())
        output += count*2*model['hidden_size']*4 + pairs.stat().st_size
    elif stage in ('generate','compress','calibrate','evaluate-dev','evaluate-test'):
        evaluation=stage.startswith('evaluate')
        questions=cfg['development_questions'] if stage=='evaluate-dev' else (1269 if evaluation else cfg['maximum_source_questions'])
        candidates=1 if evaluation else cfg['candidates_per_question']
        methods=6 if evaluation else 9
        tokens=max(cfg['caps']) if evaluation else cfg['generation']['max_new_tokens']
        output += questions*candidates*methods*tokens*policy['serialized_bytes_per_token']
    factor=policy['safety_factor']
    return {**details, 'temporary_required_bytes':math.ceil(temporary*factor),
            'output_required_bytes':math.ceil(output*factor), 'safety_factor':factor}


def choose_location(scratch_free, publication_free, temporary, output):
    if publication_free < output:
        raise RuntimeError(f'Insufficient publication filesystem capacity: {publication_free} < {output}')
    if scratch_free < temporary:
        raise RuntimeError(f'Insufficient designated scratch: {scratch_free} < {temporary}; fallback is forbidden')
    return 'designated_scratch'


def validate_scratch(policy, environ):
    user=pwd.getpwuid(os.getuid()).pw_name
    designated=Path('/share/jekml')/user/'tmp'
    registered=Path(policy['scratch_root'])
    if registered.resolve() != designated.resolve() or not designated.is_dir():
        raise RuntimeError('Registered scratch must be the existing administrator-designated directory')
    expected=(designated/f"{user}-phase13-frozen-{environ['SLURM_JOB_ID']}").resolve()
    if expected.parent != designated.resolve():
        raise RuntimeError('Invalid job scratch directory')
    for key in ('TMPDIR','TMP','TEMP'):
        if key not in environ or Path(environ[key]).resolve() != expected:
            raise RuntimeError(f'{key} must point to the designated job scratch: {expected}')
    return expected


def quota_capacity(policy, scratch):
    spec=policy['scratch_quota']; reports={}; remaining=[]
    attr=subprocess.check_output([spec['attribute_command'],'-L',str(scratch)],text=True,timeout=30)
    if not any(line.strip() == 'fileset name:         '+spec['fileset'] for line in attr.splitlines()):
        # Whitespace in mmlsattr output varies across GPFS releases.
        actual=[line.split(':',1)[1].strip() for line in attr.splitlines() if line.strip().startswith('fileset name:')]
        if actual != [spec['fileset']]:raise RuntimeError('Scratch fileset differs from quota registration')
    reports['attributes']=attr
    for kind,identity in (('user',pwd.getpwuid(os.getuid()).pw_name),('group',spec['group'])):
        output=subprocess.check_output([spec['command'],'-u' if kind=='user' else '-g',identity,'-Y',spec['filesystem']],text=True,timeout=30)
        reports[kind]=output
        headers=None; selected=[]
        for line in output.splitlines():
            fields=line.split(':')
            if len(fields)>2 and fields[2]=='HEADER':headers=fields;continue
            if headers and fields[:2]==headers[:2]:
                row=dict(zip(headers,fields))
                if row.get('filesetname')==spec['fileset']:selected.append(row)
        if len(selected)!=1:raise RuntimeError(f'Cannot establish {kind} quota for designated scratch')
        row=selected[0]
        for resource in ('block','files'):
            limits=[int(row[resource+x]) for x in ('Quota','Limit') if int(row[resource+x])>0]
            if limits:
                available=min(limits)-int(row[resource+'Usage'])-int(row[resource+'InDoubt'])
                if resource=='block':remaining.append(max(0,available)*1024)
                elif available < policy.get('minimum_available_inodes',256):
                    raise RuntimeError('Insufficient scratch inode quota')
    return min(remaining) if remaining else None,reports


def configure(cfg, stage, method=''):
    budget=estimate(cfg,stage,method)
    root=Path(cfg['result_root']); job=os.environ['SLURM_JOB_ID'];policy=cfg['runtime']['storage_policy']
    selected=validate_scratch(policy,os.environ)
    selected.mkdir(exist_ok=True)
    quota_free,reports=quota_capacity(policy,selected)
    scratch_free=shutil.disk_usage(selected).free
    if quota_free is not None:scratch_free=min(scratch_free,quota_free)
    publication_free=shutil.disk_usage(root).free
    required=budget['temporary_required_bytes']
    if selected.stat().st_dev == root.stat().st_dev:required+=budget['output_required_bytes']
    location=choose_location(scratch_free,publication_free,required,budget['output_required_bytes'])
    tempfile.tempdir=str(selected)
    for key,name in [('HF_DATASETS_CACHE','datasets'),('MPLCONFIGDIR','matplotlib'),
                     ('XDG_CACHE_HOME','cache'),('TORCH_EXTENSIONS_DIR','torch_extensions'),
                     ('TRITON_CACHE_DIR','triton'),('NUMBA_CACHE_DIR','numba'),
                     ('CUDA_CACHE_PATH','cuda'),('JOBLIB_TEMP_FOLDER','joblib')]:
        target=selected/name;target.mkdir(exist_ok=True);os.environ[key]=str(target)
    out=root/'storage_admission'/job/(stage+'_'+str(os.getpid())+'.json');out.parent.mkdir(parents=True,exist_ok=True)
    record={**budget,'job_id':job,'hostname':socket.gethostname(),'location':location,'selected_path':str(selected),
            'scratch_available_bytes_including_quota':scratch_free,'publication_free_bytes':publication_free,
            'quota_available_bytes':quota_free,'quota_reports':reports,'fallback_allowed':False,
            'temporary_environment':{key:os.environ[key] for key in ('TMPDIR','TMP','TEMP','HF_DATASETS_CACHE','MPLCONFIGDIR')}}
    probe=selected/('storage_probe_'+str(os.getpid()))
    with probe.open('xb') as f:f.write(b'0'*4096);f.flush();os.fsync(f.fileno())
    probe.unlink()
    out.write_text(json.dumps(record,indent=2)+'\n')
    os.environ['LBD_PILOT_STORAGE_EVIDENCE']=str(out)
    print('Pilot storage admission: '+json.dumps(record),flush=True)
    return record
