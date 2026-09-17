"""Archive owned scratch from terminal SAE jobs before explicit recovery."""
from pathlib import Path
import json
import os
import shutil
import socket
import subprocess

from .experiment_io import read_json
from .factorial import file_sha256
from .ncsu_reproduction import save, seal, verify


def archive(config_path):
    cfg=read_json(config_path);root=Path(cfg['result_root'])
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    if socket.gethostname().split('.')[0] != cfg['node']:raise ValueError('Archive must run on the original node')
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);records=[]
    before=shutil.disk_usage('/var/tmp')._asdict()
    for item in cfg['failed_jobs']:
        job=int(item['job_id']);seed=int(item['seed'])
        state=subprocess.run(['sacct','-j',str(job),'-X','-n','-P','--format=JobID,State,ExitCode,ElapsedRaw,NodeList'],
            text=True,capture_output=True,check=True).stdout.strip()
        lines=[line.split('|') for line in state.splitlines() if line.split('|')[0]==str(job)]
        if len(lines)!=1 or lines[0][1]!='FAILED' or lines[0][4]!=cfg['node']:
            raise ValueError('Source job is not terminal FAILED on the registered node')
        source=Path('/var/tmp')/f"{cfg['user']}-phase13-frozen-{job}"/f'sae_seed_{seed}'
        target=root/f'job_{job}'
        if source.exists():
            if source.is_symlink() or source.stat().st_uid!=os.getuid():raise ValueError('Unsafe scratch owner/path')
            files=[p for p in source.rglob('*') if p.is_file()]
            if any(p.is_symlink() or p.stat().st_uid!=os.getuid() for p in files):raise ValueError('Unexpected scratch file owner/link')
            shutil.copytree(source,target)
            hashes={}
            for path in files:
                copied=target/path.relative_to(source);digest=file_sha256(path)
                if file_sha256(copied)!=digest:raise ValueError('Scratch archive hash mismatch')
                hashes[str(copied)]=digest
            save(root/f'job_{job}_archive.json',{'job_id':job,'seed':seed,'source':str(source),'sacct':state,
                'hashes':hashes,'source_bytes':sum(p.stat().st_size for p in files),'copied_before_cleanup':True})
            # Only this task's failed-job SAE directory is removed after every
            # byte is preserved in shared storage. Other job scratch is untouched.
            shutil.rmtree(source)
            records.append({'job_id':job,'seed':seed,'archive':str(target),'files':len(files),'owned_source_removed':True})
        else:records.append({'job_id':job,'seed':seed,'source_missing':True,'sacct':state})
    save(root/'summary.json',{'records':records,'disk_before':before,'disk_after':shutil.disk_usage('/var/tmp')._asdict(),
        'scope':'Only registered terminal failed SAE job directories; all existing bytes archived and verified before cleanup.'})
    seal(root/'COMPLETE.json',[Path(config_path),*[p for p in root.rglob('*') if p.is_file()]],stage='failed_sae_scratch_archive')
