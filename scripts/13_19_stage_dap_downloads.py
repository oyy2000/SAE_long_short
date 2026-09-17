#!/usr/bin/env python3
"""Stream pinned public DAP source files; CPU Slurm subsequently hashes/parses them.

Use on a host with outbound HTTPS. Retain partial downloads on failure, publish
only exact-size completed files, and never overwrite existing files silently.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import requests
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());staged=Path(cfg['staged_root']);targets=[]
    for name,spec in cfg['datasets'].items():
        info=json.loads(Path(spec['metadata_path']).read_text())
        if info['sha']!=spec['revision']:raise ValueError('Revision changed')
        prefix=spec.get('file_prefix','data/')
        for item in info['siblings']:
            if item['rfilename'].startswith(prefix) and item['rfilename'].endswith('.parquet'):targets.append((name,spec,item))
    def transfer(target):
        name,spec,item=target;dest=staged/name/item['rfilename'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.stat().st_size!=item['size']:raise ValueError('Existing file has wrong size: '+str(dest))
            return {'path':str(dest),'bytes':dest.stat().st_size,'status':'existing_pending_cpu_hash_check'}
        temp=dest.with_name(dest.name+'.partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        url='https://huggingface.co/datasets/'+spec['repo_id']+'/resolve/'+spec['revision']+'/'+item['rfilename']+'?download=true'
        print('Downloading',name,item['rfilename'],item['size'],flush=True)
        try:
            with requests.get(url,stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                with temp.open('xb') as handle:
                    for block in response.iter_content(chunk_size=1024*1024):
                        if block:handle.write(block)
            if temp.stat().st_size!=item['size']:raise ValueError('Wrong download size')
            temp.rename(dest)
        except Exception as error:
            # Avoid printing redirected signed URLs or credentials in exceptions.
            raise RuntimeError('Download failed for '+name+'/'+item['rfilename']+' ('+type(error).__name__+')') from None
        print('Transferred',name,item['rfilename'],flush=True)
        return {'path':str(dest),'bytes':dest.stat().st_size,'status':'transferred_pending_cpu_hash_check'}
    with ThreadPoolExecutor(max_workers=cfg['download_workers']) as pool:records=list(pool.map(transfer,targets))
    save(staged/'TRANSFER_COMPLETE.json',{'status':'transfer_complete_not_hash_verified','files':records,
        'completed_utc':datetime.now(timezone.utc).isoformat(),'cpu_sha256_verification_required':True})
