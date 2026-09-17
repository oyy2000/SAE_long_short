#!/usr/bin/env python3
"""Prepare and train an answer-free layer-17 SAE on audited NCSU activations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from length_budget_distill.ncsu_reproduction import verify,seal,save,isolated_gpu_preflight
from length_budget_distill.pilot_storage import validate_scratch,quota_capacity
from length_budget_distill.sae_body_training import prepare,sample
from length_budget_distill.experiment_io import publish_files_hash_verified
from length_budget_distill.factorial import file_sha256


def storage(cfg, stage):
    import shutil
    policy=cfg['runtime']['storage_policy'];scratch=validate_scratch(policy,os.environ)
    required=cfg['storage_requirements_bytes'][stage]
    quota,reports=quota_capacity(policy,scratch)
    free=shutil.disk_usage(scratch).free
    available=min(free,quota) if quota is not None else free
    if available<required:raise RuntimeError(f'Designated scratch has {available} bytes, requires {required}')
    save(Path(cfg['result_root'])/'storage_admission'/os.environ['SLURM_JOB_ID']/f'{stage}.json',
         {'stage':stage,'scratch':str(scratch),'required_bytes':required,'available_bytes':available,
          'filesystem_free_bytes':free,'quota_free_bytes':quota,'quota_reports':reports,'fallback':False})
    return scratch


def train(cfg,args):
    root=Path(cfg['result_root']);verify(root/'token_samples/COMPLETE.json')
    scratch=validate_scratch(cfg['runtime']['storage_policy'],os.environ)
    out=scratch/'answer_free_sae';out.mkdir(exist_ok=False)
    isolated_gpu_preflight(args.config,root/'preflight'/os.environ['SLURM_JOB_ID'],expected_name='H200')
    config=root/'protocol/training_config.json';samples=root/'token_samples'
    command=[sys.executable,str(CODE/'scripts/2_4_train_topk_sae.py'),'--config',str(config),
             '--sample-root',str(samples),'--layer-index',str(cfg['layer_index']),'--k',str(cfg['k']),
             '--output-dir',str(out/'metrics'),'--checkpoint-dir',str(out/'checkpoints')]
    save(root/'training/intent.json',{'command':command,'temporary_checkpoint_dir':str(out/'checkpoints')})
    start=time.monotonic();subprocess.run(command,check=True)
    from length_budget_distill.experiment_io import validated_artifact_marker
    metrics=out/'metrics/training_metrics.json';model=out/'checkpoints/sae_model.safetensors'
    validated_artifact_marker(out/'metrics/SAE_TRAINING_COMPLETE',expected_status='complete',
                              hash_bindings={'training_metrics_sha256':metrics,'model_sha256':model})
    actual=json.loads(metrics.read_text());expected=json.loads(config.read_text())
    from length_budget_distill.factorial import canonical_sha256
    if actual['config_hash']!=canonical_sha256(expected) or actual['max_steps']!=cfg['sae_steps']:
        raise ValueError('SAE training did not match registered config')
    dest=Path(cfg['checkpoint_root']);hashes=publish_files_hash_verified(model.parent,dest,('sae_model.safetensors',))
    from shutil import copyfile
    copyfile(metrics,root/'training/training_metrics.json')
    save(root/'training/summary.json',{'elapsed_seconds':time.monotonic()-start,
         'model_sha256':hashes['sae_model.safetensors'],'training_config_sha256':file_sha256(config),
         'sample_manifest_sha256':file_sha256(samples/'sample_manifest.json'),
         'checkpoint_source_sha256':file_sha256(CODE/'scripts/2_4_train_topk_sae.py')})
    seal(root/'training/COMPLETE.json',[root/'protocol/FROZEN.json',samples/'COMPLETE.json',config,
         root/'training/summary.json',root/'training/training_metrics.json',dest/'sae_model.safetensors'],formal_claim_allowed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['prepare','sample','train'],required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());verify(Path(cfg['result_root'])/'protocol/FROZEN.json')
    if CODE.resolve()!=Path(cfg['code_root']).resolve():raise ValueError('Use frozen source tree')
    storage(cfg,args.stage)
    if args.stage=='prepare':prepare(cfg)
    elif args.stage=='sample':sample(cfg)
    else:train(cfg,args)
