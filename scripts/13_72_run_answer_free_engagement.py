#!/usr/bin/env python3
"""Freeze and audit same-state SAE target engagement for the answer-free dictionary."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'src'))
from length_budget_distill.ncsu_reproduction import isolated_gpu_preflight, seal, verify
from length_budget_distill.pilot_storage import validate_scratch, quota_capacity
from length_budget_distill.sae_target_engagement import freeze_engagement, run_engagement
from length_budget_distill.sae_engagement_analysis import analyze_engagement


def admit_storage(cfg, stage):
    scratch = validate_scratch(cfg['runtime']['storage_policy'], os.environ)
    quota_free, reports = quota_capacity(cfg['runtime']['storage_policy'], scratch)
    free = min(shutil.disk_usage(scratch).free, quota_free) if quota_free is not None else shutil.disk_usage(scratch).free
    needed = cfg['storage_required_bytes'][stage]
    if free < needed:
        raise RuntimeError(f'Insufficient designated scratch for {stage}: {free} < {needed}')
    print(json.dumps({'stage': stage, 'scratch': str(scratch), 'available_bytes': free,
                      'required_bytes': needed, 'quota_reports': reports}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('freeze', 'readback', 'analyze'), required=True)
    parser.add_argument('--feature-slot', type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if CODE.resolve() != Path(cfg['code_root']).resolve():
        raise ValueError('Use frozen engagement source')
    verify(Path(cfg['launch_root']) / 'protocol/LAUNCH_FROZEN.json')
    verify(Path(cfg['screen_root']) / 'analysis/COMPLETE.json')
    admit_storage(cfg, args.stage)
    protocol = Path(cfg['launch_root']) / 'protocol/clean_protocol.json'
    if args.stage == 'freeze':
        freeze_engagement(protocol, CODE)
    elif args.stage == 'readback':
        if args.feature_slot not in (0, 1, 2):
            raise ValueError('Invalid feature slot')
        isolated_gpu_preflight(args.config,
                               Path(cfg['launch_root']) / 'preflight' / os.environ['SLURM_JOB_ID'],
                               expected_name='L40S')
        run_engagement(protocol, CODE, args.feature_slot)
    else:
        analyze_engagement(protocol, CODE, version=1, finalize_failure=True)
        screen = Path(cfg['screen_root'])
        bindings = [screen / 'engagement_analysis_v1/engagement_analysis_manifest.json',
                    screen / 'engagement_protocol/protocol.json',
                    *(screen / f'engagement_shards/feature_{i:02d}/manifest.json' for i in range(3))]
        seal(Path(cfg['launch_root']) / 'analysis/COMPLETE.json', bindings,
             formal_claim_allowed=False)
