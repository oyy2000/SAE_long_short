#!/usr/bin/env python3
"""Freeze the clean-feature readback protocol and submit its dependency chain."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.ncsu_reproduction import save, seal, verify
from length_budget_distill.sae_clean_features import load_clean_protocol


def submit(cfg, key, stage, parents=(), feature_slot=None):
    launch = Path(cfg['launch_root'])
    out = launch / 'jobs' / key
    out.mkdir(parents=True, exist_ok=False)
    gpu = stage == 'readback'
    route = cfg['runtime']['routes']['l40s' if gpu else 'cpu']
    command = ['sbatch', '--parsable', '--account=' + route['account'],
               '--partition=' + route['partition'], '--qos=' + route['qos'],
               '--nodes=1', '--ntasks=1', '--cpus-per-task=' + str(route['cpus']),
               '--mem=' + route['memory'], '--time=' + route['time'],
               '--job-name=sae_body_' + key, '--output=' + str(out / '%j.log')]
    if gpu:
        command.append('--gres=' + route['gres'])
    if parents:
        command += ['--dependency=afterok:' + ':'.join(map(str, parents)),
                    '--kill-on-invalid-dep=yes']
    command += [str(launch / 'code/scripts/slurm/13_6_run_frozen_python.sh'),
                str(launch / 'code'), str(launch / 'protocol/launch_config.json'),
                cfg['runtime']['python'], cfg['runtime']['overlay'],
                '13_72_run_answer_free_engagement.py', '--stage', stage]
    if feature_slot is not None:
        command += ['--feature-slot', str(feature_slot)]
    save(out / 'intent.json', {'command': command, 'parents': list(parents)})
    job = int(subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip().split(';')[0])
    save(out / 'submission.json', {'job_id': job, 'command': command})
    return job


if __name__ == '__main__':
    screen = ROOT / 'results/phase13_sae_feature_separation_v1/exploratory/answer_free_sae_v2/screen'
    verify(screen / 'analysis/COMPLETE.json')
    selected = json.loads((screen / 'feature_gate/selected_features.json').read_text())
    if not selected['gate_passed'] or not selected['selected']:
        raise ValueError('No clean feature passed')
    launch = screen / 'engagement_launch_v1'
    if launch.exists():
        raise FileExistsError(launch)
    base, _ = load_clean_protocol(screen / 'protocol/frozen_protocol.json')
    phase5 = json.loads((ROOT / 'configs/phase5_sae_clean_feature_causal_v1.json').read_text())
    base['engagement'] = phase5['engagement'] | {
        'match_on_readback_positions': True,
        'require_dev_control_coverage': True,
    }
    base['old_selected_features'].setdefault('long_feature_ids', [])
    original = json.loads((screen / 'protocol/launch_config.json').read_text())
    runtime = original['runtime']
    cfg = {'project_root': str(ROOT), 'screen_root': str(screen), 'launch_root': str(launch),
           'runtime': runtime,
           'storage_required_bytes': {'freeze': 1073741824, 'readback': 2147483648,
                                      'analyze': 1073741824},
           'claim_boundary': 'Observed dev/test exploratory engagement, not new-question confirmation.'}
    inventory = subprocess.check_output(['sinfo', '-N', '-p', 'gpu_partners', '-o', '%N %P %G %t'], text=True)
    qos = subprocess.check_output(['sacctmgr', '-nP', 'show', 'qos', 'short_gpu,short',
                                   'format=Name,MaxWall,GrpTRES,MaxTRESPU'], text=True)
    if 'gpu:l40s:4 mix' not in inventory or 'short_gpu|02:00:00' not in qos:
        raise RuntimeError('Registered L40S route unavailable')
    launch.mkdir(parents=True)
    save(launch / 'protocol/scheduler_preflight.json', {'inventory': inventory, 'qos': qos})
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(ROOT / folder, launch / 'code' / folder, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    cfg['code_root'] = str(launch / 'code')
    save(launch / 'protocol/launch_config.json', cfg)
    save(launch / 'protocol/clean_protocol.json', base)
    (launch / 'protocol/PROTOCOL_FROZEN').write_text(
        f'status=frozen\nconfig_hash={canonical_sha256(base)}\n'
        f'config_sha256={file_sha256(launch / "protocol/clean_protocol.json")}\n')
    load_clean_protocol(launch / 'protocol/clean_protocol.json')
    seal(launch / 'protocol/SOURCES.json',
         [p for p in (launch / 'code').rglob('*') if p.is_file() and not p.is_symlink()])
    seal(launch / 'protocol/LAUNCH_FROZEN.json',
         [screen / 'analysis/COMPLETE.json', launch / 'protocol/launch_config.json',
          launch / 'protocol/clean_protocol.json', launch / 'protocol/PROTOCOL_FROZEN',
          launch / 'protocol/SOURCES.json', launch / 'protocol/scheduler_preflight.json'],
         formal_claim_allowed=False)
    freeze = submit(cfg, 'engagement_freeze', 'freeze')
    readbacks = [submit(cfg, f'engagement_{i:02d}', 'readback', [freeze], i)
                 for i in range(3)]
    analysis = submit(cfg, 'engagement_analyze', 'analyze', readbacks)
    save(launch / 'protocol/submitted_jobs.json',
         {'freeze': freeze, 'readbacks': readbacks, 'analysis': analysis,
          'generation_submitted': False})
    print(json.dumps({'freeze': freeze, 'readbacks': readbacks, 'analysis': analysis}))
