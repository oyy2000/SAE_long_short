#!/usr/bin/env python3
"""Submit a registered SAE or mathematical-candidate stage from frozen code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.ncsu_reproduction import save, seal, verify

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'test', 'extract', 'fit', 'probe', 'smoke', 'generate', 'merge', 'dap', 'tokenskip'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--route', choices=['cpu', 'h100', 'h200', 'l40s'], required=True)
    p.add_argument('--split', choices=['smoke', 'discovery', 'dev'], default='smoke')
    p.add_argument('--family', choices=['dose', 'ablation'], default='dose')
    p.add_argument('--cohort', choices=['smoke', 'development', 'student_pool'], default='smoke')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--dependency', type=int, nargs='+')
    a = p.parse_args(); path = Path(a.config).resolve(); cfg = json.loads(path.read_text())
    entrypoint = cfg.get('entrypoint', '13_22_measure_sae_readback.py')
    if entrypoint == '13_22_measure_sae_readback.py':
        allowed = {'prepare', 'test', 'extract', 'fit', 'probe'}
        axis_name, axis_value = '--split', a.split
    elif entrypoint == '13_25_run_sae_generation_controls.py':
        allowed = {'prepare', 'test', 'smoke', 'generate'}
        axis_name, axis_value = '--family', a.family
    elif entrypoint == '13_29_generate_math_candidates.py':
        allowed = {'prepare', 'test', 'generate', 'merge'}
        axis_name, axis_value = '--cohort', a.cohort
    elif entrypoint == '13_31_compress_math_candidates.py':
        allowed = {'prepare', 'test', 'dap', 'tokenskip', 'merge'}
        axis_name, axis_value = '--cohort', a.cohort
    elif entrypoint == '13_32_prepare_math_directions.py':
        allowed = {'prepare', 'test', 'extract'}
        axis_name, axis_value = '--split', a.split
    else: raise ValueError('Unregistered experiment entrypoint')
    if a.stage not in allowed: raise ValueError('Stage is incompatible with the configured entrypoint')
    if (a.stage in ('prepare', 'test', 'fit', 'merge')) != (a.route == 'cpu'):
        raise ValueError('CPU for preparation/tests/direction fitting/merge; GPU for generation and readback')
    project = Path(cfg['project_root']); logs = project/cfg['launch_root']; logs.mkdir(parents=True, exist_ok=True)
    if a.stage in ('prepare', 'test'):
        launch = logs/(a.stage+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')); code = launch/'code'
        for folder in ('src', 'scripts', 'configs', 'tests'):
            shutil.copytree(ROOT/folder, code/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'), symlinks=True)
        path = launch/'frozen_config.json'; save(path, cfg)
        seal(launch/'FROZEN.json', [path]+[f for f in code.rglob('*') if f.is_file() and not f.is_symlink()])
    else:
        code = Path(cfg['code_root']); root = Path(cfg['result_root'])
        verify(root/'protocol/FROZEN.json'); verify(root/'protocol/SOURCES.json')
    route = cfg['runtime']['routes'][a.route]
    cmd = ['sbatch', '--parsable', '--account='+route['account'], '--partition='+route['partition'], '--qos='+route['qos'],
        '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(route['cpus']), '--mem='+route['memory'], '--time='+route['time'],
        '--job-name='+cfg.get('job_name_prefix','p13_sae')+'_'+a.stage+'_'+axis_value+'_'+str(a.shard),
        '--output='+str(logs/('%j_'+a.stage+'_'+axis_value+'_'+str(a.shard)+'.log'))]
    if a.route != 'cpu': cmd += ['--gres='+route['gres']]
    if route.get('nodelist'): cmd += ['--nodelist='+route['nodelist']]
    if a.dependency: cmd += ['--dependency=afterok:'+':'.join(map(str, a.dependency))]
    cmd += [str(code/'scripts/slurm/13_6_run_frozen_python.sh'), str(code), str(path),
        cfg['runtime']['python'], cfg['runtime']['overlay'], entrypoint,
        '--stage', a.stage, axis_name, axis_value, '--shard', str(a.shard)]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    job = result.stdout.strip().split(';')[0]
    record = {'job_id': job, 'command': cmd, 'stage': a.stage, 'split': a.split, 'shard': a.shard,
        'route': a.route, 'entrypoint': entrypoint, 'family': a.family, 'cohort':a.cohort,
        'submitted_utc': datetime.now(timezone.utc).isoformat(), 'completed': False}
    save(logs/(job+'_submission.json'), record); print(json.dumps(record))
