"""Record runtime compatibility and asset availability; never authorize generation."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys

from .experiment_io import read_json, write_json_exclusive, write_text_exclusive
from .factorial import file_sha256


def runtime_preflight(config_path: Path, project: Path, output: Path) -> dict:
    import torch
    import transformers

    config_path = config_path.resolve()
    project = project.resolve()
    config = read_json(config_path)
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    blockers = []
    for name, item in config['required_assets'].items():
        path = project / item['path']
        exists = path.is_file()
        sha = file_sha256(path) if exists else None
        valid = exists and (not item.get('sha256') or sha == item['sha256'])
        assets[name] = {'path': str(path), 'resolved_path': str(path.resolve()),
                        'exists': exists, 'sha256': sha,
                        'expected_sha256': item.get('sha256'), 'available': valid}
        if not valid:
            blockers.append(f'{name}: missing' if not exists else f'{name}: hash mismatch')
    expected_python = Path(config['python_executable']).resolve()
    if Path(sys.executable).resolve() != expected_python:
        blockers.append('unexpected_python_executable')
    if platform.node().split('.')[0].lower() != config['expected_node']:
        blockers.append('unexpected_node')
    if not torch.cuda.is_available():
        blockers.append('cuda_unavailable')
    runtime_script = Path(config['runtime_script'])
    if not runtime_script.is_file():
        blockers.append('runtime_script_missing')
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
               '-p', config['test_pattern']]
    tests = subprocess.run(command, cwd=project, text=True, capture_output=True,
                           timeout=180)
    write_text_exclusive(output / 'compatibility_tests.log', tests.stdout + tests.stderr)
    if tests.returncode:
        blockers.append('compatibility_tests_failed')
    gpu = subprocess.run(['nvidia-smi'], text=True, capture_output=True, timeout=30)
    write_text_exclusive(output / 'nvidia_smi.txt', gpu.stdout + gpu.stderr)
    report = {
        'status': 'blocked' if blockers else 'availability_checks_passed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'formal_claim_allowed': False, 'generation_executed': False,
        'generation_authorized_by_this_preflight': False,
        'scope': 'Runtime unit tests and file availability only; full model, provenance, and intervention validation still required.',
        'blockers': blockers, 'primary_feature_id': config['primary_feature_id'],
        'assets': assets, 'node': platform.node(), 'allocation': os.getenv('SLURM_JOB_ID'),
        'python_executable': sys.executable, 'python_version': platform.python_version(),
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count(),
        'hf_home': os.getenv('HF_HOME'), 'tmpdir': os.getenv('TMPDIR'),
        'runtime_script_sha256': file_sha256(runtime_script) if runtime_script.is_file() else None,
        'config_path': str(config_path), 'config_sha256': file_sha256(config_path),
        'source_sha256': file_sha256(Path(__file__)),
        'test_source_sha256': file_sha256(project / 'tests' / config['test_pattern']),
        'tests': {'command': command, 'exit_code': tests.returncode,
                  'log_sha256': file_sha256(output / 'compatibility_tests.log')},
        'gpu_snapshot_exit_code': gpu.returncode,
        'gpu_snapshot_sha256': file_sha256(output / 'nvidia_smi.txt'),
    }
    write_json_exclusive(output / 'preflight.json', report)
    # This is deliberately not an experimental completion marker.
    return report
