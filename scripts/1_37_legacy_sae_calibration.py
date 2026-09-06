#!/usr/bin/env python3
"""Queue, run, and audit dev-only SAE intervention calibration."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from length_budget_distill import legacy_sae_calibration as impl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'prepare', 'worker', 'generate', 'analyze'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--dependency')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gpu-ids')
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    if args.action == 'register':
        if not args.dependency: parser.error('register requires --dependency')
        impl.register(args.config, args.dependency)
    elif args.action == 'prepare': impl.prepare(args.config)
    elif args.action == 'analyze': impl.analyze(args.config)
    elif args.action == 'generate':
        if args.task is None: parser.error('generate requires --task')
        impl.generate(args.config, args.task)
    else:
        if args.worker is None or not args.workers or not args.gpu_ids:
            parser.error('worker requires --worker, --workers, --gpu-ids')
        impl.worker(args.config, args.worker, args.workers, args.gpu_ids.split(','))


if __name__ == '__main__': main()
