#!/usr/bin/env python3
"""Run the replication-gated SAE continuation without touching old artifacts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from length_budget_distill import legacy_sae_continuation as impl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'prepare', 'worker', 'audit'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--dependency')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shards', type=int)
    parser.add_argument('--gpu-ids')
    args = parser.parse_args()
    config = impl.config_load(args.config)
    if args.action == 'register':
        if not args.dependency: parser.error('--dependency is required')
        impl.register(config, args.dependency)
    elif args.action == 'prepare': impl.prepare_and_submit(config)
    elif args.action == 'audit': impl.audit(config)
    else:
        if args.shard is None or not args.shards or not args.gpu_ids: parser.error('worker requires shard, shards, gpu-ids')
        impl.worker(config, args.shard, args.shards, args.gpu_ids.split(','))


if __name__ == '__main__': main()
