#!/usr/bin/env python3
"""Prepare, launch, run, or audit the isolated historical-recipe replication."""
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from length_budget_distill import legacy_replication as impl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'submit', 'worker', 'train', 'evaluate', 'analyze'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--name')
    parser.add_argument('--runtime')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shards', type=int)
    parser.add_argument('--gpu-ids')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = impl.config_load(args.config)
    if args.stage in ('prepare', 'submit', 'analyze'):
        getattr(impl, args.stage)(config)
    elif args.stage == 'worker':
        if args.shard is None or not args.shards or not args.gpu_ids or not args.runtime:
            parser.error('worker requires shard, shards, gpu-ids and runtime')
        impl.worker(config, args.shard, args.shards, args.gpu_ids.split(','), args.runtime)
    else:
        if not args.name or not args.runtime:
            parser.error('train/evaluate requires name and runtime')
        getattr(impl, args.stage)(config, args.name, args.runtime)


if __name__ == '__main__':
    main()
