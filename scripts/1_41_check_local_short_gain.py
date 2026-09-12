#!/usr/bin/env python3
"""Recover historical short data and measure gains in the C31 local runtime."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from length_budget_distill import legacy_replication, local_short_gain_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'analyze'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = legacy_replication.config_load(args.config)
    if args.stage == 'worker':
        if not args.arm:
            parser.error('worker requires --arm')
        local_short_gain_check.worker(config, args.arm)
    else:
        getattr(local_short_gain_check, args.stage)(config)


if __name__ == '__main__':
    main()
