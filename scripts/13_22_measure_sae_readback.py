#!/usr/bin/env python3
"""Run one frozen SAE extraction, control-direction fit, or local causal stage."""
import argparse
import logging
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.sae_mechanism_readback import prepare, load, extract, fit_directions, probe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['prepare', 'test', 'extract', 'fit', 'probe'], required=True)
    parser.add_argument('--split', choices=['smoke', 'discovery', 'dev'], default='smoke')
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'test':
        import unittest
        suite = unittest.TestSuite()
        for pattern in ('test_sae_mechanism_readback.py', 'test_sae_paired_intervention.py', 'test_sae_norm_intervention.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): sys.exit(1)
    elif args.stage == 'prepare': prepare(args.config)
    elif args.stage == 'fit': fit_directions(load(args.config))
    elif args.stage == 'extract': extract(load(args.config), args.split, args.shard)
    else: probe(load(args.config), args.split, args.shard)
