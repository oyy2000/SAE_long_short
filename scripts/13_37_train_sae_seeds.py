#!/usr/bin/env python3
"""Prepare immutable sample views or train one additional SAE seed."""
import argparse
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['test','prepare','train'],required=True)
    parser.add_argument('--seed',type=int);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage=='test':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sae_seed_stability.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():sys.exit(1)
    else:
        from length_budget_distill.sae_seed_stability import prepare,train
        if args.stage=='prepare':prepare(args.config)
        elif args.seed is None:parser.error('--seed is required for training')
        else:train(args.config,args.seed)
