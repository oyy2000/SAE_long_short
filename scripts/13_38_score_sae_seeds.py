#!/usr/bin/env python3
"""Prepare or run original-rule feature scoring for an additional SAE seed."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['test','prepare','score'],required=True);p.add_argument('--seed',type=int)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='test':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sae_seed_features.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():sys.exit(1)
    else:
        from length_budget_distill.sae_seed_features import prepare,score
        if a.stage=='prepare':prepare(a.config)
        elif a.seed is None:p.error('--seed required')
        else:score(a.config,a.seed)
