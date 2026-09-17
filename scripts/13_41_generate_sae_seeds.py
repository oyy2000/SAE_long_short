#!/usr/bin/env python3
"""Run the separately registered equal-count SAE seed supplement."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['test','prepare','smoke','generate'],required=True)
    p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='test':
        import unittest
        suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern)
            for pattern in ('test_sae_seed_generation.py','test_sae_generation_controls.py',
                            'test_sae_generation_analysis.py','test_sae_mechanism_readback.py'))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():sys.exit(1)
    else:
        from length_budget_distill.sae_seed_generation import prepare,generate
        if a.stage=='prepare':prepare(a.config)
        else:generate(a.config,a.shard,smoke=a.stage=='smoke')
