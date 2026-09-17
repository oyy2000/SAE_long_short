#!/usr/bin/env python3
"""Audit and summarize the full SAE generation grid after fresh answer review."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--test',action='store_true');args=parser.parse_args()
    if args.test:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_sae_generation_analysis.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():sys.exit(1)
    else:
        from length_budget_distill.sae_generation_analysis import analyze
        analyze(args.config)
