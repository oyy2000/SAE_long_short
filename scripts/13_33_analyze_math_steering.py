#!/usr/bin/env python3
"""Audit complete shared-teacher curves and select registered operating points."""
import argparse
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.unified_steering_analysis import analyze
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--test',action='store_true')
    a=p.parse_args()
    if a.test:
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_unified_steering_analysis.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:analyze(a.config)
