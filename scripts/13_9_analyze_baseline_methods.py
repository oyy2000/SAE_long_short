#!/usr/bin/env python3
"""Audit and summarize complete ASC method checks without changing raw results."""
import argparse
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.baseline_method_analysis import analyze_asc
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--test',action='store_true')
    args=p.parse_args()
    if args.test:
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_baseline_method_analysis.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        raise SystemExit(0)
    result=analyze_asc(args.config)
    print(json.dumps({'status':result['status'],'summaries':result['summaries'],'changed_grades':result['changed_grades']},indent=2))
