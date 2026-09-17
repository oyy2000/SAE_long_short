#!/usr/bin/env python3
"""Test the shared student evaluator or run its synthetic GPU adapter smoke."""
import argparse
import logging
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['test','smoke'],required=True);p.add_argument('--student')
    args=p.parse_args();logging.basicConfig(level=logging.INFO)
    if args.stage=='test':
        suite=unittest.TestSuite()
        for pattern in ('test_unified_student_evaluation.py','test_tokenskip_reproduction.py','test_student_evaluation_prompt.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:
        if not args.student:p.error('Smoke requires --student')
        from length_budget_distill.unified_student_evaluation import smoke
        smoke(args.config,args.student)
