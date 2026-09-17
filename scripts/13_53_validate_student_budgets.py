#!/usr/bin/env python3
"""Prepare or evaluate the registered development-only student budget grid."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.student_cap_validation import prepare, run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['test', 'prepare', 'generate'], required=True)
    parser.add_argument('--student')
    parser.add_argument('--method', default='base')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'test':
        suite = unittest.TestSuite()
        for pattern in ('test_student_cap_validation.py', 'test_unified_student_evaluation.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    elif args.stage == 'prepare':
        prepare(args.config)
    else:
        if not args.student:
            parser.error('Generation requires a registered student')
        run(args.config, args.student, args.method, args.seed)
