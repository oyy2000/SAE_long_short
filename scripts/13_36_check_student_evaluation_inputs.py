#!/usr/bin/env python3
"""Check native evaluation prompt formats and context budgets on CPU."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.student_evaluation_preflight import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_student_evaluation_prompt.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    run(a.config)
