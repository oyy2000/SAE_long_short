#!/usr/bin/env python3
"""Prepare all-eight common support or train a registered unified student cell."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.unified_student_distillation import prepare, train

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--stage',choices=['test','prepare','train'],required=True)
    p.add_argument('--student',choices=['qwen1_5b_student','qwen3b_student'])
    p.add_argument('--method',choices=['B'+str(i) for i in range(8)]);p.add_argument('--seed',type=int)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='test':
        suite=unittest.TestSuite()
        for pattern in ('test_unified_student_distillation.py','test_reviewed_math_candidate_regrading.py',
                        'test_unified_text_compression.py','test_unified_steering_analysis.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    elif a.stage=='prepare':prepare(a.config)
    else:
        if a.student is None or a.method is None or a.seed is None:p.error('Training requires --student, --method and --seed')
        train(a.config,a.student,a.method,a.seed)
