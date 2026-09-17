#!/usr/bin/env python3
"""Freeze MATH calibration/dev/student roles with grouped near duplicates."""
import argparse
import logging
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.math_cohort_freeze import freeze

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--test', action='store_true')
    a = p.parse_args(); logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_math_cohort_freeze.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): sys.exit(1)
    else: freeze(a.config)
