#!/usr/bin/env python3
"""Build and diagnose the immutable-source MATH grading overlay."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info('Building MATH cohort overlay with %s', args.config)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_reviewed_math*.py')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        sys.exit(1)
    from length_budget_distill.reviewed_math_cohort import build
    build(args.config)
