#!/usr/bin/env python3
"""Run reference-integrity fixtures and scan the registered MATH cohorts."""
import argparse
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_math_reference_integrity.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    from length_budget_distill.math_reference_integrity import scan
    scan(args.config)
