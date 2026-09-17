#!/usr/bin/env python3
"""Freeze and measure MATH calibration directions for B3 and B7."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.math_steering_directions import prepare, extract

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['test','prepare','extract'], required=True)
    p.add_argument('--split', default='smoke'); p.add_argument('--shard', type=int, default=0)
    a = p.parse_args(); logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.stage == 'test':
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern)
            for pattern in ['test_math_steering_directions.py','test_sae_mechanism_readback.py'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
    elif a.stage == 'prepare': prepare(a.config)
    else: extract(a.config)
