#!/usr/bin/env python3
"""Run one frozen full-generation SAE control stage."""
import argparse
import logging
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.sae_generation_controls import prepare, load, run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['prepare', 'test', 'smoke', 'generate'], required=True)
    p.add_argument('--family', choices=['dose', 'ablation'], default='dose')
    p.add_argument('--shard', type=int, default=0)
    a = p.parse_args(); logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.stage == 'prepare': prepare(a.config)
    elif a.stage == 'test':
        import unittest
        suite = unittest.TestSuite()
        for pattern in ('test_sae_generation_controls.py', 'test_sae_norm_intervention.py', 'test_sae_generation_analysis.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): sys.exit(1)
    else: run(load(a.config), a.family, a.shard, smoke=a.stage == 'smoke')
