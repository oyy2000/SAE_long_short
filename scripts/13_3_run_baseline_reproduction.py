#!/usr/bin/env python3
"""Run one frozen ASC/DAP/TokenSkip preparation or method-validation stage."""
import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from length_budget_distill import baseline_reproduction as run
from length_budget_distill.experiment_io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['test','prepare','asc-calibrate','asc-smoke','asc-train','asc-smoke-eval','asc-eval','dap','tokenskip'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('stage=%s config=%s code_root=%s', args.stage, args.config, ROOT)
    allowed = read_json(args.config).get('allowed_stages')
    if allowed is not None and args.stage not in allowed:
        raise ValueError('Stage is not registered for this input protocol: '+args.stage)
    if args.stage == 'test':
        import unittest
        suite = unittest.TestSuite()
        for pattern in ('test_compression_baselines.py','test_math_baseline_calibration.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): sys.exit(1)
        return
    if args.stage == 'prepare':
        run.prepare(args.config)
        return
    cfg = run.load_frozen(args.config)
    if args.stage == 'asc-calibrate':
        run.generate_calibration_pairs(cfg)
    elif args.stage in ('asc-smoke','asc-train'):
        run.train_asc(cfg, smoke=args.stage == 'asc-smoke')
    elif args.stage in ('asc-smoke-eval','asc-eval'):
        run.evaluate_asc(cfg, smoke=args.stage == 'asc-smoke-eval')
    elif args.stage == 'dap':
        run.rewrite_dap(cfg)
    elif args.stage == 'tokenskip':
        run.compress_tokenskip(cfg)


if __name__ == '__main__': main()
