#!/usr/bin/env python3
"""Audit pinned metadata, reference structure and overlap for the later 25K pool."""
import argparse
import logging
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['test', 'run'], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'test':
        import unittest
        from length_budget_distill.experiment_io import read_json
        from length_budget_distill.ncsu_reproduction import seal, verify
        cfg = read_json(args.config)
        launch = Path(cfg['launch_root'])
        verify(launch/'FROZEN.json')
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_openthoughts_inventory.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): sys.exit(1)
        seal(launch/'TEST_COMPLETE.json', [Path(args.config), launch/'FROZEN.json'],
             tests_run=result.testsRun, job_id=int(os.environ['SLURM_JOB_ID']))
    else:
        from length_budget_distill.openthoughts_inventory import run
        run(args.config)
