#!/usr/bin/env python3
"""Prepare or execute one model's MATH development cap diagnostic."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.math_cap_calibration import prepare,load,run,extend

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--stage',choices=['test','prepare','run','extend'],required=True)
    p.add_argument('--model');a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='test':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_math_cap_calibration.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():sys.exit(1)
    elif a.stage=='prepare':prepare(a.config)
    elif a.stage=='extend':extend(a.config)
    else:run(load(a.config),a.model)
