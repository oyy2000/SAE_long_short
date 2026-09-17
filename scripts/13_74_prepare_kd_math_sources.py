#!/usr/bin/env python3
"""Merge immutable raw MATH shards or audit the reviewed scoring changes."""
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',required=True,choices=['merge','impact']);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    from length_budget_distill.kd_math_preparation import run
    run(a.config,a.stage)
