#!/usr/bin/env python3
"""Measure paired held-out ASC energies and mathematical-context KL."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.asc_fidelity_diagnostic import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--teacher',choices=['qwen','r1'],required=True);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run(a.config,a.teacher)
