#!/usr/bin/env python3
"""Prepare a masked prose-answer queue or finalize its versioned decisions."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.gsm8k_answer_review import prepare,finalize
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['prepare','finalize'],required=True);p.add_argument('--decisions')
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='prepare':prepare(a.config)
    else:
        if not a.decisions:p.error('--decisions is required for finalization')
        finalize(a.config,a.decisions)
