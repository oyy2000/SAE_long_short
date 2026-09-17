#!/usr/bin/env python3
"""Materialize and audit the reference 84-cell matrix without GPU submission."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.baseline_plan import write_matrix
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();print(write_matrix(ROOT/args.config,ROOT))
