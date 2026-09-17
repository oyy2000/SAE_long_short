#!/usr/bin/env python3
"""Audit and analyze the reviewed equal-count SAE seed supplement."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    from length_budget_distill.sae_generation_analysis import analyze_seed_supplement
    analyze_seed_supplement(a.config)
