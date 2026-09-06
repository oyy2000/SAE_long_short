#!/usr/bin/env python3
"""Audit clean token codes and apply the frozen feature-selection gates."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.sae_clean_analysis import analyze_clean_features
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    analyze_clean_features(Path(a.config).resolve(),ROOT)
