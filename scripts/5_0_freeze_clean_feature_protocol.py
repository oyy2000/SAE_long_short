#!/usr/bin/env python3
"""Freeze fixed-token feature cleaning and conditional causal validation."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.sae_clean_features import freeze_clean_protocol
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    freeze_clean_protocol(Path(a.config).resolve(),ROOT)
