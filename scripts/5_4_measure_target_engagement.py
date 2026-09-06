#!/usr/bin/env python3
"""Measure target and off-target SAE code changes for one registered feature."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.sae_target_engagement import run_engagement
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--feature-slot',type=int,required=True);p.add_argument('--version',type=int,choices=[1,2],default=1);a=p.parse_args();run_engagement(Path(a.config).resolve(),ROOT,a.feature_slot,a.version)
