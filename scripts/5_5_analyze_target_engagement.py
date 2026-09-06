#!/usr/bin/env python3
"""Audit state-level target engagement and enforce the generation gate."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.sae_engagement_analysis import analyze_engagement
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--version',type=int,choices=[1,2],required=True);p.add_argument('--finalize-failure',action='store_true');a=p.parse_args();analyze_engagement(Path(a.config).resolve(),ROOT,a.version,a.finalize_failure)
