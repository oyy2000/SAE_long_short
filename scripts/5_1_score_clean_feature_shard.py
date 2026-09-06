#!/usr/bin/env python3
"""Encode equal-count clean reasoning tokens from one immutable activation shard."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.sae_clean_features import score_clean_shard
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--shard-index',type=int,required=True);a=p.parse_args()
    score_clean_shard(Path(a.config).resolve(),ROOT,a.shard_index)
