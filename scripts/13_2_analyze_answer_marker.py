#!/usr/bin/env python3
"""Audit Answer-region activation using saved SAE sparse events; CPU only."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.answer_marker_diagnostic import run_diagnostic
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    result=run_diagnostic(ROOT/args.config,ROOT)
    print(json.dumps({k:result[k] for k in ['status','traces','problems','short_answer_mass_share_range','short_top_token_all_answer']},indent=2))
