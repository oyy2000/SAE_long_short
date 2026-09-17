#!/usr/bin/env python3
"""Verify complete GSM pilot artifacts and recompute aggregate metrics."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from length_budget_distill.gsm8k_pilot_summary import audit
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    audit(parser.parse_args().config)
