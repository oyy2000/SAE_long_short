#!/usr/bin/env python3
"""Audit paired long/short DAP outputs and write a comparison figure."""
import argparse
import logging
from length_budget_distill.dap_long_trace_check import analyze

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    analyze(args.config)
