#!/usr/bin/env python3
"""Run one frozen small KD stage with matched SFT controls."""
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',required=True,choices=['test','prepare','smoke','launch-pilot','cell','train','evaluate','analyze'])
    parser.add_argument('--arm',default='kd',choices=['sft','kd']);parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    from length_budget_distill.token_kd_pilot import dispatch
    dispatch(args)
