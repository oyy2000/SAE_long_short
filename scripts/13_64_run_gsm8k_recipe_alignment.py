#!/usr/bin/env python3
"""Run a separately frozen GSM8K optimization-recipe sensitivity stage."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from length_budget_distill.gsm8k_recipe_alignment import dispatch

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--method',default='')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--group',default='')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    cfg=json.loads(Path(args.config).read_text())
    if os.environ.get('SLURM_JOB_ID') and cfg.get('runtime',{}).get('storage_policy'):
        from length_budget_distill.pilot_storage import configure
        configure(cfg,args.stage,args.method)
    dispatch(args)
