#!/usr/bin/env python3
"""Run one frozen GSM8K student-utility pilot stage."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import os

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--group',default='')
    parser.add_argument('--method',default='')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    cfg=json.loads(Path(args.config).read_text())
    if cfg.get("runtime", {}).get("storage_policy") and os.environ.get("SLURM_JOB_ID") and args.stage != "prepare":
        from length_budget_distill.pilot_storage import configure
        configure(cfg,args.stage,args.method)
    from length_budget_distill.gsm8k_pilot_runtime import dispatch
    dispatch(args)
