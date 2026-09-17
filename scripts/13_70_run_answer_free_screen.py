#!/usr/bin/env python3
"""Freeze, score, and analyze fixed-support features in the new SAE dictionary."""
import argparse
import json
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from length_budget_distill.sae_body_screen import dispatch,submit
from length_budget_distill.ncsu_reproduction import verify,save

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['freeze','score','analyze'],required=True)
    parser.add_argument('--shard',type=int);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if CODE.resolve()!=Path(cfg['code_root']).resolve():raise ValueError('Use frozen scoring source')
    dispatch(cfg,args.stage,args.shard)
    if args.stage=='freeze':
        jobs=[submit(cfg,f'score_{i:02d}','score',gpu=True,shard=i) for i in range(cfg['scoring']['shards'])]
        analysis=submit(cfg,'analyze','analyze',parent=jobs)
        save(Path(cfg['result_root'])/'protocol/submitted_jobs.json',{'score_shards':jobs,'analysis':analysis,
             'intervention_submitted':False})
