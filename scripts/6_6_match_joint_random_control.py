#!/usr/bin/env python3
"""Freeze, run, and audit the structure-matched random control addendum."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_joint_control import freeze,generate,analyze
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=['freeze','generate','analyze'])
parser.add_argument('--config',default='configs/phase6_local_matched_joint_control_v1.json')
parser.add_argument('--shard',type=int)
args=parser.parse_args();addon=read_json(ROOT/args.config);config=read_json(ROOT/addon['parent_config'])
if args.stage=='freeze':freeze(config,addon,ROOT/args.config)
elif args.stage=='analyze':analyze(config)
else:
    if args.shard is None or not 0<=args.shard<addon['shards']:parser.error('A registered --shard is required')
    generate(config,args.shard)
