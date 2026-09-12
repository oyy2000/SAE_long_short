#!/usr/bin/env python3
"""Freeze analysis rules, audit a teacher cohort, or draw standalone figures."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_local_report import freeze_analysis,analyze,plots
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=['freeze','analyze','plots'])
parser.add_argument('--config',default='configs/phase6_local_length_controlled_strength_v1.json')
parser.add_argument('--split',choices=['dev','test'])
args=parser.parse_args();config=read_json(ROOT/args.config)
if args.stage=='freeze':freeze_analysis(config)
elif args.stage=='plots':plots(config)
else:
    if args.split is None:parser.error('analyze requires --split')
    analyze(config,args.split)
