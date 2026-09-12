#!/usr/bin/env python3
"""Freeze or run one shard of measured relative-norm SAE interventions."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_norm_intervention import freeze_generation,generate_shard
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=['freeze','generate'])
parser.add_argument('--config',default='configs/phase6_local_length_controlled_strength_v1.json')
parser.add_argument('--split',choices=['dev','test'])
parser.add_argument('--shard',type=int)
args=parser.parse_args()
config=read_json(ROOT/args.config)
if args.stage=='freeze': freeze_generation(config)
else:
    if args.split is None or args.shard is None or not 0<=args.shard<config['generation']['shards']:
        parser.error('generate requires --split and registered --shard')
    generate_shard(config,args.split,args.shard)
