#!/usr/bin/env python3
"""Recover rank data, extract one residual shard, or prepare SAE samples."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill import sae_local_data as work

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=['recover', 'extract', 'sample'])
parser.add_argument('--config', default='configs/phase6_local_length_controlled_strength_v1.json')
parser.add_argument('--shard', type=int)
args = parser.parse_args()
config_path = (ROOT / args.config).resolve()
config = read_json(config_path)
if args.stage == 'recover':
    work.recover(config, config_path)
elif args.stage == 'extract':
    if args.shard is None or not 0 <= args.shard < config['activation_extraction']['trajectory_shards']:
        parser.error('extract requires a registered --shard')
    work.extract(config, args.shard)
else:
    work.sample(config)
