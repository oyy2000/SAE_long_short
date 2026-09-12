#!/usr/bin/env python3
"""Evaluate a local SAE on common support and rank features on dev questions."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_local_screen import score
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='configs/phase6_local_length_controlled_strength_v1.json')
parser.add_argument('--condition', required=True)
args = parser.parse_args()
config = read_json(ROOT/args.config)
if args.condition not in config['sampling']['conditions']:
    parser.error('Unregistered sampling condition')
score(config, args.condition)
