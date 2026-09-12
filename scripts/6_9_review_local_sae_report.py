#!/usr/bin/env python3
"""Publish the presentation-reviewed report after the complete experiment audit."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_local_presentation_review import review
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/phase6_local_length_controlled_strength_v1.json')
args=parser.parse_args();review(read_json(ROOT/args.config))
