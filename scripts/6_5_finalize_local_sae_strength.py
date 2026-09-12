#!/usr/bin/env python3
"""Write the final docs report only after complete, audited teacher experiments."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_local_finalize import finish,preview
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/phase6_local_length_controlled_strength_v1.json')
parser.add_argument('--preview',action='store_true',help='Render completed teacher findings; no experiment completion marker')
args=parser.parse_args()
(preview if args.preview else finish)(read_json(ROOT/args.config))
