#!/usr/bin/env python3
"""Archive scratch from explicitly registered terminal SAE seed jobs."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['archive'],required=True);a=p.parse_args()
    from length_budget_distill.sae_seed_recovery import archive
    archive(a.config)
