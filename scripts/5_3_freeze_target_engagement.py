#!/usr/bin/env python3
"""Freeze the primary clean feature and matched random controls before readback."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.sae_target_engagement import freeze_engagement
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--version',type=int,choices=[1,2],default=1);a=p.parse_args();freeze_engagement(Path(a.config).resolve(),ROOT,a.version)
