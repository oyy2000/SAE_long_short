#!/usr/bin/env python3
"""Diagnose all stored/recomputed field differences behind raw merge failure."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    from length_budget_distill.kd_math_grade_fields import run
    run(a.config)
