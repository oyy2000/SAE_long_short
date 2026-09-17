#!/usr/bin/env python3
"""Publish corrected ASC calibration pairs in a separate frozen branch."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.baseline_grading_repair import repair
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    print(json.dumps(repair(a.config),indent=2))
