#!/usr/bin/env python3
"""Apply documented source-data review decisions to a separate derived manifest."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.baseline_data_review import review
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    print(json.dumps(review(p.parse_args().config),indent=2))
