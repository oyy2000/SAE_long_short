#!/usr/bin/env python3
"""Audit the new typed grader on complete registered mathematics cohorts."""
import argparse,json,logging,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.typed_math_audit import audit
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    print(json.dumps(audit(p.parse_args().config),indent=2))
