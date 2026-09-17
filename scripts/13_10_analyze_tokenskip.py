#!/usr/bin/env python3
"""Publish audited base, author, and trained-replica TokenSkip comparisons."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.tokenskip_analysis import analyze
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    result=analyze(p.parse_args().config)
    print(json.dumps({'status':result['status'],'audited_predictions':result['audited_predictions']}))
