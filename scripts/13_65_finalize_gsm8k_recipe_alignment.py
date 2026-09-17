#!/usr/bin/env python3
"""Finalize aligned predictions with the evaluation-only revision hash bound."""
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.gsm8k_recipe_alignment import analyze
from length_budget_distill.ncsu_reproduction import verify
from length_budget_distill.pilot_storage import configure

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['analyze-aligned'],required=True)
    for field in ('round','group','shard','method','seed'):parser.add_argument('--'+field)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    cfg=read_json(args.config)
    verify(Path(__file__).resolve().parents[2]/'protocol/SOURCES.json')
    verify(cfg['execution_marker'])
    configure(cfg,'analyze-aligned')
    analyze(cfg)
