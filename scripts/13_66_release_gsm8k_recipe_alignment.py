#!/usr/bin/env python3
"""Release the aligned matrix after measured training and evaluation admission."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.gsm8k_recipe_alignment import queue
from length_budget_distill.ncsu_reproduction import verify
from length_budget_distill.pilot_storage import configure

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['queue-aligned'],required=True)
    for field in ('round','group','shard','method','seed'):parser.add_argument('--'+field)
    args=parser.parse_args()
    code=Path(__file__).resolve().parents[1]
    verify(code.parent/'protocol/SOURCES.json')
    cfg=read_json(args.config);verify(cfg['execution_marker'])
    configure(cfg,'queue-aligned')
    queue(cfg,analysis_code_root=code)
