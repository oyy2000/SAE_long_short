#!/usr/bin/env python3
"""Verify released DAP data correspondence and prepare long-trace inputs."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.dap_paired_sources import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');prepare(a.config)
