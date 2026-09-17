#!/usr/bin/env python3
"""Review pilot state hourly with the registered small, read-only subagent."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from length_budget_distill.gsm8k_pilot_monitor import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--launch-root',required=True);p.add_argument('--monitor-root',required=True)
    p.add_argument('--once',action='store_true');p.add_argument('--initial-delay',type=int,default=3600)
    a=p.parse_args();run(a.config,a.launch_root,a.monitor_root,once=a.once,initial_delay=a.initial_delay)
