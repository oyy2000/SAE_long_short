#!/usr/bin/env python3
"""Run one TokenSkip author-data SFT or evaluation stage."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill import tokenskip_reproduction as run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke-train','smoke-eval','train','evaluate'])
    p.add_argument('--config',required=True)
    p.add_argument('--model',choices=['base','author','replica'],default='replica')
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    logging.info('TokenSkip stage=%s model=%s config=%s code=%s',a.stage,a.model,a.config,ROOT)
    if a.stage=='prepare':run.prepare(a.config)
    else:
        cfg=run.load_frozen(a.config)
        if a.stage in ('train','smoke-train'):run.train_student(cfg,smoke=a.stage=='smoke-train')
        else:run.evaluate_student(cfg,a.model,smoke=a.stage=='smoke-eval')
