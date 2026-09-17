#!/usr/bin/env python3
"""Prepare, generate, or audit the fresh SAE mechanism input cohort."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill import sae_mechanism_preparation as run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','smoke','generate','merge'])
    p.add_argument('--config',required=True);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    logging.info('SAE mechanism stage=%s shard=%s config=%s code=%s',a.stage,a.shard,a.config,ROOT)
    if a.stage=='prepare':run.prepare(a.config)
    else:
        cfg=run.load_frozen(a.config)
        if a.stage=='merge':run.merge(cfg)
        else:run.generate(cfg,smoke=a.stage=='smoke',shard=a.shard)
