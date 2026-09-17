#!/usr/bin/env python3
"""Run DAP/TokenSkip on frozen common-teacher mathematics candidates."""
import argparse
import logging
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['test','prepare','dap','tokenskip','merge'],required=True)
    p.add_argument('--cohort',default='smoke',choices=['smoke','development','student_pool']);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='test':
        import unittest
        suite=unittest.TestSuite()
        for pattern in ('test_compression_baselines.py','test_unified_math_candidates.py','test_unified_text_compression.py',
                        'test_reviewed_math_candidate_regrading.py','test_reviewed_math_grading.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():sys.exit(1)
    else:
        from length_budget_distill import unified_text_compression as run
        if a.stage=='prepare':run.prepare(a.config)
        else:
            cfg=run.load(a.config)
            if a.stage=='merge':run.merge(cfg,a.cohort)
            else:getattr(run,'run_'+a.stage)(cfg,a.cohort,a.shard)
