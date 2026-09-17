#!/usr/bin/env python3
"""Test or submit the bounded main-pool compression and SFT preparation DAG."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['test','submit'],required=True);a=p.parse_args()
    if a.stage=='test':
        import unittest
        from length_budget_distill.experiment_io import read_json
        from length_budget_distill.ncsu_reproduction import seal,verify
        cfg=read_json(a.config);launch=Path(cfg['launch_root']);verify(launch/'FROZEN.json')
        suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern)
            for pattern in ('test_unified_pipeline_launch.py','test_unified_student_distillation.py'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():sys.exit(1)
        seal(launch/'TEST_COMPLETE.json',[Path(a.config),launch/'FROZEN.json'],tests_run=result.testsRun,job_id=int(os.environ['SLURM_JOB_ID']))
    else:
        from length_budget_distill.unified_pipeline_launch import submit
        submit(a.config)
