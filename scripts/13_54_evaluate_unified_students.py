#!/usr/bin/env python3
"""Freeze, generate or merge the real student's full benchmark cohort."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from length_budget_distill.unified_student_benchmarks import prepare, run, merge, audit_benchmark_cohort, COUNTS
from length_budget_distill.experiment_io import read_json
from length_budget_distill.records import read_jsonl
from length_budget_distill.ncsu_reproduction import verify, save, seal

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['test', 'prepare', 'generate', 'merge'], required=True)
    parser.add_argument('--student'); parser.add_argument('--method', default='base')
    parser.add_argument('--seed', type=int); parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'test':
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_unified_student_benchmarks.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        cfg = read_json(args.config); cohorts = Path(cfg['cohort_root']); launch = Path(cfg['launch_root'])
        verify(launch/'FROZEN.json'); verify(cohorts/'COMPLETE.json')
        paths = [cohorts/'evaluation'/(name+'.jsonl') for name in COUNTS]
        questions = [q for path in paths for q in read_jsonl(path)]
        audit_benchmark_cohort(questions)
        out = launch/'test_validation'; out.mkdir(exist_ok=False)
        save(out/'summary.json', {'tests': suite.countTestCases(), 'actual_locked_questions': len(questions),
             'benchmark_counts': COUNTS, 'test_predictions_generated': 0, 'formal_evaluation_complete': False})
        seal(out/'COMPLETE.json', [Path(args.config), launch/'FROZEN.json', cohorts/'COMPLETE.json',
             *paths, out/'summary.json'], stage='student_benchmark_worker_tests_and_real_cohort_validation',
             formal_evaluation_complete=False)
        logging.info('Checked all %d locked evaluation records without generating answers', len(questions))
    elif args.stage == 'prepare': prepare(args.config)
    else:
        if args.student is None: parser.error('Specify a registered student')
        if args.stage == 'generate':
            if args.shard is None: parser.error('Generation requires a registered shard')
            run(args.config, args.student, args.method, args.seed, args.shard)
        else: merge(args.config, args.student, args.method, args.seed)
