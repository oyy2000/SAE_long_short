#!/usr/bin/env python3
"""Execute one explicit stage of the fixed-cohort multiple-answer experiment."""
import argparse
import logging
from length_budget_distill.ncsu_multi_answer import dispatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'supplement', 'build', 'student', 'analyze', 'audit', 'submit'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--name', default='base')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    dispatch(args)
