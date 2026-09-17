#!/usr/bin/env python3
"""Execute one registered NCSU SAE intervention stage."""
import argparse
import logging
from length_budget_distill.ncsu_intervention import dispatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'generate', 'analyze', 'submit'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    dispatch(args)
