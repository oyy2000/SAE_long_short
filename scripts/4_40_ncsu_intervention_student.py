#!/usr/bin/env python3
"""Run a conditional NCSU intervention student stage."""
import argparse
import logging
from length_budget_distill.ncsu_intervention_student import dispatch
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','generate','build','student','analyze','submit'])
    p.add_argument('--config',required=True)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--name',default='base')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    dispatch(args)
