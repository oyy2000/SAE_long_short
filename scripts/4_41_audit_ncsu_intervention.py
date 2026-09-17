#!/usr/bin/env python3
"""Audit completed NCSU intervention artifacts without modifying frozen results."""
import argparse
import logging
from length_budget_distill.experiment_io import read_json
from length_budget_distill.ncsu_intervention_student import audit_delivery

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    report=audit_delivery(read_json(args.config),args.output)
    print({k:report[k] for k in ('status','unique_hashed_files','verified_markers','main_generation_records','adapter_count','evaluation_runs','evaluation_records')})
