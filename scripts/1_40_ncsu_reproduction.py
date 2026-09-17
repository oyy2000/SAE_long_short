#!/usr/bin/env python3
"""Run one explicit stage of the NCSU teacher/student/SAE reproduction."""
import argparse
import logging
from length_budget_distill.ncsu_reproduction import dispatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "generate", "merge", "student", "sae_extract", "sae_sample", "sae_train", "sae_audit", "features_freeze", "features_score", "features_analyze", "finalize", "submit"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--name", default="base")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dispatch(args)
