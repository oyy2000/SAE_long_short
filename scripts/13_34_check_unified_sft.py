#!/usr/bin/env python3
"""Freeze synthetic inputs or check the unified baseline SFT interface."""
import argparse
import logging
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from length_budget_distill.unified_sft_smoke import prepare, smoke

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, choices=["test", "prepare", "smoke"])
    p.add_argument("--model", choices=["qwen1_5b_student", "qwen3b_student"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "test":
        suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern="test_completion_supervision.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
    elif args.stage == "prepare": prepare(args.config)
    else:
        if not args.model: p.error("--model is required for GPU smoke")
        smoke(args.config, args.model)
