#!/usr/bin/env python3
"""Audit a complete teacher strength sweep and render paired dose-response plots."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from length_budget_distill.sae_intervention_sweep import analyze_strength_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyze_strength_sweep(Path(args.config).resolve(), PROJECT_ROOT)


if __name__ == "__main__":
    main()
