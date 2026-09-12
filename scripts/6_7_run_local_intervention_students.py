#!/usr/bin/env python3
"""Conditional main teacher generation, controlled student SFT, and full evaluation."""
import argparse
import sys
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from length_budget_distill.experiment_io import read_json
from length_budget_distill.sae_local_student import freeze_main,generate_main,build,train_eval,analyze_student
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage',choices=['freeze','generate','build','train_eval','analyze'])
parser.add_argument('--config',default='configs/phase6_local_length_controlled_strength_v1.json')
parser.add_argument('--shard',type=int)
parser.add_argument('--arm')
args=parser.parse_args();config=read_json(ROOT/args.config)
if args.stage=='freeze':freeze_main(config)
elif args.stage=='generate':
    if args.shard is None or not 0<=args.shard<7:parser.error('Registered --shard 0..6 required')
    generate_main(config,args.shard)
elif args.stage=='build':build(config)
elif args.stage=='train_eval':
    if not args.arm:parser.error('--arm required')
    subprocess.run(['bash',str(ROOT/'scripts/slurm/6_7_student_gpu_admission.sh'),args.arm],check=True)
    train_eval(config,args.arm)
else:analyze_student(config)
