#!/bin/bash
#SBATCH --job-name=1_32_repl_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --oversubscribe
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -u scripts/1_31_legacy_replication.py analyze --config "${LEGACY_CONFIG:?}"
