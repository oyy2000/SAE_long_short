#!/bin/bash
#SBATCH -J 1_19_phase1_5_analyze
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
python3 -u scripts/1_19_analyze_credit_pilot.py --config "${PHASE1_5_CONFIG}" \
  --evaluation-root results/phase1_5_credit_allocation_v1/formal/evaluation/pilot

