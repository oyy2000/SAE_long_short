#!/bin/bash
#SBATCH -J 1_14_phase1_final
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/1_14_analyze_policy_sft.py --config "${CONFIG}" \
  --test-root results/phase1_teaching_utility_v1/formal/policy_evaluation/confirm

