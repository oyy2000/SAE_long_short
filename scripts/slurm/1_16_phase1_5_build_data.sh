#!/bin/bash
#SBATCH -J 1_16_phase1_5_data
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
CONFIG="${PHASE1_5_CONFIG:-results/phase1_5_credit_allocation_v1/formal/protocol/frozen_protocol.json}"
UTILITY_CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/1_16_build_credit_allocation_data.py --config "${CONFIG}" --utility-config "${UTILITY_CONFIG}" \
  --step-score-root results/phase1_5_credit_allocation_v1/formal/step_scores

