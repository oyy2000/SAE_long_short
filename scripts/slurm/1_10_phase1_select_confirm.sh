#!/bin/bash
#SBATCH -J 1_10_phase1_select
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/1_13_select_confirmation_policies.py --config "${CONFIG}" \
  --screen-root results/phase1_teaching_utility_v1/formal/policy_evaluation/screen

