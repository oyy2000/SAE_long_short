#!/bin/bash
#SBATCH -J 1_7_phase1_analyze
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/1_11_analyze_exact_ctv.py --config "${CONFIG}" \
  --score-root results/phase1_teaching_utility_v1/formal/ctv_scores \
  --exact-root results/phase1_teaching_utility_v1/formal/exact_ctv
python3 -u scripts/1_9_build_policy_sft_data.py --config "${CONFIG}" \
  --score-root results/phase1_teaching_utility_v1/formal/ctv_scores
python3 -u scripts/1_21_build_mid_sft_anchor_data.py --config "${CONFIG}"
python3 -u scripts/1_22_build_anchor_sensitivity_candidates.py --config "${CONFIG}"

