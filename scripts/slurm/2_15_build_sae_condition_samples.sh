#!/bin/bash
#SBATCH -J 2_15_sae_samples
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=06:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1
CONFIG="${ABLATION_CONFIG:-results/phase2_sae_sampling_ablation_v1/exploratory/protocol/frozen_protocol.json}"
CONDITION="${SAE_CONDITION:?SAE_CONDITION is required}"
OUTPUT="results/phase2_sae_sampling_ablation_v1/exploratory/token_samples/${CONDITION}"
python3 -u scripts/2_15_build_sae_condition_samples.py \
  --config "${CONFIG}" --condition "${CONDITION}" --output-dir "${OUTPUT}"
