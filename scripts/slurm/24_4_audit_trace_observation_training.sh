#!/bin/bash

#SBATCH -J 24_4_phase0_train_audit
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${FACT_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
cd /home/youyang7/projects/SAE_long_short

RESULT_ROOT="${RESULT_ROOT:-results/trace_length_observation_gate0_v1}"
FROZEN_CONFIG="${FROZEN_CONFIG:-${RESULT_ROOT}/formal/protocol/frozen_protocol.json}"

python3 -u scripts/24_4_audit_trace_observation_training.py \
  --config "${FROZEN_CONFIG}" \
  --dataset-manifest "${RESULT_ROOT}/formal/data/dataset_manifest.json" \
  --training-dir "${RESULT_ROOT}/formal/training" \
  --output-dir "${RESULT_ROOT}/formal/training/audit"

echo "phase0_training_audit_complete result_root=${RESULT_ROOT}"
