#!/bin/bash

#SBATCH -J 24_1_phase0_data
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${FACT_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
cd /home/youyang7/projects/SAE_long_short

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/trace_length_observation_gate0_v1.json}"
RESULT_ROOT="${RESULT_ROOT:-results/trace_length_observation_gate0_v1}"
FROZEN_DIR="${RESULT_ROOT}/formal/protocol"
FROZEN_CONFIG="${FROZEN_DIR}/frozen_protocol.json"

if [ ! -f "${FROZEN_DIR}/PROTOCOL_FROZEN" ]; then
  python3 -u scripts/24_0_freeze_trace_observation_protocol.py \
    --config "${TEMPLATE_CONFIG}" \
    --output-dir "${FROZEN_DIR}"
fi

python3 -u scripts/24_1_build_trace_observation_data.py \
  --config "${FROZEN_CONFIG}" \
  --stage formal \
  --output-dir "${RESULT_ROOT}/formal/data"

echo "phase0_data_complete result_root=${RESULT_ROOT}"
