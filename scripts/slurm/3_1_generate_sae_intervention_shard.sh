#!/bin/bash
#SBATCH -J 3_1_sae_intervene
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/SAE_long_short/scripts/slurm/_gpu_idle_gate.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export GPU_ADMISSION_POLICY="${GPU_ADMISSION_POLICY:-memory_fit}"
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB:-20000}" 1 "${WAIT_SECONDS:-30}" 2
export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
CONFIG="${INTERVENTION_CONFIG:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory/protocol/frozen_protocol.json}"
STAGE="${GENERATION_STAGE:?GENERATION_STAGE is required}"
SHARD_INDEX="${GENERATION_SHARD_INDEX:?GENERATION_SHARD_INDEX is required}"
SHARD_COUNT="${GENERATION_SHARD_COUNT:?GENERATION_SHARD_COUNT is required}"
OUTPUT_ROOT="${GENERATION_OUTPUT_ROOT:?GENERATION_OUTPUT_ROOT is required}"
python3 -u scripts/3_1_generate_sae_intervention_shard.py \
  --config "${CONFIG}" \
  --stage "${STAGE}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count "${SHARD_COUNT}" \
  --output-dir "${OUTPUT_ROOT}/shard_$(printf '%02d' "${SHARD_INDEX}")_of_$(printf '%02d' "${SHARD_COUNT}")"
