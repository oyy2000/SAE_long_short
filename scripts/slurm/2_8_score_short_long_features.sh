#!/bin/bash
#SBATCH -J 2_8_sae_features
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
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
export GPU_ADMISSION_POLICY="${GPU_ADMISSION_POLICY:-memory_fit}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
MIN_GPUS="${MIN_GPUS:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
CONFIG="${FEATURE_CONFIG:-results/phase2_short_long_feature_analysis_v1/exploratory/protocol/frozen_protocol.json}"
python3 -u scripts/2_9_launch_short_long_feature_scoring.py \
  --config "${CONFIG}" \
  --gpu-ids "${GPU_CSV}" \
  --matrix-shard-index "${MATRIX_SHARD_INDEX:?MATRIX_SHARD_INDEX is required}" \
  --matrix-shard-count "${MATRIX_SHARD_COUNT:-4}" \
  --skip-complete
