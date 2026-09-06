#!/bin/bash
#SBATCH -J 2_4_sae_train
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/SAE_long_short/scripts/slurm/_gpu_idle_gate.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"; MIN_GPUS="${MIN_GPUS:-2}"; WAIT_SECONDS="${WAIT_SECONDS:-60}"; STABLE_CHECKS=2
MAX_USED_MIB="${MAX_USED_MIB:-500}"; MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
GPU_IDS=(); select_gpus_for_approved_node "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_CSV=$(IFS=,; echo "${GPU_IDS[*]:0:2}")
CONFIG="${PHASE2_CONFIG:-results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/2_5_launch_sae_training.py --config "${CONFIG}" --gpu-ids "${GPU_CSV}" --skip-complete \
  --matrix-shard-index "${MATRIX_SHARD_INDEX:-0}" --matrix-shard-count "${MATRIX_SHARD_COUNT:-1}"
