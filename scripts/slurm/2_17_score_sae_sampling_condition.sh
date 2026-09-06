#!/bin/bash
#SBATCH -J 2_17_sae_score
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
export GPU_ADMISSION_POLICY="${GPU_ADMISSION_POLICY:-memory_fit}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB}" 1 "${WAIT_SECONDS:-30}" 2
export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
CONFIG="${ABLATION_CONFIG:-results/phase2_sae_sampling_ablation_v1/exploratory/protocol/frozen_protocol.json}"
CONDITION="${SAE_CONDITION:?SAE_CONDITION is required}"
python3 -u scripts/2_17_score_sae_sampling_condition.py \
  --config "${CONFIG}" \
  --condition "${CONDITION}"
