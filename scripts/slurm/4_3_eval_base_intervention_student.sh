#!/bin/bash
#SBATCH -J 4_3_interv_base
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
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
CONFIG="${INTERVENTION_CONFIG:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory/protocol/frozen_protocol.json}"
OUTPUT="${BASE_EVAL_OUTPUT:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory/evaluation/base_student}"
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB:-14000}" 1 "${WAIT_SECONDS:-30}" 2
export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
python3 -u scripts/4_3_eval_intervention_student.py \
  --config "${CONFIG}" --model-id base_student --output-dir "${OUTPUT}"
