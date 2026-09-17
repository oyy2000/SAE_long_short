#!/bin/bash
#SBATCH -J 1_21_phase1_5_eval
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
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"; MIN_GPUS=3; WAIT_SECONDS="${WAIT_SECONDS:-60}"; STABLE_CHECKS=2
MAX_USED_MIB="${MAX_USED_MIB:-500}"; MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
GPU_IDS=(); select_gpus_for_approved_node "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_CSV=$(IFS=,; echo "${GPU_IDS[*]:0:3}")
python3 -u scripts/1_26_launch_credit_evaluation.py --stage confirm --gpu-ids "${GPU_CSV}" \
  --config "${PHASE1_5_CONFIG}" --utility-config "${PHASE1_CONFIG}"

