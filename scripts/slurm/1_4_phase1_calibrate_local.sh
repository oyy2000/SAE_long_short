#!/bin/bash
#SBATCH -J 1_4_phase1_calibrate
#SBATCH -N 1
#SBATCH --cpus-per-task=8
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
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"; MIN_GPUS=1; WAIT_SECONDS="${WAIT_SECONDS:-60}"; STABLE_CHECKS=2
MAX_USED_MIB="${MAX_USED_MIB:-500}"; MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
GPU_IDS=(); select_gpus_for_approved_node "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/1_5_calibrate_ctv_step.py --config "${CONFIG}"
python3 -u scripts/1_6_build_local_probe_map.py --config "${CONFIG}"

