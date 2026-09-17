#!/bin/bash
#SBATCH -J 1_6_phase1_exact
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
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"; MIN_GPUS=3; WAIT_SECONDS="${WAIT_SECONDS:-60}"; STABLE_CHECKS=2
MAX_USED_MIB="${MAX_USED_MIB:-500}"; MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
GPU_IDS=(); select_gpus_for_approved_node "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python3 -u scripts/1_8_measure_exact_ctv_shard.py \
    --config "${CONFIG}" --shard-index "${shard}" --shard-count 3 \
    --output-dir "results/phase1_teaching_utility_v1/formal/exact_ctv/shard_0${shard}_of_03" \
    >"logs/phase1_exact_shard_${shard}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done; exit "${status}"

