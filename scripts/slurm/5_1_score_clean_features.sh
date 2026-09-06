#!/bin/bash
#SBATCH -J 5_1_clean_sae
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/beegfs/youyang7/.conda/envs/sft
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_gpu_idle_gate.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
CONFIG="${CLEAN_CONFIG:?CLEAN_CONFIG is required}"
STAGE="${CLEAN_STAGE:-score}"
EXTRA_ARGS=()
case "${STAGE}" in
  score) ENTRY=scripts/5_1_score_clean_feature_shard.py; SLOT_ARGUMENT=--shard-index ;;
  engagement) ENTRY=scripts/5_4_measure_target_engagement.py; SLOT_ARGUMENT=--feature-slot; EXTRA_ARGS=(--version "${ENGAGEMENT_VERSION:-1}") ;;
  *) echo "Unsupported CLEAN_STAGE=${STAGE}" >&2; exit 2 ;;
esac
GPU_IDS=()
nvidia-smi
select_gpus_for_approved_node "${MIN_FREE_MIB:-10000}" 3 10 2
PIDS=()
for SHARD in 0 1 2; do
  nvidia-smi -i "${GPU_IDS[$SHARD]}" --query-gpu=index,memory.free --format=csv,noheader
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$SHARD]}" python "${ENTRY}" --config "${CONFIG}" "${SLOT_ARGUMENT}" "${SHARD}" "${EXTRA_ARGS[@]}" >"logs/phase5_clean_${STAGE}${ENGAGEMENT_VERSION:-}_${SHARD}-${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then STATUS=1; fi
done
exit "${STATUS}"
