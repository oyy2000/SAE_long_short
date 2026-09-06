#!/bin/bash
#SBATCH -J 3_3_sae_main_gen
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
# Use a colon-delimited value because commas are separators in `sbatch --export`
# and would silently truncate the shard list to its first element.
IFS=':' read -r -a SHARD_INDICES <<< "${GENERATION_SHARD_INDICES:?GENERATION_SHARD_INDICES is required}"
WORKER_COUNT="${GENERATION_WORKER_COUNT:?GENERATION_WORKER_COUNT is required}"
if [ "${#SHARD_INDICES[@]}" -lt "${WORKER_COUNT}" ]; then
  echo "Shard count must be at least worker count." >&2
  exit 2
fi
GPU_IDS=()
nvidia-smi
select_gpus_for_approved_node "${MIN_FREE_MIB:-20000}" "${WORKER_COUNT}" "${WAIT_SECONDS:-30}" 2
CONFIG="${INTERVENTION_CONFIG:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory/protocol/frozen_protocol.json}"
STAGE="${GENERATION_STAGE:-main}"
OUTPUT_ROOT="${GENERATION_OUTPUT_ROOT:?GENERATION_OUTPUT_ROOT is required}"
SHARD_COUNT="${GENERATION_SHARD_COUNT:-12}"
PIDS=()
for ((WORKER = 0; WORKER < WORKER_COUNT; WORKER++)); do
  (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[$WORKER]}"
    for ((POSITION = WORKER; POSITION < ${#SHARD_INDICES[@]}; POSITION += WORKER_COUNT)); do
      SHARD_INDEX="${SHARD_INDICES[$POSITION]}"
      python3 -u scripts/3_1_generate_sae_intervention_shard.py \
        --config "${CONFIG}" \
        --stage "${STAGE}" \
        --shard-index "${SHARD_INDEX}" \
        --shard-count "${SHARD_COUNT}" \
        --output-dir "${OUTPUT_ROOT}/shard_$(printf '%02d' "${SHARD_INDEX}")_of_$(printf '%02d' "${SHARD_COUNT}")" \
        >"${GENERATION_LOG_PREFIX:-logs/phase3_main_generation}_shard_$(printf '%02d' "${SHARD_INDEX}")-${SLURM_JOB_ID}.log" 2>&1
    done
  ) &
  PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    STATUS=1
  fi
done
exit "${STATUS}"
