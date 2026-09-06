#!/bin/bash

#SBATCH -J 24_3_phase0_train
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

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

RESULT_ROOT="${RESULT_ROOT:-results/trace_length_observation_gate0_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/trace_length_observation_gate0_v1}"
FROZEN_CONFIG="${FROZEN_CONFIG:-${RESULT_ROOT}/formal/protocol/frozen_protocol.json}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${RESULT_ROOT}/formal/data/dataset_manifest.json}"
LAUNCHER_SHARDS="${LAUNCHER_SHARDS:-3}"
LAUNCHER_SHARD_INDEX="${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
MIN_GPUS="${MIN_GPUS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
RUN_TAG="${RUN_TAG:-phase0_train_${LAUNCHER_SHARD_INDEX}_${SLURM_JOB_ID:-manual}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-phase0-observation/${RUN_TAG}}"

if [ ! -f "${RESULT_ROOT}/formal/data/DATA_COMPLETE" ]; then
  echo "Phase-0 data is not complete: ${RESULT_ROOT}" >&2
  exit 2
fi

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LOCAL_RUNTIME_ROOT}/adapters"
mkdir_with_retry "${RESULT_ROOT}/formal/training"
mkdir_with_retry "${CHECKPOINT_ROOT}/formal"

echo "host=$(hostname) shard=${LAUNCHER_SHARD_INDEX}/${LAUNCHER_SHARDS} min_free_mib=${MIN_FREE_MIB} min_gpus=${MIN_GPUS}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_IDS=("${GPU_IDS[@]:0:${MIN_GPUS}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

python3 -u scripts/24_3_launch_trace_observation_training.py \
  --config "${FROZEN_CONFIG}" \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --work-dir "${RESULT_ROOT}/formal/training" \
  --checkpoint-root "${CHECKPOINT_ROOT}/formal" \
  --runtime-root "${LOCAL_RUNTIME_ROOT}/adapters" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel "${MIN_GPUS}" \
  --launcher-shards "${LAUNCHER_SHARDS}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX}" \
  --skip-complete

echo "phase0_training_shard_complete shard=${LAUNCHER_SHARD_INDEX}"
