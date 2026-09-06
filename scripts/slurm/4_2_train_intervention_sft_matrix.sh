#!/bin/bash
#SBATCH -J 4_2_interv_sft
#SBATCH -N 1
#SBATCH --cpus-per-task=16
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
RESULT_ROOT="${RESULT_ROOT:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory}"
CONFIG="${INTERVENTION_CONFIG:-${RESULT_ROOT}/protocol/frozen_protocol.json}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${RESULT_ROOT}/sft_data/sft_data_manifest.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/phase3_sae_intervention_distillation_pilot_v1}"
LAUNCHER_SHARDS="${LAUNCHER_SHARDS:-3}"
LAUNCHER_SHARD_INDEX="${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}"
GPU_COUNT="${GPU_COUNT:?GPU_COUNT is required}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-sae-intervention-sft/${SLURM_JOB_ID}}"
test -f "${RESULT_ROOT}/sft_data/SFT_DATA_COMPLETE"
mkdir -p "${LOCAL_RUNTIME_ROOT}/adapters" "${LOCAL_RUNTIME_ROOT}/tmp"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB:-18000}" "${GPU_COUNT}" "${WAIT_SECONDS:-30}" 2
GPU_IDS=("${GPU_IDS[@]:0:${GPU_COUNT}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
python3 -u scripts/4_2_launch_intervention_sft_matrix.py \
  --config "${CONFIG}" \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --work-dir "${RESULT_ROOT}/training" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --runtime-root "${LOCAL_RUNTIME_ROOT}/adapters" \
  --gpu-ids "${GPU_ID_CSV}" \
  --launcher-shards "${LAUNCHER_SHARDS}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX}" \
  --skip-complete
