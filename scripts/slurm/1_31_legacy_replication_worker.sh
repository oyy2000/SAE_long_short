#!/bin/bash
#SBATCH --job-name=1_31_legacy_repl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_gpu_idle_gate.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GPU_ADMISSION_POLICY=memory_fit
LEGACY_RUNTIME=$(mktemp -d /var/tmp/youyang7-legacy-repl-${SLURM_JOB_ID}-XXXXXX)
export TMPDIR=${LEGACY_RUNTIME}/tmp
export HF_DATASETS_CACHE=${LEGACY_RUNTIME}/datasets
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
GPU_IDS=()
select_gpus_for_approved_node 18000 1 30 2
LEGACY_GPUS=$(IFS=,; echo "${GPU_IDS[*]}")
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -u scripts/1_31_legacy_replication.py worker \
  --config "${LEGACY_CONFIG:?}" --shard "${LEGACY_SHARD:?}" --shards "${LEGACY_SHARDS:?}" \
  --gpu-ids "$LEGACY_GPUS" --runtime "$LEGACY_RUNTIME"
