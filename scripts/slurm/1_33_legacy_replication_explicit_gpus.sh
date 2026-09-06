#!/bin/bash
#SBATCH --job-name=1_33_legacy_repl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --oversubscribe
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LEGACY_RUNTIME=$(mktemp -d /var/tmp/youyang7-legacy-repl-${SLURM_JOB_ID}-XXXXXX)
export TMPDIR=${LEGACY_RUNTIME}/tmp HF_DATASETS_CACHE=${LEGACY_RUNTIME}/datasets
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
# Explicit disjoint physical IDs avoid sharing a GPU with another replication
# worker. The common worker rechecks memory capacity twice before each launch.
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -u scripts/1_31_legacy_replication.py worker \
  --config "${LEGACY_CONFIG:?}" --shard "${LEGACY_SHARD:?}" --shards "${LEGACY_SHARDS:?}" \
  --gpu-ids "${LEGACY_GPU_IDS:?}" --runtime "$LEGACY_RUNTIME"
