#!/bin/bash
# Supplemental runtime wrapper: preserves the original frozen extraction source.
set -euo pipefail
CODE_ROOT="${1:?frozen code root}"
CONFIG="${2:?frozen main config}"
SFT_PYTHON="${3:?Python}"
SFT_OVERLAY="${4:?TRL overlay}"
ENTRY="${5:?math SDPA entrypoint}"
SHARD="${6:?shard index}"
export PYTHONPATH="$SFT_OVERLAY:$CODE_ROOT/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# NCSU administrator-designated scratch; never use the node system /var filesystem.
NCSU_JOB_USER="$(id -un)"
export TMPDIR="/share/jekml/${NCSU_JOB_USER}/tmp/${NCSU_JOB_USER}-ncsu-sae-math-${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" HF_DATASETS_CACHE="$TMPDIR/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"$SFT_PYTHON" "$ENTRY" --config "$CONFIG" --shard-index "$SHARD"
