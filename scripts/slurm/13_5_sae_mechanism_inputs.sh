#!/bin/bash
set -euo pipefail
CODE_ROOT="${1:?code root}"
CONFIG="${2:?config}"
SFT_PYTHON="${3:?Python}"
MATH_OVERLAY="${4:?math overlay}"
shift 4
export PYTHONPATH="$MATH_OVERLAY:$CODE_ROOT/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS=1
# NCSU administrator-designated scratch; never use the node system /var filesystem.
NCSU_JOB_USER="$(id -un)"
export TMPDIR="/share/jekml/${NCSU_JOB_USER}/tmp/${NCSU_JOB_USER}-phase13-sae-mechanism-${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" HF_DATASETS_CACHE="$TMPDIR/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
fi
"$SFT_PYTHON" "$CODE_ROOT/scripts/13_13_prepare_sae_mechanism.py" "$@" --config "$CONFIG"
