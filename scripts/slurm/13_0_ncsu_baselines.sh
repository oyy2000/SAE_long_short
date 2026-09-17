#!/bin/bash
set -euo pipefail
CODE_ROOT="${1:?code root}"
CONFIG="${2:?config}"
SFT_PYTHON="${3:?Python}"
BASELINE_OVERLAY="${4:?baseline dependency overlay}"
STAGE="${5:?stage}"
BASELINE_CACHE_ROOT="${6:?auxiliary tokenizer cache root}"
export PYTHONPATH="$BASELINE_OVERLAY:$CODE_ROOT/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# NCSU administrator-designated scratch; never use the node system /var filesystem.
NCSU_JOB_USER="$(id -un)"
export TMPDIR="/share/jekml/${NCSU_JOB_USER}/tmp/${NCSU_JOB_USER}-phase13-${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export HF_DATASETS_CACHE="$TMPDIR/datasets" MPLCONFIGDIR="$TMPDIR/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TIKTOKEN_CACHE_DIR="$BASELINE_CACHE_ROOT/tiktoken"
export NLTK_DATA="$BASELINE_CACHE_ROOT/nltk"
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
fi
"$SFT_PYTHON" "$CODE_ROOT/scripts/13_3_run_baseline_reproduction.py" "$STAGE" --config "$CONFIG"
