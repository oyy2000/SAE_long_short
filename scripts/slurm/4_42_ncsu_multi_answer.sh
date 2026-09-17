#!/bin/bash
set -euo pipefail
CODE_ROOT="${1:?code root}"
CONFIG="${2:?config}"
SFT_PYTHON="${3:?Python}"
SFT_OVERLAY="${4:?overlay}"
shift 4
export PYTHONPATH="$SFT_OVERLAY:$CODE_ROOT/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# NCSU administrator-designated scratch; never use the node system /var filesystem.
NCSU_JOB_USER="$(id -un)"
export TMPDIR="/share/jekml/${NCSU_JOB_USER}/tmp/${NCSU_JOB_USER}-ncsu-multi-answer-${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export HF_DATASETS_CACHE="$TMPDIR/datasets" MPLCONFIGDIR="$TMPDIR/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
fi
"$SFT_PYTHON" "$CODE_ROOT/scripts/4_42_ncsu_multi_answer.py" "$@" --config "$CONFIG"
