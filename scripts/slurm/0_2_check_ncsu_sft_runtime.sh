#!/bin/bash
# Supply Slurm resource requests, environment Python, and result JSON explicitly.
set -euo pipefail
SFT_PYTHON="${1:?absolute environment Python path is required}"
OUTPUT_JSON="${2:?output JSON path is required}"
cd "${SLURM_SUBMIT_DIR:?}"
bash scripts/slurm/0_1_ncsu_gpu_probe.sh
# Respect Slurm's assigned CUDA_VISIBLE_DEVICES; never scan for extra GPUs.
# NCSU administrator-designated scratch; never use the node system /var filesystem.
NCSU_JOB_USER="$(id -un)"
export TMPDIR="/share/jekml/${NCSU_JOB_USER}/tmp/${NCSU_JOB_USER}-sft-check-${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" MPLCONFIGDIR="$TMPDIR/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
"$SFT_PYTHON" scripts/0_2_check_ncsu_sft_runtime.py --output "$OUTPUT_JSON"
