#!/bin/bash
set -euo pipefail
CODE_ROOT="${1:?code root}"
CONFIG="${2:?analysis config}"
SFT_PYTHON="${3:?Python}"
MATH_OVERLAY="${4:?math grader overlay}"
export PYTHONPATH="$MATH_OVERLAY:$CODE_ROOT/src"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
# NCSU administrator-designated scratch; never use the node system /var filesystem.
NCSU_JOB_USER="$(id -un)"
export TMPDIR="/share/jekml/${NCSU_JOB_USER}/tmp/${NCSU_JOB_USER}-phase13-analysis-${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
"$SFT_PYTHON" "$CODE_ROOT/scripts/13_9_analyze_baseline_methods.py" --config "$CONFIG"
