#!/bin/bash
set -euo pipefail
CODE_ROOT="${1:?code root}"
CONFIG="${2:?config}"
SFT_PYTHON="${3:?Python}"
DEPENDENCY_OVERLAY="${4:?dependency overlay}"
ENTRYPOINT="${5:?script name}"
shift 5
[[ "$ENTRYPOINT" != */* && -f "$CODE_ROOT/scripts/$ENTRYPOINT" ]]
export PYTHONPATH="$DEPENDENCY_OVERLAY:$CODE_ROOT/src"
# Establish scratch before starting any Python process; inherited overrides cannot
# redirect caches or Trainer checkpoints to the node system filesystem.
NCSU_JOB_USER="$(id -un)"
NCSU_SCRATCH_ROOT="/share/jekml/${NCSU_JOB_USER}/tmp"
[[ -d "$NCSU_SCRATCH_ROOT" && -w "$NCSU_SCRATCH_ROOT" ]] || { echo "Designated NCSU scratch unavailable: $NCSU_SCRATCH_ROOT" >&2; exit 1; }
export TMPDIR="$NCSU_SCRATCH_ROOT/${NCSU_JOB_USER}-phase13-frozen-${SLURM_JOB_ID:?Slurm job required}"
export TMP="$TMPDIR" TEMP="$TMPDIR" HF_DATASETS_CACHE="$TMPDIR/datasets" MPLCONFIGDIR="$TMPDIR/matplotlib"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export TRITON_CACHE_DIR="$TMPDIR/triton" NUMBA_CACHE_DIR="$TMPDIR/numba"
export CUDA_CACHE_PATH="$TMPDIR/cuda" JOBLIB_TEMP_FOLDER="$TMPDIR/joblib"
unset LBD_RUNTIME_OUTPUT_DIR LBD_PILOT_STORAGE_EVIDENCE
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR" "$CUDA_CACHE_PATH" "$JOBLIB_TEMP_FOLDER"
[[ "$(readlink -f "$TMPDIR")" == "$(readlink -f "$NCSU_SCRATCH_ROOT")/${NCSU_JOB_USER}-phase13-frozen-${SLURM_JOB_ID}" ]] || { echo "Job scratch escaped designated root" >&2; exit 1; }
echo "NCSU scratch: job=$SLURM_JOB_ID TMPDIR=$TMPDIR"
# Opt-in ABI runtime; legacy frozen launchers and configurations remain unchanged.
PILOT_NATIVE_PRELOAD="$("$SFT_PYTHON" -c 'import json,sys; print(":".join(json.load(open(sys.argv[1])).get("runtime",{}).get("native_preload",[])))' "$CONFIG")"
if [[ -n "$PILOT_NATIVE_PRELOAD" ]]; then
    export LD_PRELOAD="$PILOT_NATIVE_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
fi
"$SFT_PYTHON" "$CODE_ROOT/scripts/$ENTRYPOINT" --config "$CONFIG" "$@"
