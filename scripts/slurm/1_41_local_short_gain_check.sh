#!/bin/bash
# Run inside an existing C31 allocation; independent arms use separate GPUs.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
source scripts/slurm/_gpu_idle_gate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export GPU_ADMISSION_POLICY=memory_fit OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
GAIN_CONFIG=configs/phase1_local_short_gain_check_v1.json
GAIN_RESULT=results/phase1_local_short_gain_check_v1
GAIN_RUNTIME=/mnt/local/youyang7/SAE_long_short_c31/runs/phase1_local_short_gain_check_v1
mkdir -p "$GAIN_RUNTIME/tmp" "$GAIN_RUNTIME/datasets"
export TMPDIR="$GAIN_RUNTIME/tmp" HF_DATASETS_CACHE="$GAIN_RUNTIME/datasets"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$GAIN_RESULT/logs/gpu_processes_before.csv"
GPU_IDS=()
select_gpus_for_approved_node 18000 3 3 2
nvidia-smi > "$GAIN_RESULT/logs/nvidia_smi_before.txt"
arms=(base local_smoke short__seed_17)
pids=()
for index in 0 1 2; do
  arm="${arms[$index]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/1_41_check_local_short_gain.py worker \
    --config "$GAIN_CONFIG" --arm "$arm" > "$GAIN_RESULT/logs/${arm}.log" 2>&1 &
  pids+=("$!")
  printf 'arm=%s gpu=%s pid=%s\n' "$arm" "${GPU_IDS[$index]}" "${pids[$index]}"
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
  echo 'One or more arms failed; inspect logs. No aggregate completion marker written.' >&2
  exit 1
fi
"$SAE_EXPERIMENT_PYTHON" scripts/1_41_check_local_short_gain.py analyze --config "$GAIN_CONFIG"
# This directory contains only disposable files of the new experiment.
rm -rf -- "$GAIN_RUNTIME"
