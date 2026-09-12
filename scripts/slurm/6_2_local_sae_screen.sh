#!/bin/bash
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
source scripts/slurm/_gpu_idle_gate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit
SCREEN_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$SCREEN_RESULT/logs/processes_before_screen.csv"
GPU_IDS=()
select_gpus_for_approved_node 12000 3 3 2
nvidia-smi > "$SCREEN_RESULT/logs/gpus_before_screen.txt"
conditions=(full_token_uniform full_trace_balanced prefix64_trace_balanced)
pids=()
for index in 0 1 2; do
  condition="${conditions[$index]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/6_2_score_local_length_sae.py --condition "$condition" > "$SCREEN_RESULT/logs/screen_${condition}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"
