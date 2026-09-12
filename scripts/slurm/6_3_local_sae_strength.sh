#!/bin/bash
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
source scripts/slurm/_gpu_idle_gate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit
STRENGTH_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
STRENGTH_SPLIT="${1:?Specify dev or test}"
case "$STRENGTH_SPLIT" in dev|test) ;; *) exit 2 ;; esac
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$STRENGTH_RESULT/logs/processes_before_${STRENGTH_SPLIT}_generation.csv"
GPU_IDS=()
select_gpus_for_approved_node 24000 4 3 2
nvidia-smi > "$STRENGTH_RESULT/logs/gpus_before_${STRENGTH_SPLIT}_generation.txt"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/6_3_generate_local_sae_strength.py generate \
    --split "$STRENGTH_SPLIT" --shard "$shard" > "$STRENGTH_RESULT/logs/generate_${STRENGTH_SPLIT}_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"
