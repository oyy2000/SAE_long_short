#!/bin/bash
# Main generation uses seven disjoint shards across C31/C32; SFT uses four C31 GPUs.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
FOLLOWUP_STAGE="${1:?generate or students}"
FOLLOWUP_NODE="$(hostname -s | tr '[:upper:]' '[:lower:]')"
FOLLOWUP_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
source scripts/slurm/_gpu_idle_gate.sh
case "$FOLLOWUP_NODE" in
 c31)
  source scripts/slurm/_activate_legacy_sae_c31_v2.sh
  FOLLOWUP_PYTHON="$SAE_EXPERIMENT_PYTHON"
  FOLLOWUP_SHARDS=(0 1 2 3)
  ;;
 c32)
  FOLLOWUP_RUNTIME=/var/tmp/youyang7-phase6-runtime-c32-v1
  FOLLOWUP_PYTHON="$FOLLOWUP_RUNTIME/envs/sft/bin/python"
  export PYTHONPATH="$FOLLOWUP_RUNTIME/overlays/legacy_trl_096_v2:$PWD/src"
  export HF_HOME="$FOLLOWUP_RUNTIME/hf" HF_HUB_CACHE="$FOLLOWUP_RUNTIME/hf/hub" HF_DATASETS_CACHE="$FOLLOWUP_RUNTIME/datasets"
  export TMPDIR="$FOLLOWUP_RUNTIME/tmp" XDG_CACHE_HOME="$FOLLOWUP_RUNTIME/cache"
  unset TRANSFORMERS_CACHE
  FOLLOWUP_SHARDS=(4 5 6)
  ;;
 *) exit 2 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit TOKENIZERS_PARALLELISM=false
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$FOLLOWUP_RESULT/logs/${FOLLOWUP_STAGE}_${FOLLOWUP_NODE}_processes.csv"
GPU_IDS=()
select_gpus_for_approved_node 24000 "${#FOLLOWUP_SHARDS[@]}" 3 2
nvidia-smi > "$FOLLOWUP_RESULT/logs/${FOLLOWUP_STAGE}_${FOLLOWUP_NODE}_gpus.txt"
pids=()
case "$FOLLOWUP_STAGE" in
 generate)
  for index in "${!FOLLOWUP_SHARDS[@]}"; do
   shard="${FOLLOWUP_SHARDS[$index]}"
   CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" "$FOLLOWUP_PYTHON" -u scripts/6_7_run_local_intervention_students.py generate --shard "$shard" > "$FOLLOWUP_RESULT/logs/main_generation_${shard}.log" 2>&1 &
   pids+=("$!")
  done
  ;;
 students)
  [[ "$FOLLOWUP_NODE" == c31 ]] || exit 2
  conditions=(no_steering selected_target matched_random historical_short)
  for index in 0 1 2 3; do
   (
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}"
    for regime in equal_examples equal_target_tokens; do
     arm="${regime}__${conditions[$index]}__seed_17"
     "$FOLLOWUP_PYTHON" -u scripts/6_7_run_local_intervention_students.py train_eval --arm "$arm" > "$FOLLOWUP_RESULT/logs/student_${arm}.log" 2>&1
    done
   ) &
   pids+=("$!")
  done
  ;;
 *) exit 2 ;;
esac
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"
