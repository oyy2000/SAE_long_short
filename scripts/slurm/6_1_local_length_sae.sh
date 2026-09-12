#!/bin/bash
# Inside the user's existing C31 allocation; keep all keepalive jobs intact.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
source scripts/slurm/_gpu_idle_gate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit
LOCAL_SAE_CONFIG=configs/phase6_local_length_controlled_strength_v1.json
LOCAL_SAE_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
LOCAL_SAE_RUNTIME=/mnt/local/youyang7/SAE_long_short_c31/runs/phase6_local_length_controlled_strength_v1
mkdir -p "$LOCAL_SAE_RUNTIME/tmp"
export TMPDIR="$LOCAL_SAE_RUNTIME/tmp"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$LOCAL_SAE_RESULT/logs/processes_before_extraction.csv"
GPU_IDS=()
select_gpus_for_approved_node 22000 4 3 2
nvidia-smi > "$LOCAL_SAE_RESULT/logs/gpus_before_extraction.txt"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/6_1_prepare_local_length_sae.py extract \
    --config "$LOCAL_SAE_CONFIG" --shard "$shard" > "$LOCAL_SAE_RESULT/logs/extract_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [ "$failed" -ne 0 ]; then exit 1; fi
"$SAE_EXPERIMENT_PYTHON" scripts/6_1_prepare_local_length_sae.py sample --config "$LOCAL_SAE_CONFIG" > "$LOCAL_SAE_RESULT/logs/sample.log" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$LOCAL_SAE_RESULT/logs/processes_before_training.csv"
GPU_IDS=()
select_gpus_for_approved_node 16000 3 3 2
nvidia-smi > "$LOCAL_SAE_RESULT/logs/gpus_before_training.txt"
conditions=(full_token_uniform full_trace_balanced prefix64_trace_balanced)
pids=()
for index in 0 1 2; do
  condition="${conditions[$index]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/2_4_train_topk_sae.py \
    --config "$LOCAL_SAE_CONFIG" --sample-root "$LOCAL_SAE_RUNTIME/samples/$condition" \
    --layer-index 17 --k 64 --output-dir "$LOCAL_SAE_RESULT/sae_training/$condition" \
    --checkpoint-dir "checkpoints/phase6_local_length_controlled_strength_v1/$condition" \
    > "$LOCAL_SAE_RESULT/logs/train_${condition}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"
