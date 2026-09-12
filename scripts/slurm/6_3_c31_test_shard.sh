#!/bin/bash
# Complete the fourth fixed test shard after dev shard 0 releases its GPU.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
source scripts/slurm/_gpu_idle_gate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit
TEST_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
test -f "$TEST_RESULT/generation/dev/shard_0/GENERATION_COMPLETE.json"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$TEST_RESULT/logs/c31_processes_before_test_shard3.csv"
GPU_IDS=()
select_gpus_for_approved_node 24000 1 3 2
nvidia-smi > "$TEST_RESULT/logs/c31_gpus_before_test_shard3.txt"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/6_3_generate_local_sae_strength.py generate \
  --split test --shard 3 > "$TEST_RESULT/logs/generate_test_3.log" 2>&1
