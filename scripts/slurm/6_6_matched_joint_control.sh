#!/bin/bash
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
source scripts/slurm/_gpu_idle_gate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit
JOINT_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
"$SAE_EXPERIMENT_PYTHON" -m unittest discover -s tests -p test_sae_joint_control.py > "$JOINT_RESULT/logs/joint_control_tests.log" 2>&1
"$SAE_EXPERIMENT_PYTHON" scripts/6_6_match_joint_random_control.py freeze
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$JOINT_RESULT/logs/processes_before_joint_control.csv"
GPU_IDS=()
select_gpus_for_approved_node 24000 3 3 2
nvidia-smi > "$JOINT_RESULT/logs/gpus_before_joint_control.txt"
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" "$SAE_EXPERIMENT_PYTHON" -u scripts/6_6_match_joint_random_control.py generate --shard "$shard" > "$JOINT_RESULT/logs/joint_control_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [ "$failed" -ne 0 ]; then exit 1; fi
"$SAE_EXPERIMENT_PYTHON" scripts/6_6_match_joint_random_control.py analyze > "$JOINT_RESULT/logs/analyze_joint_control.log" 2>&1
