#!/bin/bash
# Recheck the worker's assigned GPU before each new student workload.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_gpu_idle_gate.sh
STUDENT_ARM="${1:?Registered student arm required}"
STUDENT_GPU="${CUDA_VISIBLE_DEVICES:?Expected one assigned physical GPU}"
[[ "$STUDENT_GPU" =~ ^[0-9]+$ ]] || exit 2
export GPU_ADMISSION_POLICY=memory_fit
STUDENT_ADMISSION_LOG="results/phase6_local_length_controlled_strength_v1/exploratory/logs/student_${STUDENT_ARM}_admission.log"
{
 while true; do
  GPU_IDS=()
  select_gpus_for_approved_node "${STUDENT_MIN_FREE_MIB:-24000}" 1 3 2
  admitted=0
  for candidate in "${GPU_IDS[@]}"; do if [[ "$candidate" == "$STUDENT_GPU" ]]; then admitted=1; fi; done
  if [[ "$admitted" == 1 ]]; then break; fi
  echo "Waiting for sufficient memory on assigned GPU ${STUDENT_GPU}; allocations and existing processes remain open."
  sleep 30
 done
 nvidia-smi -i "$STUDENT_GPU"
 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} >> "$STUDENT_ADMISSION_LOG" 2>&1
