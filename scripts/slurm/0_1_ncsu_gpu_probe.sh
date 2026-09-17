#!/bin/bash
# Short NCSU allocation probe. Supply account, partition, QOS, GRES, and log
# paths explicitly through sbatch. This does not launch an experiment.
set -euo pipefail
date --iso-8601=seconds
hostname
printf 'JOB_ID=%s PARTITION=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID:-}" "${SLURM_JOB_PARTITION:-}" "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
printf 'NCSU_GPU_VISIBILITY_PROBE_COMPLETE\n'
