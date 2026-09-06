#!/bin/bash
#SBATCH --job-name=1_35_legacy_sae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_gpu_idle_gate.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 GPU_ADMISSION_POLICY=memory_fit
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface HF_HUB_OFFLINE=1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
GPU_IDS=()
select_gpus_for_approved_node 12000 1 30 2
SAE_GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -u scripts/1_34_legacy_sae_continuation.py worker \
 --config "${LEGACY_CONFIG:?}" --shard "${SAE_SHARD:?}" --shards "${SAE_SHARDS:?}" --gpu-ids "$SAE_GPU_LIST"
