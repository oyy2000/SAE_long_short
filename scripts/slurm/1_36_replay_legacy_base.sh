#!/bin/bash
#SBATCH --job-name=1_36_base_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --oversubscribe
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# GPU 0 becomes available only after shard 2 finishes; shard 0 retains GPU 2.
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python - <<'PY'
import sys,time
sys.path.insert(0,'src')
from length_budget_distill.legacy_replication import gpu_capacity
while not gpu_capacity('0',18000): time.sleep(30)
PY
export CUDA_VISIBLE_DEVICES=0
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -u scripts/1_36_replay_legacy_base.py --config "${LEGACY_CONFIG:?}"
