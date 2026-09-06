#!/bin/bash
#SBATCH -J 2_19_sae_audit
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --oversubscribe
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
python3 -u scripts/2_19_audit_sae_sampling_ablation.py \
  --failed-training-job "${FAILED_TRAINING_JOB:-278985}" \
  --retry-training-job "${RETRY_TRAINING_JOB:-278993}"
