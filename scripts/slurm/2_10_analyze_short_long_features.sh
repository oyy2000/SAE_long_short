#!/bin/bash
#SBATCH -J 2_10_sae_analysis
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
CONFIG="${FEATURE_CONFIG:-results/phase2_short_long_feature_analysis_v1/exploratory/protocol/frozen_protocol.json}"
python3 -u scripts/2_10_analyze_short_long_features.py --config "${CONFIG}"
python3 -u scripts/2_11_audit_short_long_feature_analysis.py --config "${CONFIG}"
