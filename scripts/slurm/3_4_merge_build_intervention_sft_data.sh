#!/bin/bash
#SBATCH -J 3_4_interv_data
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
CONFIG="${INTERVENTION_CONFIG:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory/protocol/frozen_protocol.json}"
python3 -u scripts/3_3_merge_intervention_generations.py --config "${CONFIG}"
python3 -u scripts/4_0_build_intervention_sft_datasets.py --config "${CONFIG}"
