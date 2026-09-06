#!/bin/bash
#SBATCH -J 1_1_phase1_questions
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${FACT_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/1_1_build_utility_question_splits.py --config "${CONFIG}"

