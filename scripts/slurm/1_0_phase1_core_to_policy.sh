#!/bin/bash
#SBATCH -J 1_0_phase1_policy
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
python3 -u scripts/1_0_submit_phase1_policy_sft.py --submit-phase1-5-continuation

