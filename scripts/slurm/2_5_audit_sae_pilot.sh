#!/bin/bash
#SBATCH -J 2_5_sae_audit
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1
CONFIG="${PHASE2_CONFIG:-results/phase2_sae_pilot_v1/formal/protocol/frozen_protocol.json}"
python3 -u scripts/2_6_audit_sae_pilot.py --config "${CONFIG}"

