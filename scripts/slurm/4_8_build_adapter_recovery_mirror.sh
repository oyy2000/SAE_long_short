#!/bin/bash
#SBATCH -J 4_8_adapter_mirror
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --oversubscribe
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
RUN_CONFIG_ROOT="${RUN_CONFIG_ROOT:-/home/youyang7/projects/SAE_long_short/results/phase3_sae_intervention_distillation_pilot_v1/exploratory/training/run_configs}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/mnt/beegfs/youyang7/projects/SAE_long_short/phase3_sae_intervention_distillation_pilot_v1/checkpoints}"
DESTINATION_ROOT="${DESTINATION_ROOT:-/mnt/beegfs/youyang7/projects/SAE_long_short/phase3_sae_intervention_distillation_pilot_v1/recovery_adapters_v1}"
/home/youyang7/.conda/envs/sft/bin/python -u scripts/4_8_build_adapter_recovery_mirror.py \
  --runtime-root "${RUNTIME_ROOT}" \
  --run-config-root "${RUN_CONFIG_ROOT}" \
  --canonical-root "${CANONICAL_ROOT}" \
  --destination-root "${DESTINATION_ROOT}"
