#!/bin/bash
#SBATCH -J 4_9_eval_mirror
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --oversubscribe
#SBATCH --time=01:00:00
#SBATCH --output=/var/tmp/%u-4_9-eval-mirror-%j.out
#SBATCH --error=/var/tmp/%u-4_9-eval-mirror-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
EVALUATION_SOURCE_ROOT="${EVALUATION_SOURCE_ROOT:?EVALUATION_SOURCE_ROOT is required}"
EXPECTED_COUNT="${EXPECTED_COUNT:?EXPECTED_COUNT is required}"
DESTINATION_ROOT="${DESTINATION_ROOT:-/mnt/beegfs/youyang7/projects/SAE_long_short/phase3_sae_intervention_distillation_pilot_v1/recovery_evaluations_v1}"
ARGS=(
  --evaluation-root "${EVALUATION_SOURCE_ROOT}"
  --destination-root "${DESTINATION_ROOT}"
  --expected-count "${EXPECTED_COUNT}"
)
if [ -n "${REPLACEMENT_1:-}" ]; then
  ARGS+=(--replacement "${REPLACEMENT_1}")
fi
if [ -n "${REPLACEMENT_2:-}" ]; then
  ARGS+=(--replacement "${REPLACEMENT_2}")
fi
PYTHONDONTWRITEBYTECODE=1 /home/youyang7/.conda/envs/sft/bin/python -u \
  scripts/4_9_build_evaluation_recovery_mirror.py "${ARGS[@]}"
