#!/bin/bash
#SBATCH -J 4_7_repair_adapter
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --oversubscribe
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
SOURCE_DIR="${SOURCE_DIR:?SOURCE_DIR is required}"
CANONICAL_DIR="${CANONICAL_DIR:?CANONICAL_DIR is required}"
BACKUP_SUFFIX="${BACKUP_SUFFIX:?BACKUP_SUFFIX is required}"
/home/youyang7/.conda/envs/sft/bin/python -u scripts/4_7_repair_remote_io_adapter.py \
  --source-dir "${SOURCE_DIR}" \
  --canonical-dir "${CANONICAL_DIR}" \
  --backup-suffix "${BACKUP_SUFFIX}"
