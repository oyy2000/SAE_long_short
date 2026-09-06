#!/bin/bash
#SBATCH -J 4_5_interv_analysis
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PIP_NO_CACHE_DIR=1
BASE_PYTHON="/home/youyang7/.conda/envs/sft/bin/python"
ANALYSIS_RECOVERY_ROOT="/var/tmp/${USER}-sae-analysis-recovery-v1"
ANALYSIS_PACKAGES="${ANALYSIS_RECOVERY_ROOT}/packages"
if [ ! -f "${ANALYSIS_RECOVERY_ROOT}/PACKAGES_READY" ]; then
  PACKAGE_TMP=$(mktemp -d "/var/tmp/${USER}-sae-analysis-packages.XXXXXX")
  "${BASE_PYTHON}" -m pip install --target "${PACKAGE_TMP}" \
    numpy==1.26.4 matplotlib==3.10.6
  PYTHONPATH="${PACKAGE_TMP}" "${BASE_PYTHON}" -c \
    'import numpy,matplotlib; assert numpy.__version__ == "1.26.4"; assert matplotlib.__version__ == "3.10.6"'
  mkdir -p "${ANALYSIS_RECOVERY_ROOT}"
  if [ -e "${ANALYSIS_PACKAGES}" ]; then
    find "${ANALYSIS_PACKAGES}" -mindepth 1 -delete
    rmdir "${ANALYSIS_PACKAGES}"
  fi
  mv "${PACKAGE_TMP}" "${ANALYSIS_PACKAGES}"
  printf 'numpy=1.26.4\nmatplotlib=3.10.6\n' > "${ANALYSIS_RECOVERY_ROOT}/PACKAGES_READY"
fi
export PYTHONPATH="${ANALYSIS_PACKAGES}"
export SAE_ADAPTER_RECOVERY_ROOT="${SAE_ADAPTER_RECOVERY_ROOT:-/mnt/beegfs/youyang7/projects/SAE_long_short/phase3_sae_intervention_distillation_pilot_v1/recovery_adapters_v1}"
export SAE_EVALUATION_RECOVERY_ROOT="${SAE_EVALUATION_RECOVERY_ROOT:-/mnt/beegfs/youyang7/projects/SAE_long_short/phase3_sae_intervention_distillation_pilot_v1/recovery_evaluations_v1}"
RESULT_ROOT="${RESULT_ROOT:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory}"
CONFIG="${INTERVENTION_CONFIG:-${RESULT_ROOT}/protocol/frozen_protocol.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/phase3_sae_intervention_distillation_pilot_v1}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${RESULT_ROOT}/evaluation}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RESULT_ROOT}/analysis}"
FIGURE_DIR="${FIGURE_DIR:-figures/phase3_sae_intervention_distillation_pilot_v1/exploratory}"
LOCAL_ANALYSIS_ROOT="${LOCAL_ANALYSIS_ROOT:-/var/tmp/${USER}-sae-intervention-analysis/${SLURM_JOB_ID}}"
mkdir -p "${LOCAL_ANALYSIS_ROOT}"
export SAE_ANALYSIS_OUTPUT_STAGE_ROOT="${LOCAL_ANALYSIS_ROOT}/published"
export SAE_AUDIT_OUTPUT_STAGE_ROOT="${LOCAL_ANALYSIS_ROOT}/audit_published"
"${BASE_PYTHON}" -u scripts/4_5_analyze_intervention_distillation.py \
  --config "${CONFIG}" --checkpoint-root "${CHECKPOINT_ROOT}" \
  --evaluation-root "${EVALUATION_ROOT}" --output-dir "${ANALYSIS_DIR}" \
  --figure-dir "${FIGURE_DIR}"
"${BASE_PYTHON}" -u scripts/4_6_audit_intervention_distillation.py \
  --config "${CONFIG}" --checkpoint-root "${CHECKPOINT_ROOT}" \
  --evaluation-root "${EVALUATION_ROOT}" --analysis-dir "${ANALYSIS_DIR}" \
  --output-dir "${RESULT_ROOT}/final_audit"
