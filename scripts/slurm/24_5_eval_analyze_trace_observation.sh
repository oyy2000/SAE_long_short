#!/bin/bash

#SBATCH -J 24_5_phase0_eval
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/SAE_long_short/scripts/slurm/_gpu_idle_gate.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/SAE_long_short

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

RESULT_ROOT="${RESULT_ROOT:-results/trace_length_observation_gate0_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/trace_length_observation_gate0_v1}"
FROZEN_CONFIG="${FROZEN_CONFIG:-${RESULT_ROOT}/formal/protocol/frozen_protocol.json}"
TRAINING_AUDIT="${TRAINING_AUDIT:-${RESULT_ROOT}/formal/training/audit/training_audit.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
MIN_GPUS="${MIN_GPUS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
RUN_TAG="${RUN_TAG:-phase0_eval_${SLURM_JOB_ID:-manual}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-phase0-eval/${RUN_TAG}}"

if [ ! -f "${RESULT_ROOT}/formal/training/audit/TRAINING_COMPLETE" ]; then
  echo "Phase-0 training audit is not complete: ${RESULT_ROOT}" >&2
  exit 2
fi

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

echo "host=$(hostname) min_free_mib=${MIN_FREE_MIB} min_gpus=${MIN_GPUS}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_IDS=("${GPU_IDS[@]:0:${MIN_GPUS}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

python3 -u scripts/24_5_launch_trace_observation_evaluation.py \
  --config "${FROZEN_CONFIG}" \
  --training-audit "${TRAINING_AUDIT}" \
  --output-dir "${RESULT_ROOT}/formal/evaluation" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel "${MIN_GPUS}" \
  --skip-complete

conda activate "${FACT_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
python3 -u scripts/24_6_analyze_trace_observation.py \
  --config "${FROZEN_CONFIG}" \
  --evaluation-manifest "${RESULT_ROOT}/formal/evaluation/evaluation_manifest.json" \
  --output-dir "${RESULT_ROOT}/formal/analysis" \
  --figure-dir "${FIGURE_ROOT}/formal"

python3 -u scripts/24_7_audit_trace_observation_experiment.py \
  --config "${FROZEN_CONFIG}" \
  --result-root "${RESULT_ROOT}" \
  --stage formal

echo "phase0_experiment_complete result_root=${RESULT_ROOT}"
