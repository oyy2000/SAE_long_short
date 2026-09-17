#!/bin/bash
#SBATCH -J 4_4_interv_eval
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/projects/SAE_long_short/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/SAE_long_short/logs/%x-%j.err
set -euo pipefail
source /home/youyang7/projects/SAE_long_short/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/SAE_long_short
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export GPU_ADMISSION_POLICY="${GPU_ADMISSION_POLICY:-memory_fit}"
RESULT_ROOT="${RESULT_ROOT:-results/phase3_sae_intervention_distillation_pilot_v1/exploratory}"
CONFIG="${INTERVENTION_CONFIG:-${RESULT_ROOT}/protocol/frozen_protocol.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/phase3_sae_intervention_distillation_pilot_v1}"
EVALUATION_OUTPUT_ROOT="${EVALUATION_OUTPUT_ROOT:-${RESULT_ROOT}/evaluation}"
LAUNCHER_SHARDS="${LAUNCHER_SHARDS:-3}"
LAUNCHER_SHARD_INDEX="${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}"
GPU_COUNT="${GPU_COUNT:?GPU_COUNT is required}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-sae-intervention-eval/${SLURM_JOB_ID}}"
mkdir -p "${LOCAL_RUNTIME_ROOT}/tmp" "${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export HF_HOME="${LOCAL_RUNTIME_ROOT}/huggingface"

# Recovery environment is intentionally node-local. It bypasses transient or
# permanently unreadable package inodes on BeeGFS without mutating the shared env.
BASE_PYTHON="/home/youyang7/.conda/envs/sft/bin/python"
RECOVERY_ROOT="/var/tmp/${USER}-sae-eval-recovery-v1"
RECOVERY_PACKAGES="${RECOVERY_ROOT}/packages"
if [ ! -f "${RECOVERY_ROOT}/PACKAGES_READY" ]; then
  PACKAGE_TMP=$(mktemp -d "/var/tmp/${USER}-sae-eval-packages.XXXXXX")
  "${BASE_PYTHON}" -m pip install --target "${PACKAGE_TMP}" \
    numpy==1.26.4 torch==2.10.0 transformers==4.48.3 datasets==3.2.0 \
    peft==0.12.0 accelerate==1.2.1
  PYTHONPATH="${PACKAGE_TMP}" "${BASE_PYTHON}" -c \
    'import numpy,torch,transformers,datasets,peft; assert numpy.__version__ == "1.26.4"; assert transformers.__version__ == "4.48.3"'
  mkdir -p "${RECOVERY_ROOT}"
  if [ -e "${RECOVERY_PACKAGES}" ]; then
    find "${RECOVERY_PACKAGES}" -mindepth 1 -delete
    rmdir "${RECOVERY_PACKAGES}"
  fi
  mv "${PACKAGE_TMP}" "${RECOVERY_PACKAGES}"
  printf 'numpy=1.26.4\ntorch=2.10.0\ntransformers=4.48.3\n' > "${RECOVERY_ROOT}/PACKAGES_READY"
fi
export PYTHONPATH="${RECOVERY_PACKAGES}"

SOURCE_MODEL="/mnt/beegfs/youyang7/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LOCAL_MODEL="${RECOVERY_ROOT}/qwen2p5_1p5b_989aa798"
if [ ! -f "${LOCAL_MODEL}/MODEL_READY" ]; then
  MODEL_TMP=$(mktemp -d "/var/tmp/${USER}-sae-eval-model.XXXXXX")
  COPIED=0
  for ATTEMPT in $(seq 1 120); do
    rm -f "${MODEL_TMP}/model.safetensors"
    if cp -L "${SOURCE_MODEL}/model.safetensors" "${MODEL_TMP}/model.safetensors"; then
      COPIED=1
      break
    fi
    echo "Transient model-weight staging failure retry=${ATTEMPT}/120" >&2
    sleep 5
  done
  test "${COPIED}" -eq 1
  for NAME in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do
    curl --fail --location --retry 20 --retry-delay 3 \
      "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/${NAME}" \
      --output "${MODEL_TMP}/${NAME}"
  done
  MODEL_HASH=$(sha256sum "${MODEL_TMP}/model.safetensors" | awk '{print $1}')
  test "${MODEL_HASH}" = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
  printf 'revision=989aa7980e4cf806f80c7fef2b1adb7bc71aa306\nmodel_sha256=%s\n' "${MODEL_HASH}" > "${MODEL_TMP}/MODEL_READY"
  if [ -e "${LOCAL_MODEL}" ]; then
    find "${LOCAL_MODEL}" -mindepth 1 -delete
    rmdir "${LOCAL_MODEL}"
  fi
  mv "${MODEL_TMP}" "${LOCAL_MODEL}"
fi
export SAE_EVAL_MODEL_PATH="${LOCAL_MODEL}"
export SAE_ADAPTER_RECOVERY_ROOT="${SAE_ADAPTER_RECOVERY_ROOT:-/mnt/beegfs/youyang7/projects/SAE_long_short/phase3_sae_intervention_distillation_pilot_v1/recovery_adapters_v1}"
export SAE_EVAL_ADAPTER_STAGE_ROOT="${LOCAL_RUNTIME_ROOT}/adapters"
export SAE_EVAL_OUTPUT_STAGE_ROOT="${LOCAL_RUNTIME_ROOT}/outputs"
export SAE_EVAL_LAUNCHER_LOG_STAGE_ROOT="${LOCAL_RUNTIME_ROOT}/launcher_logs"
export SAE_EVAL_LAUNCHER_MANIFEST_STAGE_ROOT="${LOCAL_RUNTIME_ROOT}/launcher_manifests"
mkdir -p "${SAE_EVAL_ADAPTER_STAGE_ROOT}" "${SAE_EVAL_OUTPUT_STAGE_ROOT}" \
  "${SAE_EVAL_LAUNCHER_LOG_STAGE_ROOT}" "${SAE_EVAL_LAUNCHER_MANIFEST_STAGE_ROOT}"
GPU_IDS=()
select_gpus_for_approved_node "${MIN_FREE_MIB:-14000}" "${GPU_COUNT}" "${WAIT_SECONDS:-30}" 2
GPU_IDS=("${GPU_IDS[@]:0:${GPU_COUNT}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
LAUNCHER_ARGS=(
  --config "${CONFIG}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --output-root "${EVALUATION_OUTPUT_ROOT}"
  --gpu-ids "${GPU_ID_CSV}"
  --launcher-shards "${LAUNCHER_SHARDS}"
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX}"
  --skip-complete
)
if [ -n "${MODEL_ID_FILTER:-}" ]; then
  LAUNCHER_ARGS+=(--model-id-filter "${MODEL_ID_FILTER}")
fi
"${BASE_PYTHON}" -u scripts/4_4_launch_intervention_evaluation.py "${LAUNCHER_ARGS[@]}"
