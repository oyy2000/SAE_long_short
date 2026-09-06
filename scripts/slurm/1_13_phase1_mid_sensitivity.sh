#!/bin/bash
#SBATCH -J 1_13_phase1_mid_anchor
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
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"; MIN_GPUS=3; WAIT_SECONDS="${WAIT_SECONDS:-60}"; STABLE_CHECKS=2
MAX_USED_MIB="${MAX_USED_MIB:-500}"; MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
GPU_IDS=(); select_gpus_for_approved_node "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
CONFIG="${PHASE1_CONFIG:-results/phase1_teaching_utility_v1/formal/protocol/frozen_protocol.json}"
ROOT=results/phase1_teaching_utility_v1/formal/mid_anchor_sensitivity
ADAPTER=checkpoints/phase1_teaching_utility_v1/anchors/mid_sft_random_50pct
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python3 -u scripts/1_5_calibrate_ctv_step.py --config "${CONFIG}" \
  --anchor-adapter "${ADAPTER}" --output-dir "${ROOT}/calibration"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python3 -u scripts/1_7_score_ctv_gradient_shard.py --config "${CONFIG}" \
  --anchor-adapter "${ADAPTER}" --calibration-report "${ROOT}/calibration/calibration_report.json" \
  --candidate-shard results/phase1_teaching_utility_v1/formal/anchor_sensitivity_inputs/mid_sft_exact_candidates.jsonl \
  --output-dir "${ROOT}/ctv_scores/shard_00_of_01"
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python3 -u scripts/1_8_measure_exact_ctv_shard.py --config "${CONFIG}" \
    --anchor-adapter "${ADAPTER}" --calibration-report "${ROOT}/calibration/calibration_report.json" \
    --exact-candidates results/phase1_teaching_utility_v1/formal/anchor_sensitivity_inputs/mid_sft_exact_candidates.jsonl \
    --shard-index "${shard}" --shard-count 3 --output-dir "${ROOT}/exact_ctv/shard_0${shard}_of_03" \
    >"logs/phase1_mid_exact_${shard}-${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if [ "${status}" -ne 0 ]; then exit "${status}"; fi
python3 -u scripts/1_11_analyze_exact_ctv.py --config "${CONFIG}" \
  --score-root "${ROOT}/ctv_scores" --exact-root "${ROOT}/exact_ctv" \
  --output-dir "${ROOT}/ctv_analysis" --figure-dir figures/phase1_teaching_utility_v1/mid_anchor_sensitivity \
  --expected-question-count 100 --analysis-label mid_sft_anchor

