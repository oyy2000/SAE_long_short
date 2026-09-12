#!/bin/bash
# Fixed test shards 0/1/2; identical assets, independent caches, no test analysis.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_gpu_idle_gate.sh
C32_RUNTIME=/var/tmp/youyang7-phase6-runtime-c32-v1
C32_PYTHON="$C32_RUNTIME/envs/sft/bin/python"
export PYTHONPATH="$C32_RUNTIME/overlays/legacy_trl_096_v2:/home/youyang7/projects/SAE_long_short/src"
export HF_HOME="$C32_RUNTIME/hf" HF_HUB_CACHE="$C32_RUNTIME/hf/hub" HF_DATASETS_CACHE="$C32_RUNTIME/datasets"
export TMPDIR="$C32_RUNTIME/tmp" XDG_CACHE_HOME="$C32_RUNTIME/cache"
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 GPU_ADMISSION_POLICY=memory_fit TOKENIZERS_PARALLELISM=false
C32_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
"$C32_PYTHON" - <<'PY' > "$C32_RESULT/logs/c32_assets_verified.json"
import json,platform
from pathlib import Path
import torch,transformers
from length_budget_distill.sae_local_data import verify
p=Path('results/phase6_local_length_controlled_strength_v1/exploratory/RECOVERY_COMPLETE.json')
j=json.loads(p.read_text())
for item in j['teacher_files']:verify(item)
print(json.dumps({'status':'passed','node':platform.node(),'teacher_files':j['teacher_files'],
                 'torch':torch.__version__,'transformers':transformers.__version__},indent=2))
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$C32_RESULT/logs/c32_processes_before_test_generation.csv"
GPU_IDS=()
select_gpus_for_approved_node 24000 3 3 2
nvidia-smi > "$C32_RESULT/logs/c32_gpus_before_test_generation.txt"
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" "$C32_PYTHON" -u scripts/6_3_generate_local_sae_strength.py generate \
    --split test --shard "$shard" > "$C32_RESULT/logs/generate_test_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
exit "$failed"
