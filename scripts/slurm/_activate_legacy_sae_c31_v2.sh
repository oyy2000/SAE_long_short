#!/bin/bash
# Source inside a C31 allocation. The recovery environment itself is unchanged.
source /mnt/local/youyang7/SAE_long_short_c31/activate_runtime.sh
export SAE_LEGACY_TRL_OVERLAY=/mnt/local/youyang7/SAE_long_short_c31/overlays/legacy_trl_096_v2
export PYTHONPATH="${SAE_LEGACY_TRL_OVERLAY}:${PYTHONPATH}"
export SAE_EXPERIMENT_PYTHON=/mnt/local/youyang7/SAE_long_short_c31/envs/sft/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
