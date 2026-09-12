#!/bin/bash
# Continue the registered DAG after all seven independent teacher shards finish.
set -euo pipefail
cd /home/youyang7/projects/SAE_long_short
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
FOLLOWUP_RESULT=results/phase6_local_length_controlled_strength_v1/exploratory
for attempt in $(seq 1 720); do
 complete=0
 for shard in 0 1 2 3 4 5 6; do
  if [[ -f "$FOLLOWUP_RESULT/student_followup/generation/shard_${shard}/COMPLETE.json" ]]; then complete=$((complete+1)); fi
 done
 if [[ "$complete" == 7 ]]; then break; fi
 if [[ "$attempt" == 720 ]]; then echo 'Timed out waiting for complete teacher shards'; exit 1; fi
 sleep 30
done
echo 'All seven main teacher shards complete; audit and build the eight SFT datasets.'
"$SAE_EXPERIMENT_PYTHON" -u scripts/6_7_run_local_intervention_students.py build > "$FOLLOWUP_RESULT/logs/build_students.log" 2>&1
echo 'Data audit passed; start the eight registered student training/evaluation runs.'
bash scripts/slurm/6_7_local_student_followup.sh students > "$FOLLOWUP_RESULT/logs/student_wrapper.log" 2>&1
echo 'Eight student workers completed; analyze full paired evaluation.'
"$SAE_EXPERIMENT_PYTHON" -u scripts/6_7_run_local_intervention_students.py analyze > "$FOLLOWUP_RESULT/logs/analyze_students.log" 2>&1
echo 'Student analysis completed. Final report awaits full artifact audit and figure inspection.'
