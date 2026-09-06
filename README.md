# Student-Specific Trace Utility and Credit Allocation

This standalone repository implements the gated GSM8K experiment program for testing whether short teacher traces help a Qwen2.5-1.5B student for content-related reasons rather than token-budget or loss-normalization artifacts.

## Evidence boundary

The execution and claim dependencies are:

1. Phase 0 was paused by researcher decision after all 108 training runs and 15 of 109 evaluations; it has no Gate-0 result.
2. Phase 1 was implemented but not submitted.
3. Phase 2 now runs directly as an exploratory mixed-trace SAE pilot on the sealed 881-question pool.
4. Utility-conditioned feature claims, causal steering, formal distillation, and cross-student experiments remain unavailable until their missing utility and falsification evidence is supplied.

Queued jobs, partial adapters, smoke checks, and Taylor agreement are not completed experimental evidence. Every formal stage writes hash-bound manifests, per-example predictions where applicable, statistical analysis, and an explicit completion marker.

## Layout

- `src/length_budget_distill/`: reusable CTV, baseline, evaluation, statistics, credit-allocation, and weighted-SFT logic.
- `src/trace_length_observation/`: Phase-0 selection, fairness budgets, controlled SFT, and analysis.
- `scripts/`: phase-first entrypoints.
- `scripts/slurm/`: gated, sharded Slurm jobs with physical-GPU checks.
- `configs/`: registered source protocols.
- `results/`: lightweight evidence, metrics, predictions, reports, and markers.
- `figures/`: publication figures produced by analysis scripts.
- `checkpoints/`: stable links to large BeeGFS adapter roots.
- `docs/`: estimands, gate rules, and implementation boundaries.

GPU admission on C30, C31, C32, and C49 is uniformly memory-fit: repeated physical checks require enough remaining memory for the registered workload plus margin, without requiring low utilization or near-zero existing allocation. Existing processes and `grabgpu` keepalives are preserved.

## Current execution status

Phase 0 jobs `278963` and `278965` were cancelled at 2026-09-04 02:58 EDT. All 108 audited adapters, 15 completed model evaluations, and 19,035 prediction rows were preserved. `PHASE0_PAUSED` records that Gate 0 was not evaluated. The active branch is `phase2_sae_pilot_v1`; activation extraction runs on C32, token sampling runs on C31, and the six SAE runs are split across C31 (`k=32`) and C32 (`k=64`) before a joint terminal audit.

The first failed Phase-0 attempt is preserved under `results/trace_length_observation_gate0_v1/formal/failed_attempt_1/` and is excluded from the formal root. It failed because some correct parent traces used `####` or `\boxed{}` answer formats; the corrected formal run audits answer-maskability before training.

## Validation

Run the CPU test and syntax suite with:

```bash
cd /home/youyang7/projects/SAE_long_short
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -m compileall -q src scripts
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -m ruff check --ignore E402 scripts/1_*.py scripts/2_*.py src/length_budget_distill tests
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python -m unittest discover -s tests
for script in scripts/slurm/1_*.sh scripts/slurm/2_*.sh; do bash -n "${script}"; done
```

## Execution

Check Gate 0 without mutating state:

```bash
python scripts/1_0_check_gate0.py
```

Submit the active mixed-trace SAE pilot:

```bash
python scripts/2_0_submit_sae_pilot.py
```

The retained but currently unsubmitted Phase-1 branch can be launched separately with:

```bash
python scripts/1_0_submit_phase1_utility_core.py --submit-policy-continuation
```

See the protocol documents in `docs/` for estimands and gate criteria.
