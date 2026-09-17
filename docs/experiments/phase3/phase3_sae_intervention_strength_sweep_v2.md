# Phase 3: stronger teacher intervention sweep v2

This isolated exploratory rerun increases SAE intervention strength after the v1 pilot showed negligible length changes. Historical v1 artifacts remain in their original root.

- Config: `configs/phase3_sae_intervention_strength_sweep_v2.json`.
- Frozen protocol: `results/phase3_sae_intervention_strength_sweep_v2/exploratory/protocol/frozen_protocol.json`.
- Short enhancement strengths: 1, 4, 8, 12; matched random enhancement uses the same grid.
- Long suppression strengths: 0.25, 1, 2. Strength 2 subtracts twice the active component and can reverse its contribution.
- Baseline: no steering, regenerated with matched prompts, natural prefixes, and continuation seeds.
- Cohort: 64 dev questions, one candidate, 12 conditions, 768 records, eight disjoint question shards. The cohort includes the prior 24 calibration questions; it is not an independent confirmation pool.
- Fixed settings: layer 17, the same SAE and feature sets, 64-token common prefix, 512-token output cap, temperature 0.7, top-p 0.95, and a 15% hidden-state-relative perturbation cap.
- Scope: teacher behavior only. No new student training is included in this sweep.

The result root is a stable project symlink to BeeGFS. The launch manifest records the C31 existing allocation, C32 batch job, source hashes, shard assignments, and GPU memory-fit threshold. GPU keepalive and other existing processes remain active.

## Analysis

Run after all generation shards have completed:

```bash
/mnt/beegfs/youyang7/.conda/envs/sft/bin/python scripts/3_4_analyze_sae_strength_sweep.py --config results/phase3_sae_intervention_strength_sweep_v2/exploratory/protocol/frozen_protocol.json
```

The analyzer checks shard hashes, complete condition/question/candidate coverage, duplicate cells, actual common prefixes, and paired seeds. It writes metrics, merged records, a Chinese Markdown report, PNG/PDF dose-response plots, a hash manifest, and `STRENGTH_SWEEP_COMPLETE`. Existing analysis directories are never overwritten.

Effects use question-clustered paired bootstrap intervals. These are pointwise exploratory intervals without multiplicity correction. The screening rule requires at least 5% mean shortening, a length-difference interval wholly below zero, and no more than a 5 percentage point accuracy decline in the point estimate. The accuracy rule does not establish noninferiority. Enhancement is also compared with a matched-strength random direction.

The model hook records diagnostics across each generation batch, including forward positions after individual sequences have ended. These diagnostics describe batch forward activity rather than exact per-response effective-token exposure.

## Implementation validation

The existing calibration loader and condition summary were extracted into `src/length_budget_distill/sae_intervention_sweep.py` and reused by the existing selector and new analyzer. Historical calibration regression reproduced all stored metrics on 168 records across four audited shards. A temporary synthetic check validated exact paired effects, candidate clustering, random comparisons, figure creation, and rejection of mismatched paired seeds; synthetic artifacts were removed and are not experimental evidence.

The existing generator accepts an optional random-strength calibration grid; the original configuration still creates its original seven calibration conditions. The existing multi-GPU wrapper accepts an explicit stage and log prefix. New figures retain the Phase-3 palette but use dose-response lines and paired intervals because the scientific comparison concerns intervention strength.

## Completed outcome

All eight shards and 768 registered records completed. The source hashes match the launch record, and all output artifact hashes and the completion marker were verified. C31 generation exited successfully within the preserved grabgpu allocation; C32 job 279193 completed with exit code 0.

No condition passed the exploratory shortening screen. The no-steering mean was 255.45 tokens. Short enhancement at strength 8 changed length by -0.91 tokens (pointwise paired 95% CI [-7.95, 5.94]); strength 12 changed it by +2.55 tokens while recorded mean perturbation reached 14.56%. Long suppression at strength 2 increased length by 6.86 tokens (pointwise 95% CI [0.23, 14.66]); this is exploratory and uncorrected for multiple comparisons.

Detailed report: `results/phase3_sae_intervention_strength_sweep_v2/exploratory/strength_analysis/strength_report.md`.
Figure: `figures/phase3_sae_intervention_strength_sweep_v2/exploratory/teacher_strength_dose_response.png` (PDF also available).
