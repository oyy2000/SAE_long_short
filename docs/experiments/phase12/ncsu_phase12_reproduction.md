# NCSU teacher/student and SAE reproduction

The user authorized fresh teacher generation, student SFT, evaluation, and SAE
analysis on NCSU on 2026-09-12. Outputs stay inside this project. The registered
configuration is `configs/phase12_ncsu_reproduction_v1.json`; the execution copy,
source hashes, and independent input/model hashes are in
`results/ncsu_phase12_reproduction_v1/exploratory/protocol/` and `inputs/`.

This is a new-sample exploratory reproduction. It tests the short-trace advantage
without assuming that the result will reproduce. It does not execute the separate
CTV utility program or SAE steering/intervention experiments.

## Inputs and settings

- The 881 training question IDs are recovered from the transferred archive's
  question member and checked, including prompt and gold answer, against an
  independently downloaded revision of official GSM8K. No archived teacher
  completion, adapter, or prediction is reused. Only that complete archive member
  is needed; the archive as a whole is not asserted to be complete.
- Teacher: Qwen2.5-7B-Instruct, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`. Generate 16 candidates per question
  with the existing length-agnostic teacher prompt, temperature 0.7, top-p 0.95,
  maximum 512 new tokens. Four disjoint problem shards use independent processes
  on Slurm-assigned GPUs. Per-question seeds are derived deterministically from
  the registered base seed and problem ID.
- Select one short, medium, and long representative per eligible problem using
  the existing `within_quantile_band_median_v1` implementation. The short and long
  representatives come from the bottom and top 20% bands, not necessarily the
  absolute extremes. Require four unique correct, non-capped candidates; preserve
  all raw outputs and list any excluded questions. All three SFT conditions use
  exactly the same question support. This explicitly registered selection rule
  makes the run a method-level reproduction, not a byte-identical legacy replay.
- Student: Qwen2.5-1.5B-Instruct, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; seeds 17/42/73; LoRA r=4, alpha=16,
  dropout=0.05, all-linear; one epoch, batch size 4, gradient accumulation 1,
  learning rate 2e-5, warmup ratio 0.03, BF16, completion-only loss, maximum
  sequence length 2048. The existing `run_trl_sft` helper performs training.
- Re-evaluate the base student and all nine adapters using the existing explicit
  question evaluator, greedy decoding, batch 32, maximum 512 new tokens, and
  `test[50:1319]` (1,269 questions). Only `test[:50]` is used for smoke checks.
- SAE: reuse the existing mixed-corpus, residual-extraction, token-sampling,
  TopK training, feature-scoring, and plotting scripts. On the common eligible
  question support, include every raw candidate regardless of correctness.
  Use layers 10/17/23, k=32/64, expansion 8, seed 17, 1,500 steps, and the
  remaining original Phase-2 hyperparameters. Whole-question splits are
  617/132/132 if all 881 questions remain; the frozen rounding rule handles any
  documented exclusions before SAE fitting.
- SAE discovery uses dev questions and confirmation uses test questions.
  Retain the original first-64-token direction check, prevalence thresholds,
  paired-question statistics, and multiple-comparison correction. These are
  length associations, not student-utility labels or causal effects.

## Runtime and evidence

The `sft` environment remains pinned to the user's supplied freeze. The existing
legacy TRL overlay requirements are installed under
`envs/sft/overlays/legacy_trl_096_v1` within the configured user storage root.
Only reproduction jobs prepend this overlay, using TRL 0.9.6 and NumPy 1.26.4.
The ordinary `sft` environment still uses NumPy 2.2.6. The overlay dependency
check passes. Model downloads use the shared HF cache. Job temporary files,
HF dataset caches, and student Trainer outputs use per-job directories under `/share/jekml/youyang7/tmp`; final
student adapters are copied to project checkpoints with hash verification.

The new NCSU wrapper uses typed Slurm GRES and only the assigned CUDA device.
It logs physical GPU occupancy and checks free memory twice before each GPU
stage. It intentionally does not use the old admission wrapper, whose node
allowlist is C30/C31/C32/C49 and whose physical-device scan is inappropriate for
the NCSU GRES allocation. No other processes are modified.

The code snapshot is frozen before the smoke job; subsequent jobs validate it.
Generation shards are audited for expected problem/candidate keys and duplicates
before merge. Adapter and evaluation markers bind inputs, code, configurations,
weights, and per-example predictions. The existing SAE completion audits remain
in place. `EXPERIMENT_COMPLETE.json` is written only after all registered student
and SAE stages complete successfully. Queue submission is recorded separately
in `submission_ledger.jsonl` and `submission.json`.

The final student analysis uses crossed training-seed-by-question bootstrap for
short minus base/medium/long and Holm adjustment across the three comparisons.
It publishes accuracy and length figures, all per-example predictions, summary
statistics, and a report. Equal examples do not equalize supervision tokens,
optimizer compute, or loss-normalization effects. This run does not establish
that brevity causes better teaching, even if the short condition wins.

## Status and entrypoints

Preparation, 11 selection tests, four SAE statistics tests, and two gold-answer
conversion regression tests have passed. Initial smoke job 811450 exposed a
new-adapter bug that passed the official full rationale to a numeric verifier.
The failed artifacts and original source snapshot are retained under
`setup/failed_smoke_811450/`. The corrected adapter extracts the final target
before generation scoring and evaluation. No production candidate was generated
under the failed version.

Real-model smoke job 811456 completed successfully: all eight independently
sampled teacher answers were correct, one completion-only SFT step had finite
loss, and the adapter was evaluated on two official smoke questions. The smoke
samples are excluded from the main generation pool.

The complete 36-job DAG was submitted at 2026-09-12 01:42 EDT. Generation jobs
811465/811466/811467/811468 started on the four H100 GPUs of gpu17. Base evaluation
is job 811469, generation merge is 811470, and the final audit/report is 811500.
The intervening student and SAE jobs have explicit after-success dependencies.
Submission is not experiment completion; inspect the live queue and completion
markers for current status.

While the student comparisons were running, the 16 pending SAE GPU jobs were
rescheduled to `gpu_partners/short_gpu`, one L40S per job, with a two-hour limit.
All student training/evaluation remains on H100. The initial schedule assigned SAE extraction, training, and
feature scoring to L40S. The original DAG, job IDs, dependencies,
scientific configuration, and frozen source files were retained. The before/after
scheduler records are in `setup/sae_scheduling_revision_l40s.json`; the first
two extraction jobs started on gpu19. Physical free-memory checks passed on both
assigned L40S devices before model loading.

The explicit stage entrypoint is `scripts/1_40_ncsu_reproduction.py`.
After a successful smoke marker, its `submit` stage submits the complete
dependency-ordered DAG. The final report is expected at
`results/ncsu_phase12_reproduction_v1/exploratory/analysis/report.md`.

Recovery update (2026-09-12): gpu19 physical GPU 2 repeatedly produced illegal
CUDA memory access or stalled, including a conservative math-SDPA diagnostic.
No GPU was reset and no other user process was modified. Failed attempts remain
in `sae/failed_attempts/`. All four accepted activation shards completed with
the original extraction implementation on physical GPU 3. Downstream jobs
811959–811975 replace dependencies cancelled after the first extraction failure.
Layer-17 SAE training/scoring uses the healthy L40S; layers 10 and 23 use H100.
L40S jobs reserve the two available devices and select the healthy allocated
UUID, leaving the unreliable device unused. This scheduling change is recorded
in `setup/recovery_submission.json`; scientific settings and frozen sources are
unchanged. Completed prerequisites are recorded separately because Slurm no
longer accepts dependency references after their controller records expire.
The current final audit job is 811975.

Completion update: final job 811975 completed with exit code 0:0. All 14,096
fresh traces, nine student adapters, ten complete 1,269-example evaluations,
six SAEs, feature scoring, and both SAE audits are complete. The final delivery
check verified 104 bound artifacts and frozen source hashes. See the
[Chinese report](../../../results/ncsu_phase12_reproduction_v1/exploratory/analysis/report_zh.md).
Mean accuracies were base 67.45%, short 66.67%, medium 66.33%, and long 64.20%.
The short-minus-long contrast was +2.47 percentage points (Holm p=0.0852);
this run does not establish improvement over the base student. The primary
SAE confirmed eight short-associated and eight long-associated features.
All experiment jobs have finished.

2026-09-15 storage correction: the current launcher uses administrator-designated scratch. Historical frozen launchers and previously submitted Slurm scripts retain their original paths and require separate migration verification.
