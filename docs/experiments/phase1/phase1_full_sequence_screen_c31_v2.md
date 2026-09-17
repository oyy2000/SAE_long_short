# C31 local continuation: full-sequence feature screen v2

## Approved protocol change

On 2026-09-07 the user removed the requirement that a feature's first-64-token difference have the same direction as its whole-sequence difference, and requested continuation in `/mnt/local/youyang7/SAE_long_short_c31/activate_runtime.sh`.

The independent configuration is `configs/phase1_full_sequence_screen_c31_v2.json`. Whole-sequence direction, Holm-adjusted p <= 0.05, absolute paired d >= 0.15, three-seed mutual decoder matching, decoder cosine >= 0.5, and activation correlation >= 0.5 are retained. First-64 measurements are retained as diagnostics only. Both full-trace-trained and prefix64-trained dictionaries are screened separately under this rule. Historical scoring code, `confirmed` flags, results, and checkpoints are unchanged.

`src/length_budget_distill/sae_full_sequence_screen.py` re-evaluates stored statistics and matches rather than retraining a dictionary or pretending to regenerate missing data. `scripts/1_40_rescreen_full_sequence_features.py` exposes `preflight` and `screen`. Source markers and candidate hashes are verified before re-screening; the original significance and effect thresholds must match the frozen parent scoring protocol. Results cannot be inferred from remembered feature IDs.

## New runtime

The recovered C31 environment has Python 3.10, PyTorch 2.10.0+cu128, and Transformers 4.48.3. TRL 0.9.6 requires NumPy < 2, whereas the recovered base environment contains NumPy 2.2.6. To avoid modifying that base environment, a separate overlay was installed at:

`/mnt/local/youyang7/SAE_long_short_c31/overlays/legacy_trl_096_v2`

The overlay pins TRL 0.9.6 and NumPy 1.26.4 with supporting packages listed in `configs/phase1_legacy_trl_c31_v2_requirements.txt`. `scripts/slurm/_activate_legacy_sae_c31_v2.sh` first sources the user-provided runtime activation, then prepends only this experiment's overlay to `PYTHONPATH`. It does not modify the user's activation file or base environment packages. The installation report is `logs/legacy_trl_096_v2_install.json` under the local runtime root.

Inside allocation 279213, the new runtime successfully imported `SFTTrainer`, `SFTConfig`, and `DataCollatorForCompletionOnlyLM`; `pip check` reported no broken requirements. Five feature-screen tests and three cache/intervention tests passed. These are implementation checks, not experimental generation or student training.

## Current blocker

The executed preflight is stored at:

`/mnt/local/youyang7/SAE_long_short_c31/runs/phase1_full_sequence_screen_c31_v2/preflight.json`

Its status is `blocked_missing_inputs`. The original parent root remains a symlink into the offline BeeGFS filesystem. The checked project, node-local runtime, and relevant user-owned `/var/tmp` caches do not contain the required six-SAE checkpoints, scoring evidence, and original mixed trajectory corpus. The local Hugging Face cache contains the 1.5B student, not the pinned 7B teacher.

No v2 feature-screen result, intervention-generation job, or student-training job has been produced or submitted. No competing GPU process or user allocation was terminated. The parent artifacts must become accessible, or a hash-verifiable backup location must be supplied. Recovering the environment is not equivalent to recovering experimental data.

## Resume after artifact recovery

Use the C31 runtime inside an active allocation. For a read-only prerequisite check:

```bash
source scripts/slurm/_activate_legacy_sae_c31_v2.sh
"$SAE_EXPERIMENT_PYTHON" scripts/1_40_rescreen_full_sequence_features.py preflight \
  --config configs/phase1_full_sequence_screen_c31_v2.json
```

To re-screen recovered parent statistics:

```bash
"$SAE_EXPERIMENT_PYTHON" scripts/1_40_rescreen_full_sequence_features.py screen \
  --config configs/phase1_full_sequence_screen_c31_v2.json \
  --source-root /path/to/hash_verified/phase1_legacy_trace_sae_distillation_v1
```

The source-root option relocates inputs without weakening their registered hashes. The next execution step is to bind newly selected features and local checkpoint paths into a versioned calibration protocol; the old calibration protocol still intentionally retains its original selection. After lexical probes and paired cache-cloned calibration, proceed to audited all-881-question generation and legacy student SFT. C31-local outputs need hash-verified publication to durable storage after service recovery.
