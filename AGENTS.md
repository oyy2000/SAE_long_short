# Project Instructions

## Global Rules

- Do not use emojis in project documentation, logs, or generated reports.
- Maintain a professional, academic, and clean writing style.
- Preserve existing user changes and historical experimental artifacts.

## Project Layout

- Put reusable Python logic in `src/`.
- Put runnable phase-first entrypoints in `scripts/`.
- Put experiment settings in `configs/` rather than hard-coding them.
- Store lightweight metadata in `data/`, metrics and artifacts in `results/`, publication figures in `figures/`, and model adapters in `checkpoints/`.
- Organize documentation by purpose under `docs/`, with experiment documents in `docs/experiments/phaseN/`; maintain the [documentation index](docs/README.md) and the relevant phase index.
- Keep notebooks exploratory and migrate reusable logic into `src/` or `scripts/`.
- Prefer descriptive phase-first script names such as `5_1_generate_capacity_length_traces.py`.

## Script and Workflow Conventions

- Put reusable logic in `src/`, expose it through a small runnable script in `scripts/`, keep settings in `configs/`, write artifacts to `results/`, and place publication plots in `figures/` or the experiment's analysis artifact directory.
- Before implementing a new utility or figure, search existing `src/`, `scripts/`, and figure-generation code for reusable logic and established visual conventions. Prefer extracting or extending shared code over creating a parallel implementation; document the reason when an intentional visual or implementation divergence is necessary.
- Runnable scripts should do one clear job, expose explicit arguments and inputs/outputs, minimize hidden side effects, and log major settings.
- Use phase-first lowercase names in the form `{phase}_{subphase}_{short_description}.py`; do not use numeric-only or vague `final_v2` names.
- Avoid duplicating logic across scripts.

## Result Presentation

- Prefer figures and visual summaries for experimental results: line plots for trends, bars for method comparisons, scatter plots for correlations, pipeline flows, and ablation figures.
- Use tables only when exact values or compact comparisons are necessary.

## Experiment Evidence

- Keep exploratory, smoke-test, and formal evidence in separate directories.
- Do not treat queued jobs, partial shards, or stale summaries as completed experiments.
- A formal result requires complete shard manifests, duplicate/missing-record audits, config and input hashes, complete checkpoints, per-example predictions, aggregate statistics, and a completion marker.
- Do not reuse the historical single-candidate 7B traces as formal evidence for the capacity-by-length factorial experiment.
- Keep historical pilot claims limited to GSM8K. The user-approved Phase 13 expansion has a separate scope below; new cross-dataset claims still require a frozen evaluation protocol and complete audited evidence.

## Active Workspace and Runtime

- The active project root is `/rsstu/users/m/myoon2/satc_568382/yang_ouyang/projects/SAE_long_short` on NCSU HPC. Derive paths from the repository root in reusable code; do not import historical `/home/youyang7`, C31 `/mnt/local`, or `/mnt/beegfs` paths as active defaults.
- The active Unix and Slurm user is `youyang7`; `myoon2` appears in the shared directory path and is not the job owner. Query `id -un` when constructing user-specific scheduler commands.
- The existing Python environment is `/rsstu/users/m/myoon2/satc_568382/yang_ouyang/envs/sft/bin/python`. Existing legacy SFT reproduction jobs use the isolated `/rsstu/users/m/myoon2/satc_568382/yang_ouyang/envs/sft/overlays/legacy_trl_096_v1` overlay; do not silently change their pinned dependencies. Environment setup is documented in [docs/environment/ncsu_sft_environment.md](docs/environment/ncsu_sft_environment.md).
- Install incompatible baseline dependencies in a separate environment or explicit overlay. Preserve existing model caches, frozen execution snapshots, adapters, and predictions.
- `PROJECT_STATUS.md` is the current execution ledger. Protocols and completion markers determine experiment status; historical TODO entries and queued jobs do not.
- Current NCSU workflows use `configs/phase12_ncsu_reproduction_v1.json`, `configs/phase12_ncsu_intervention_v1.json`, and `configs/phase12_ncsu_multi_answer_v1.json`, with matching `ncsu_*.py` modules and phase-first NCSU entrypoints. Inspect these before adapting older launchers; repository presence alone does not make a historical launcher compatible with NCSU.

## Experiment Monitoring Delegation

- When monitoring experiments, delegate routine Slurm status checks, bounded log inspection, and completion-marker checks to a subagent using `gpt-5.6-luna` by default. Explicitly set `reasoning_effort="medium"` for monitoring subagents rather than inheriting the primary agent's reasoning effort. This is explicit authorization to use subagents for experiment monitoring.
- Give each monitoring subagent a bounded task and the relevant job IDs, artifact paths, and evidence requirements. Use `fork_turns="none"` or a limited history when selecting the model explicitly; provide sufficient context in the task message.
- Keep routine monitoring read-only. The primary agent remains responsible for interpreting results, deciding next steps, and authorizing any job or experiment changes. Report queued, running, failed, and verified-complete states distinctly; a scheduler completion state alone is not audited experiment completion.
- Delegate monitoring alongside useful primary-agent work when possible, avoid duplicate polling, and use a larger model only when the monitoring task requires more complex diagnosis.

## Compute Resources: NCSU

- On 2026-09-13 the user explicitly approved continuing with the existing NCSU accounts and partitions, including H100, H200, and L40S resources. NCSU is the default for new work in this folder; the old C30/C31/C32/C49 allowlist does not restrict NCSU launches.
- GPU account: `jekml_gpu`. Standard H100 jobs use partition `gpu`, QOS `gpu`, and typed GRES `gpu:h100:1`. H200 jobs use partition `gpu_partners`, QOS `short_gpu`, and typed GRES `gpu:h200:1`. Check current account associations, QOS time limits, node health, and availability before submission; node availability is not a permanent guarantee.
- L40S is also user-approved for NCSU work (2026-09-13). The single-GPU Slurm resource arguments are `--account=jekml_gpu --partition=gpu_partners --qos=short_gpu --gres=gpu:l40s:1`; set CPU, host-memory, and time requests for the workload. Slurm inventory checked on 2026-09-13 confirms L40S nodes in this partition. Refresh inventory with `sinfo -N -p gpu_partners -o '%N %P %G %t'` before submission and select healthy allocated hardware rather than freezing a node list. `l40` and `l40s` are distinct registered GPU types and must not be substituted silently.
- Within already authorized project work, choosing a suitable L40S instead of H100/H200 does not require renewed machine-allowlist approval. Scheduler eligibility, GPU health, memory fit, and walltime checks still apply at launch.
- Include L40S when scheduling workloads whose expected peak GPU memory plus the registered safety margin fits within measured free GPU memory, and whose runtime fits within the current walltime limit. Use measured workload peaks to choose among L40S, H100, and H200; do not assume they have interchangeable capacity or throughput. Keep the same experiment settings when changing hardware and record GPU model/UUID and timing in the run evidence.
- Scheduler verification on 2026-09-13 found QOS maximum walltimes of 72 hours for `gpu` and 2 hours for `short_gpu`. Treat these as a dated reference, not a permanent entitlement; choose job duration within the current partition, QOS, and association limits.
- On 2026-09-13, `short_gpu` reported group limits of `gres/gpu:h200=4` and `gres/gpu:l40s=20`, plus a per-user GPU limit of 12. These are shared QOS limits and may change. Inspect `sacctmgr show qos` and the pending job's reason when scheduling; an idle node can coexist with `QOSGrpGRES` because the QOS quota is occupied elsewhere. Preserve live jobs and their registered hardware settings while waiting for quota.
- Before submitting GPU work, compare the currently schedulable H100, H200, and L40S routes that meet the measured peak-memory-plus-margin and walltime requirements. Prefer a route that can run now. Queue only when no suitable route is currently available or a frozen scientific requirement fixes the hardware. While a job is pending for `QOSGrpGRES`, `Resources`, or a similar capacity reason, periodically recheck other approved GPU types and switch when a suitable route can execute; an idle node alone does not prove eligibility. Freeze and verify a hardware-only launcher revision, run its GPU admission and representative smoke, then cancel only this project's own duplicate pending jobs and dependencies once the replacement is viable. Never cancel other users' jobs, user-owned `grabgpu` allocations, or a running experiment merely to switch routes. Record both the old and replacement job IDs, route choice, measured GPU model/UUID, memory and timing. Keep the same experiment data, recipe, and evaluation settings when changing hardware.
- CPU account: `jekml_cpu`; use an eligible `compute` or `compute_partners` partition and its permitted QOS. Keep substantial CPU work in Slurm; login nodes are for editing, scheduling, and bounded metadata checks.
- Use `sbatch` for unattended generation, training, evaluation, and dependency-ordered stages. Use `salloc` for interactive allocations, then run commands with `srun --jobid="${SLURM_JOB_ID}" --overlap`. Do not run GPU workloads on login nodes or access compute GPUs by direct SSH outside an allocation.
- Keep interactive `salloc` allocations only while the interactive session is active; do not use a pending `salloc` as an unattended experiment queue. This does not authorize closing user-owned `grabgpu` allocations.
- Prefer one process per allocated GPU and disjoint problem shards. Use tensor/model parallelism only if a workload does not fit safely on one GPU.
- At each GPU job start, inspect `nvidia-smi`, GPU UUIDs, and compute-process occupancy. Restrict admission to the GPUs assigned by Slurm and visible to the process; do not broaden `CUDA_VISIBLE_DEVICES` to unallocated devices.
- When launching a separate GPU worker on an exclusive-process device, run CUDA admission and hardware inspection in a short-lived subprocess that exits before the worker starts. Reuse `ncsu_reproduction.isolated_gpu_preflight`; a parent-held CUDA context caused an observed NCSU launch failure.
- Use memory-fit admission: remaining free memory must cover the expected peak allocation plus a safety margin across repeated checks. High utilization alone does not imply a GPU is unavailable. Include other processes' memory use without interfering with them.
- Never terminate another process, reset a GPU, or cancel another allocation to obtain resources. Preserve user-owned `grabgpu` allocations and `gg` keepalive processes if encountered; they are not grounds for termination.
- Do not treat `--exclusive` or GRES metadata alone as proof of physical isolation. If a GPU has illegal-access or health failures, preserve the failed attempt and reschedule on healthy allocated hardware; do not reset it.
- Reuse the NCSU memory-admission and Slurm helpers in `ncsu_reproduction.py` and downstream NCSU adapters. The legacy `scripts/slurm/_gpu_idle_gate.sh` has a historical cluster allowlist; do not call it unchanged on NCSU.
- NCSU administrator directive (2026-09-15): use job-specific subdirectories under `/share/jekml/youyang7/tmp` for temporary files, HF dataset caches, and intermediate checkpoints. Do not write job data to `/var/tmp`, `/var`, or fall back to system temporary storage. Set `TMPDIR`, `TMP`, and `TEMP` consistently. Reuse existing model caches; publish only final hash-verified adapters and registered evidence to project storage. This supersedes the former node-local temporary-storage guidance. Preserve frozen historical snapshots; submitted Slurm scripts require separate inspection before any claim of migration.
- Incident basis: the administrator reported that our checkpoint writes under `/var/tmp/youyang7-phase13-frozen-823088` and `/var/tmp/youyang7-phase13-frozen-823089` exhausted gpu18's system `/var` filesystem and made the node unusable. A writable directory is not authorization to use it as experiment scratch. Do not carry temporary-storage conventions from another cluster into NCSU without checking the administrator-designated location.
- Before launching or resuming a job, verify the effective temporary, cache, and intermediate-checkpoint paths, including explicit Trainer/SAE output arguments and inherited environment variables. If the designated scratch is unavailable, unwritable, or insufficient, stop with a clear error; never silently fall back to `/tmp` or `/var/tmp`.
- Editing a working-tree launcher does not migrate frozen snapshots, Slurm-spooled batch scripts, or automatic recovery controllers. Keep old-path jobs paused until a separately recorded launcher revision and actual submission paths are verified. Preserve historical snapshots and evidence; do not claim migration or node cleanup from a source-code change alone.
- Estimate temporary-storage requirements separately for calibration, generation, evaluation, and training using actual writes, checkpoint retention, publication staging, and a justified safety margin. Check capacity and applicable quota on the administrator-designated scratch filesystem; do not substitute a generic free-space threshold or system storage fallback. Preserve GPU memory admission and failed attempts. Clean only explicitly identified, user-owned job scratch after terminal-state and publication/archive verification.
- Request realistic NCSU host memory and time for the workload. Keep the legacy prohibition on using misleading `RealMemory` values confined to the historical cluster where that issue was observed.
- Historical cluster-specific allocation commands are preserved in `docs/environment/legacy_cluster_compute_rules_20260913.md`; apply them only when work explicitly returns to C30/C31/C32/C49.

## Baseline Expansion and SAE Analysis

- The user approved the staged expansion on 2026-09-13: preserve the 878/881-question GSM8K evidence as exploratory pilot; use a 7,500-question MATH source pool as the first scale and target 25,000 unique eligible mathematics questions for the later main study. Actual training counts follow held-out-development, decontamination, and common-support audits; repeated traces never count as unique problems.
- Use `configs/phase13_baseline_expansion_v1.json` and `docs/experiments/phase13/phase13_baseline_expansion_experiment_plan_zh.md` for this new branch. This planning config is not a frozen training protocol. Freeze each executable stage before launch, keeping historical protocols unchanged.
- The newly authorized evaluation scope includes GSM8K, MATH-500, GSM8K-Hard, AQuA-RAT, and OlympiadBench English text-only mathematics. Publish claims only after dataset revisions, grading rules, train/evaluation decontamination, complete predictions, statistics, and stage audits are established.
- Preserve GSM8K `test[:50]` for smoke and `test[50:1319]` for the locked evaluation. Label previously observed cohorts explicitly. MATH-500 is not independent OOD relative to MATH training; GSM8K-Hard is a derived robustness set and must retain parent-question grouping.
- Main baseline IDs are fixed: B0 random-correct; B1 unmodified shortest-correct; B2 concise prompt; B3 dense mean-difference; B4 ASC-CES; B5 DAP; B6 TokenSkip-style ratio-conditioned distillation; B7 SAE. Current no-steering shortest-correct results are B1, not B0.
- Pin published-method versions and upstream source commits. Do not label the public legacy ASC mean-difference implementation as formal CES; preserve the published CES objective and KL constraint when implementing B4. Preserve DAP difficulty-aware rewriting and TokenSkip compression-ratio conditioning in training and inference, documenting adaptations to a separate student.
- Freeze generation candidates, answer verification, selection rules, model revisions, student hyperparameters, and tuning budgets. Report complete-pool coverage and costs alongside common-support student comparisons; equal examples, target tokens, optimizer steps, and question exposure are distinct controls.
- Address the observed Answer-marker association before expanding semantic claims: region/lexical diagnostics, same-state feature readback, non-target effects, answer-format/dense/multiple-random controls, and new-question tests. Post-hoc reanalysis is not independent feature confirmation.
- Keep three student seeds separate from SAE seed stability. The 84-run reference matrix is a staged design, not authorization to submit every scale and ablation at once; proceed through recorded dependency and evidence gates.

## Historical Capacity-Length Factorial Protocol

These constraints apply to the preserved factorial branch, not to the separately authorized Phase 13 baseline expansion.

- Formal generators are Qwen2.5-1.5B/3B/7B/14B-Instruct, with 128/256/512-token solution budgets and three candidates per problem-condition.
- Call the 1.5B generator cell a self-distillation control. The pipeline is black-box sequence-level response distillation implemented with SFT, not logit-level KD.
- Use the fixed Qwen2.5-1.5B-Instruct student and fixed LoRA hyperparameters across all factorial conditions.
- Use `configs/capacity_length_factorial_sft_v1.json` for the registered batch-size-4, gradient-accumulation-1 overlay; it is parent-hash-bound so training-only corrections do not invalidate immutable teacher traces.
- Use `test[:50]` only for smoke checks and `test[50:1319]` as the locked formal GSM8K evaluation cohort.
- Primary training is equal-example on the 12-condition common problem intersection. Equal-supervision-token training is a robustness analysis.
- A factorial adapter is complete only when its marker verifies the training-data hash, run-config hash, training and launcher source hashes, and both final LoRA file hashes.
- Use the parent-hash-bound `configs/capacity_length_factorial_eval_v1.json` overlay for batched greedy evaluation; do not silently change the locked split or decoding parameters.
- Preserve the original three-seed `capacity_length_factorial_v1` artifacts as historical evidence. The user-approved reduced rerun uses only seed 17 through `configs/capacity_length_factorial_run_seed17_v1.json`, reuses the immutable parent teacher traces, writes to a separate `capacity_length_factorial_seed17_v1` result/checkpoint root, and must be labeled as a revised single-seed protocol that does not estimate training-seed variability.
