# Phase 1: Historical recipe replication and SAE continuation

This is an independent exploratory branch named `phase1_legacy_trace_sae_distillation_v1`. It does not resume the paused Phase 0 or the earlier teaching-utility experiment.

## Historical recipe

The imported pool contains 881 questions and 14,096 raw 7B traces. The original short/medium/long SFT JSONL files are copied byte-for-byte, not rebuilt from the SAE split. The legacy training function and its evaluation dependency bundle are preserved with SHA256 records. Runtime metadata pins TRL 0.9.6, Transformers 4.48.3, and PyTorch 2.10.0.

The wrapper sets the seed before constructing LoRA and records its initial weight hash; this is an explicit reproducibility addition, not a promise of bitwise identity to historical runs. All training conditions use the original completion-only TRL trainer, one epoch, batch size 4, learning rate 2e-5, rank-4 LoRA, and maximum input length 2048. The custom controlled-loss trainer is not used.

Evaluation uses the unchanged legacy evaluator and a copied, pre-sliced question/gold fixture for GSM8K test[50:1319]. Consequently the evaluator CLI offset is 0 while the logical cohort remains the original 1,269 questions. The fixture is built from historical **gold answers**, never generated answers. A separate C49 base replay diagnoses differences from the C31 base evaluation without replacing either prediction artifact.

## Execution and evidence

Use `scripts/1_31_legacy_replication.py` with `--config configs/phase1_legacy_trace_sae_distillation_v1.json`. Its stages are `prepare`, `submit`, `worker`, `train`, `evaluate`, and `analyze`. Preparation and submission are exclusive operations: existing independent roots or submission manifests are not overwritten.

Results are under `results/phase1_legacy_trace_sae_distillation_v1`, which resolves to BeeGFS. `IMPORT_COMPLETE.json` binds imports and run configs; `RUNTIME_SOURCES.json` seals replication implementation files. Do not edit sealed runtime files during queued/running work. Per-adapter `TRAIN_COMPLETE` and per-model `EVALUATION_COMPLETE` verify their artifacts. Scheduling revisions are appended separately; the first submission manifest is historical, not the current authoritative assignment after a revision.

The nine-run replication gate requires short mean accuracy within 70.42% +/- 1 percentage point, above the current base and above long. All three seeds and all predictions must be present. A failed gate records the result and exits nonzero; it does not trigger automatic hyperparameter changes. An incomplete run has no replication-completion marker.

## SAE continuation

`scripts/1_34_legacy_sae_continuation.py register` queues a CPU preparation job after the replication analysis job succeeds. Preparation independently validates the completed gate before creating or launching SAE work. It reuses only hash-verified activation samples from the identical copied raw pool, copies the mixed corpus and activation samples into this branch, and retains audited residual-chunk pointers. Each derived manifest explicitly records unchanged sample bytes and the new training-seed protocol.

The SAE matrix is full-trace-balanced versus prefix64-trace-balanced, each with seeds 17/42/73, layer 17, TopK 64, and the existing expansion-8 training recipe. The existing SAE trainer and scorer are reused. Question splits remain 617/132/132 for feature learning and confirmation only; they do not cap future student training at 617 questions. Cross-seed matching and training curves are generated after six models and their scoring artifacts complete.

## Subsequent intervention stage

Lexical falsification, causal calibration, intervention generation, and intervention-data student SFT are subsequent work. They are not represented as completed or already running by the replication or SAE markers. The registered design uses all 881 questions for generation, 16 candidates per method/question, shortest-correct selection, and the all-method correct question intersection. Historical rank controls must be retrained on that intersection; there is no fallback. Equal-example and equal-target-token results remain separate.

No accuracy or feature-utility claim is permitted from launch status, incomplete shards, or training checkpoints alone. The historical GSM8K cohort has already been observed; these results are exploratory rather than new untouched confirmation.

## Verified update: replication and six SAE models complete

The nine-run replication completed with a passed audit: short mean 70.03%, medium 68.90%, long 67.38%. The registered C31 base is 68.01%. The separate C49 base replay is 67.14% and exactly reproduces historical text, extracted answers, correctness, and output lengths on all 1,269 questions. Both base artifacts are preserved; no baseline was silently substituted.

SAE workers 279204 (C31) and 279205 (C49) finished six models, and audit 279206 produced `sae/SAE_TRAINING_AND_SCORING_COMPLETE`. Full-trace-balanced test reconstruction explained variance is approximately 0.7781–0.7785; prefix64-balanced is 0.7940–0.7946 on their respective sampling distributions. These values do not by themselves establish that prefix64 is better on a common distribution.

The **combined** feature screen returned zero stable confirmed features. This does not mean there are no reproducible differences. A full-sequence long feature matches across seeds 17/42/73 as 10498/18161/20175, with decoder cosines approximately 0.547/0.544/0.791 and matching activation correlations above 0.5. Its whole-trace effects replicate, but seed 73's first-64 sign does not. The inherited `confirmed` flag requires first-64 sign replication as well as whole-trace confirmation. The closest prefix64-trained triangle also falls below the 0.5 cosine threshold (approximately 0.495) and fails the early sign check.

`scripts/1_39_analyze_legacy_sae_feature_screen.py` publishes new diagnostic figures under `sae/feature_screen_diagnostics/`, without altering the original scoring or selection artifacts. Its six-seed diagnostic layout intentionally differs from the old three-dictionary single-seed figure, while retaining the established full/early and short/long colours.

The cache-cloned intervention implementation and dev-calibration entrypoint are available as `sae_paired_intervention.py` and `scripts/1_37_legacy_sae_calibration.py`. Tests cover independent cache storage, first-continuation intervention timing, observed suppression, and the perturbation norm cap. Preparation under the original screen wrote `intervention_calibration_v1/NO_STABLE_FEATURES.json`; **no intervention-generation or downstream SFT job was launched**. GPU generation and lexical-probe execution are not yet validated end-to-end in this branch. A separate exploratory full-sequence-only long-feature branch requires an explicit protocol decision; it must not overwrite or relax the original screen.
