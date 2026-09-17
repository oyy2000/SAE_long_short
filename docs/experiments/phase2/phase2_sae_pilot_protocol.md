# Phase 2: Mixed-Trace TopK SAE Pilot

## Scope

This exploratory pilot starts directly from the sealed 881-question, 16-rollout Qwen2.5-7B-Instruct pool after Phase 0 was paused and Phase 1 was not run. It trains residual-stream dictionaries and measures reconstruction quality and sparsity. It does not have CTV labels and therefore cannot identify student-utility features or support causal teachability claims.

The earlier 2,000-question target is deferred. The current pool contains 14,096 raw trajectories, of which 13,962 are answer-correct. Every trajectory is included in the mixed SAE corpus; correctness and within-question length labels are analysis metadata only.

## Question split and labels

Questions are assigned by a deterministic SHA-256 ordering to 617 train, 132 dev, and 132 test questions. No question appears in more than one split.

Within each question, `short` and `long` are the bottom and top `ceil(20%)` of correct traces. `medium` is one remaining correct trace closest to the median. Remaining correct traces are labeled `other_correct`, and verifier-incorrect traces are labeled `incorrect`. These labels never filter or route SAE training examples.

## Activation extraction

The frozen teacher is Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`. Each stored prompt is replayed through the registered Qwen chat template and followed by its raw assistant solution. Only assistant completion-token activations are retained.

Post-block residual streams are captured at zero-based layers 10, 17, and 23, corresponding approximately to 40%, 65%, and 85% depth in the 28-layer model. Three trace shards are extracted by independent GPU processes and written as hash-audited BF16 safetensors chunks on BeeGFS.

## Token sampling and normalization

All activations remain stored. For the training pilot, deterministic unsigned-64 priority sampling selects 250,000 train, 50,000 dev, and 50,000 test tokens per layer without replacement. Sampling is independent by layer and question split.

Each layer subtracts the train-sample activation mean and applies one scalar chosen so the expected centered squared norm equals the hidden width of 3,584. Dev and test use the unchanged train normalizer.

## SAE matrix

Six dictionaries are trained:

- layers: 10, 17, and 23;
- architecture: TopK SAE;
- expansion factor: 8, giving 28,672 features;
- `k`: 32 and 64;
- seed: 17;
- 1,500 optimization steps with BF16 autocast and FP32 parameters;
- unit-norm decoder directions and an AuxK dead-feature reconstruction term.

The audit requires train/dev/test MSE, explained variance, mean L0, dead-feature fraction, activation frequencies, model hashes, input hashes, and six completion markers. `SAE_PILOT_COMPLETE` means only that all six dictionaries and reconstruction metrics passed the artifact audit.

## Next evidence stage

After training, feature aggregation and falsification may begin. Any utility-conditioned analysis requires returning to Phase 1 or constructing an equivalent held-out student-specific utility label. Causal steering remains downstream of both falsification and an explicit behavioral target.
