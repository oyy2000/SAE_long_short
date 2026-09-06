# Deferred Design: Utility Analysis, Causal Steering, Distillation, and Cross-Student Tests

Phase 2 is now active as the explicitly exploratory 881-question SAE pilot specified in `docs/phase2_sae_pilot_protocol.md`. Because Phase 1 was not run, that pilot has no student-utility labels. The utility-conditioned analyses below remain deferred rather than being silently replaced by length classification.

## Phase 2: utility-explaining SAE

Train one mixed dictionary over short, medium, and long traces rather than label-specific dictionaries. The initial target is residual-stream activations at approximately 40%, 65%, and 85% depth, TopK SAE, expansion factor 8, `k` in `{32, 64}`, question-level train/dev/test splits, one pilot seed, and at least three formal seeds with stable decoder-feature matching.

Feature summaries include activation frequency, mean/max, onset, first-64-token activation, step pooling, and whole-trace activation area. The primary model predicts CTV after controlling question fixed effects, log length, and student NLL. Length classification is diagnostic only.

## Phase 2.5: falsification

Every core feature must survive token injection, paraphrase invariance, lexical scrubbing, absolute-position control, early-prefix prediction, and non-reasoning counterexamples. The causal object is an early trajectory state after sampling diverges, not a pre-generation short/long intent: hidden state is identical before the first sampled token for an identical prompt.

## Phase 3: causal intervention

Use stable utility-positive and utility-negative features, activation-gated intervention, cloned 32/64-token prefixes, cloned KV caches, and identical continuation seeds. Baselines are no steering, dense short-long vectors, SAE shortness, norm-matched random SAE features, orthogonal features, and an explicit concise prompt.

Report unconditional correctness, length, repetition and cost, plus correct-conditioned length and utility. Also report correct usable traces per generated million tokens. Utility-SAE and shortness-SAE outputs must be length matched within 5% before attributing any effect to teachability.

## Phase 4: formal distillation

Use identical question IDs, one trace per question, student initialization, seeds 17/42/73, and both equal-example and equal-supervision-token budgets. The primary endpoint is student exact-match accuracy at fixed supervision-token budget. Coverage and generation cost are reported separately; common-support comparisons use the intersection on which all generation methods yield a correct trace.

## Phase 5: cross-student interaction

Learn student-specific utility directions for Qwen2.5-1.5B and Qwen2.5-3B under the same teacher. Generate two data sets and cross-train both students. The registered target is a diagonal student-by-data advantage, which distinguishes pedagogical compatibility from generic brevity.
