# Phase 1.5: Fixed-Context Credit Allocation

## Causal question

This phase isolates loss allocation from trace selection. Every `long_*` condition receives exactly the same long prompt-completion token sequence; only per-token supervision weights change. This is distinct from selecting a short trace or pruning the long trace.

Reasoning is segmented by newline first and sentence boundaries second, with sub-eight-token fragments merged while preserving a separate explicit answer step. CTV and NLL step gradients are computed while the complete long trace remains in context. Knapsack selection uses each step's token-mean score multiplied by its token count, matching the total first-order contribution under the downstream token-normalized masked loss; both mean and total scores remain in the audit data.

## Conditions

The seed-17 pilot contains:

- full long-trace supervision;
- equal total weight per reasoning step;
- budget-matched random, high-NLL, and high-CTV whole-step masks;
- long context with answer-only supervision;
- same-source CTV-pruned text;
- paired quantile-short full supervision;
- answer-only pruned text.

For the three budget-matched masks, the supervised reasoning budget is the paired quantile-short trace's reasoning-token count. A whole-step subset must fall within 5%; otherwise the question is dropped from every condition, preserving common support. Answer tokens and EOS remain supervised.

## Audit and gates

Data construction emits a deterministic 100-question sample containing step text, offsets, NLL, CTV, selected indices, and budget gaps. Training refuses to start until an explicit approval marker is bound to that sample and data-manifest hash.

The pilot proceeds to confirmation only if the CTV mask exceeds random-mask and full-long supervision on dev under paired-question intervals. Confirmation uses `long_full`, `long_random_mask`, `long_ctv_mask`, `same_source_pruned_ctv`, and `short_band_full` with seeds 17, 42, and 73 on the locked test split. Gate 1.5 requires the CTV mask to exceed random and full-long conditions under crossed seed-by-question intervals.

Confirmation jobs are not pre-submitted behind a dependency. A decision job validates the pilot decision against its completion-marker hash and submits the confirmation DAG only on a pass; on failure it writes `CONFIRMATION_BLOCKED_BY_PILOT`.

Passing Gate 1.5 permits SAE protocol registration; it is not itself evidence for an SAE mechanism.
