# Phase 1: Student-Specific Teaching Utility

## Entry rule and Phase-0 relationship

Phase 1 is execution-independent of Phase 0. Existing downstream SFT evidence is sufficient to motivate measuring which traces are useful to a particular student. Phase 0 continues as a parallel robustness audit: it determines whether a short-trace advantage may be attributed beyond token-budget, update-budget, and loss-normalization effects, but it does not determine whether CTV can be measured or validated.

The distinction is claim-level. If Phase 0 fails, Phase 1 may still establish that CTV predicts held-out exact updates and independent policy-SFT gains. It may not use those results to claim that brevity itself reflects better trace content or a latent concise-reasoning mechanism.

## Estimand

For a student anchor `theta`, a correct candidate trace `tau`, and a disjoint validation probe `V`, Counterfactual Teaching Value is

`CTV(tau) = L_V(theta) - L_V(theta - eta * grad l(theta; tau))`.

The implementation reports exact reversible LoRA micro-updates and the first-order gradient inner product. Token-mean and token-sum trace gradients use the same calibrated absolute step size. On the calibration subset, token-sum is also rerun with a per-trace norm-matched step and checked against the exact token-mean update; this is diagnostic only and cannot replace the registered estimands.

## Data and separation

- Existing source: the sealed 881-question Qwen2.5-7B, 16-rollout pool used by Phase 0.
- Extension: 1,119 additional questions, producing exactly 2,000 pool questions with 16 raw rollouts each.
- Question split: 1,400 train, 300 dev, and 300 test, assigned by a stable hash.
- Probe set: GSM8K train indices 2,000–2,255, disjoint from every candidate-pool question.
- Local utility: 32 nearest probe questions under standardized official-solution length, operation count, and base-student NLL.
- Global utility: all 256 probe questions.

The extension reuses the original 881-question pool and is therefore not an independent replication. The manifest records this limitation.

## Exact validation and regressions

An independent 32-question subset of the train split calibrates the shared micro-update step. The 300-question dev split is not used for step-size selection and contributes four candidates per question: bottom-20%-band median, median-length, top-20%-band median, and a distinct deterministic random trace. Each receives exact local/global CTV under both loss reductions.

Within-question regressions estimate length alone, student NLL alone, and length plus NLL. Ranking validation compares CTV against inverse length, inverse student NLL, RSR, SCAS, and LARK-style scores with paired question bootstraps and Holm correction.

Taylor agreement by itself cannot pass Gate 1 because exact CTV and the gradient score share their mathematical definition. The independent endpoint is full one-epoch LoRA SFT selected by each policy.

The base-anchor analysis reports global and local rationale loss as well as global and local answer-only loss. Global rationale CTV is the registered primary endpoint; the other three views are diagnostics. Exact post-update rationale and answer losses share each forward pass to avoid duplicating the dominant validation computation.

## Policy SFT and Gate 1

Eight policies are screened on dev with seed 17: CTV, RSR, SCAS, LARK-style, low NLL, absolute shortest, quantile-short, and deterministic random. CTV, the strongest non-CTV policy, quantile-short, and random are then evaluated with seeds 17, 42, and 73 on the locked test split.

This policy-SFT comparison is an independent validator of candidate ranking under one-trace-per-question training; it is not the final fixed-supervision-token endpoint. Equal-token formal distillation remains reserved for Phase 4 after the mechanism gates pass. Early and final loss-curve summaries are retained to test the “easy to fit but weak to generalize” alternative.

The mid-SFT anchor is the exact first half of the seed-17 random-policy training stream: it preserves the full run's batch order, initialization, warmup, and scheduler horizon, then stops at the registered midpoint. It is not a separately rescheduled half-length training run.

Gate 1 requires:

- CTV gradient ranking to exceed every registered surface baseline on exact global CTV under the primary token-mean reduction;
- CTV-selected SFT to exceed both random and the strongest screened non-CTV policy with a positive crossed seed-by-question 95% interval and Holm-adjusted significance;
- a completed 100-question, four-candidate mid-SFT anchor sensitivity analysis.

Passing Gate 1 permits Phase 1.5. It does not permit an SAE claim.
