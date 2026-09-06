# Phase-0 Controlled Trace-Length Observation Protocol

All paths and commands in this document are relative to the standalone repository root. The immutable parent generation evidence is exposed through the repository-local `results/capacity_length_ranked_sampling_7b_v1` link and remains canonical at its historical location.

## Purpose

This experiment tests whether the previously observed short-trace advantage survives controls for trace selection, supervision volume, training exposure, loss normalization, and answer-only supervision. It is the required gate before student teaching-utility modeling or SAE training.

The registered experiment is `trace_length_observation_gate0_v1`. It is restricted to GSM8K and uses:

- teacher: Qwen2.5-7B-Instruct;
- student: Qwen2.5-1.5B-Instruct with rank-4 LoRA;
- 16 natural teacher rollouts per training question;
- training seeds 17, 42, and 73;
- locked GSM8K `test[50:1319]` evaluation.

The evaluation cohort has been observed in earlier experiments. Consequently, the resulting evidence is comparative Gate-0 evidence and not a fresh confirmatory test.

## Candidate selection

The implementation reuses the immutable 7B raw generation pool. It does not reuse the historical selected shortest/lower-median/longest datasets.

For each question:

1. re-run the frozen numeric answer verifier;
2. remove incorrect responses;
3. perform whitespace-normalized exact-text deduplication;
4. require at least four unique correct traces;
5. rank by student-tokenizer completion length;
6. define the short and long bands as the bottom and top 20 percent;
7. select the median-length member within each tail band;
8. select the trace closest to the global median for the medium condition.

The within-band representative avoids absolute extrema whenever a tail band contains more than one trace. Every selected trace retains the complete natural response; no response is cropped.

## Fairness estimands

Three data schedules are materialized once and reused by every seed and loss condition:

- `equal_examples`: identical problem IDs and one complete trace per problem in every rank;
- `equal_target_tokens`: whole traces sampled without replacement to the smallest full-dataset completion-token total;
- `equal_processed_tokens`: every problem is exposed at least once and complete traces are repeated to the largest full-dataset non-padding model-input-token total, including the chat-formatted prompt, completion, and terminal EOS.

The last two schedules are intentionally distinct. Both retain the complete paired question cohort and repeat shorter arms deterministically. `equal_target_tokens` matches cumulative completion supervision to the largest one-pass completion total, while `equal_processed_tokens` matches cumulative non-padding forward/backward input-token exposure to the largest one-pass model-input total. Whole-trace scheduling means each target is matched up to a bounded gap of at most 512 tokens. The analysis reports completion-token updates, non-padding model-input updates, padding-inclusive model tokens, effective loss tokens, unique questions, repeated exposures, and optimizer steps separately. Padding-inclusive tokens remain a diagnostic rather than the matching variable because batch-local padding depends on deterministic batch composition.

## Loss controls

The custom Phase-0 trainer explicitly implements:

- `token_mean`: sum of target-token losses divided by the number of active target tokens in the batch;
- `sequence_mean`: mean of per-sequence mean losses;
- `full_completion`: all completion tokens and terminal EOS receive loss;
- `answer_only`: only the content of the final `Answer:` line and terminal EOS receive loss.

Prompt and separator tokens always have zero loss weight. Packing and silent truncation are forbidden. An example exceeding the registered maximum length causes a hard failure.

Training uses the same shared student math prompt as evaluation and applies the registered student's chat template before appending the complete teacher response. The original teacher-generation prompt is retained only as provenance and is never used as the student input.

The full factorial contains:

\[
3\ \text{budgets}\times2\ \text{normalizations}\times2\ \text{loss masks}
\times3\ \text{length ranks}\times3\ \text{seeds}=108\ \text{adapters}.
\]

## Evaluation and Gate 0

Every adapter is evaluated with greedy decoding on the same 1,269 GSM8K questions. The analysis reports exact-match accuracy, output length, training accounting, three-seed crossed seed/question bootstrap intervals, and paired rank contrasts.

Gate 0 uses full-completion loss and requires a positive short-minus-long accuracy effect whose 95 percent crossed-bootstrap lower bound is above zero in all four cells:

- equal-target-tokens with token-mean loss;
- equal-target-tokens with sequence-mean loss;
- equal-token-updates with token-mean loss;
- equal-token-updates with sequence-mean loss.

If every cell passes, Phase 1 teaching-utility validation may proceed. If any cell fails, the completion marker records `stop_sae_story`; the experiment remains technically complete, but the latent-feature explanation is not advanced.

## Evidence layout

- frozen protocol: `results/trace_length_observation_gate0_v1/formal/protocol/`;
- selected traces and schedules: `results/trace_length_observation_gate0_v1/formal/data/`;
- training manifests and audit: `results/trace_length_observation_gate0_v1/formal/training/`;
- adapters: `checkpoints/trace_length_observation_gate0_v1/formal/`;
- predictions: `results/trace_length_observation_gate0_v1/formal/evaluation/`;
- analysis: `results/trace_length_observation_gate0_v1/formal/analysis/`;
- publication figures: `figures/trace_length_observation_gate0_v1/formal/`;
- final marker: `results/trace_length_observation_gate0_v1/PHASE0_COMPLETE`.

Formal completion requires all 108 hash-verified adapters, 109 evaluation runs including the base model, per-example predictions, identical evaluation support, aggregate statistics, analysis artifacts, and the final completion audit.

## Commands

Freeze the parent-hash-bound protocol before creating formal data:

```bash
python scripts/24_0_freeze_trace_observation_protocol.py
```

Validate submission commands without changing scheduler state:

```bash
python scripts/24_0_submit_trace_observation_gate0.py --dry-run
```

Submit the dependency-ordered multi-node DAG only after reviewing the dry run and current physical GPU occupancy:

```bash
python scripts/24_0_submit_trace_observation_gate0.py
```

The submission uses one process per admitted GPU and three disjoint training shards on C30, C31, and C32. The shared GPU-idle gate is applied before every training or evaluation launch.

## Stage boundary

This implementation does not train an SAE and does not create Phase 2–5 evidence. Those stages remain blocked on the hash-audited Gate-0 decision. Existing exploratory one-step trace-utility artifacts are retained as separate historical evidence and are not relabeled as Phase-0 results.
