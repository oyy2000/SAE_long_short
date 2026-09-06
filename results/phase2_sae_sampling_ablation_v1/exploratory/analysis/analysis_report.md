# SAE full-sequence versus prefix-64 sampling ablation

## Registered comparison

All three TopK SAEs use layer 17, 28,672 features, k=64, seed 17, 250,000 training tokens, and 1,500 optimizer steps. The only intended difference is how training tokens are sampled: full-sequence token-uniform, full-sequence trace-balanced, or first-64-token trace-balanced. No explicit length penalty or short/long label enters the SAE loss.

All SAEs are evaluated on the same complete dev/test traces. Candidate features are discovered on dev, confirmed on test, and compared across dictionaries by mutual decoder nearest neighbors, decoder cosine, common-test activation correlation, and short/long direction agreement.

## Results

The training exposure Spearman correlations between trace length and sampled-token count are 0.768, 0.006, and 0.006, respectively.

Common-test explained variance (all tokens / first 64 tokens):

- `full_token_uniform`: 0.7765 / 0.7743
- `full_trace_balanced`: 0.7763 / 0.7748
- `prefix64_trace_balanced`: 0.6796 / 0.7943

Held-out confirmed short-associated / long-associated candidates:

- `full_token_uniform`: 5 / 8
- `full_trace_balanced`: 4 / 10
- `prefix64_trace_balanced`: 9 / 11

Stable candidate matches across dictionary pairs:

- `full_token_uniform__full_trace_balanced`: 9
- `full_token_uniform__prefix64_trace_balanced`: 0
- `full_trace_balanced__prefix64_trace_balanced`: 1
- Strict features stable across all three dictionaries: 0

## Interpretation boundary

This is a single-SAE-seed exploratory falsification. It can identify sampling and position artifacts, but it does not establish that a feature is semantic, student-utility-related, or causally changes reasoning. Token signatures and relative-position profiles should be treated as diagnostics requiring the lexical falsification tests in Phase 2.5.

An explicit length penalty is not recommended for the discovery SAE: it would make length separation partly supervised by construction. Trace-balanced sampling is the clean control for long-trace over-weighting; downstream utility should be modeled after feature extraction.

## Figures

- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/01_training_sampling_exposure.png`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/01_training_sampling_exposure.pdf`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/02_reconstruction_and_feature_yield.png`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/02_reconstruction_and_feature_yield.pdf`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/03_full_vs_first64_feature_effects.png`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/03_full_vs_first64_feature_effects.pdf`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/04_candidate_dictionary_matching.png`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/04_candidate_dictionary_matching.pdf`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/05_confirmed_feature_position_profiles.png`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/05_confirmed_feature_position_profiles.pdf`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/06_confirmed_feature_token_signatures.png`
- `/home/youyang7/projects/SAE_long_short/figures/phase2_sae_sampling_ablation_v1/06_confirmed_feature_token_signatures.pdf`
