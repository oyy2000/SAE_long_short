# Short-versus-long SAE feature analysis

## Result

Across the six layer-by-k SAEs, 66 dev-discovered features passed the registered held-out confirmation rule. For the primary layer-17, k=64 SAE, 5 were short-associated and 8 were long-associated.

The primary contrast is token-normalized and question-paired. Directional agreement in the first 64 completion tokens is required for held-out confirmation, so the result is not based on whole-trace activation area alone.

## Figures

- [01_sae_feature_difference_overview.png](/home/youyang7/projects/SAE_long_short/figures/phase2_short_long_feature_analysis_v1/01_sae_feature_difference_overview.png)
- [02_primary_sae_discovery_volcano.png](/home/youyang7/projects/SAE_long_short/figures/phase2_short_long_feature_analysis_v1/02_primary_sae_discovery_volcano.png)
- [03_heldout_feature_validation.png](/home/youyang7/projects/SAE_long_short/figures/phase2_short_long_feature_analysis_v1/03_heldout_feature_validation.png)
- [04_same_question_token_heatmap.png](/home/youyang7/projects/SAE_long_short/figures/phase2_short_long_feature_analysis_v1/04_same_question_token_heatmap.png)
- [05_token_and_position_diagnostics.png](/home/youyang7/projects/SAE_long_short/figures/phase2_short_long_feature_analysis_v1/05_token_and_position_diagnostics.png)
- [06_short_long_feature_story.png](/home/youyang7/projects/SAE_long_short/figures/phase2_short_long_feature_analysis_v1/06_short_long_feature_story.png)

## Token contexts

See [token_contexts.md](/home/youyang7/projects/SAE_long_short/results/phase2_short_long_feature_analysis_v1/exploratory/analysis/token_contexts.md).

## Claim boundary

The analysis can identify reproducible short-associated and long-associated SAE features. It cannot establish student utility, semantic identity, or causal effects.
