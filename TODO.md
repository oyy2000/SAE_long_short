# Prioritized Next Steps

## Active

- Freeze and execute `configs/phase2_sae_pilot_v1.json` on the sealed 881-by-16 raw trajectory pool.
- Extract layers 10, 17, and 23 with disjoint GPU shards and retain all completion-token activations.
- Build deterministic train/dev/test token samples and train the six registered layer-by-k TopK SAEs.
- Require six model hashes, six metrics files, reconstruction/sparsity metrics, and the terminal SAE pilot audit before calling training complete.
- Keep Phase-0 partial evaluation and Phase-1 unsubmitted artifacts intact.

## After SAE pilot completion

- Aggregate sparse per-trace and step-level feature activations on held-out questions.
- Run lexical injection, lexical scrubbing, paraphrase, position, early-prefix, and non-reasoning falsification tests.
- Do not call features student-utility features without a later CTV or independent downstream-training label.
