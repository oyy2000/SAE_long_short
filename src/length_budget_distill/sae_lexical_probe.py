"""Small lexical counterexample battery with real token-level SAE activations.

This diagnostic cannot certify a reasoning mechanism. In particular, the fixed
arithmetic paraphrases are controlled examples, not paraphrases of the full GSM8K
pool. All examples and activations are published for inspection.
"""
from __future__ import annotations
import re
from pathlib import Path

from .experiment_io import write_json_exclusive
from .factorial import file_sha256
from .legacy_replication import marker_write


def lexical_probe(model, tokenizer, controller, corpus, config, condition, out):
    import numpy as np
    import torch
    selected = config['features'][condition]
    feature_ids = selected['short'] + selected['long'] + selected['random']
    by_question = {}
    for r in corpus:
        if r['problem_id'] in config['dev_problem_ids'] and r['is_correct']:
            by_question.setdefault(r['problem_id'], []).append(r)
    samples = []
    for q in config['dev_problem_ids']:
        rows = sorted(by_question.get(q, []), key=lambda r: (r['solution_token_count'], r['trace_id']))
        if len(rows) >= 2:
            samples.extend([rows[0], rows[-1]])
        if len(samples) >= config['overlay']['falsification']['natural_dev_examples']:
            break
    samples = samples[:config['overlay']['falsification']['natural_dev_examples']]
    records = []

    def score(prompt, text, kind, identity):
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                tokenize=False, add_generation_prompt=True)
        pids = tokenizer.encode(rendered, add_special_tokens=False)
        tids = tokenizer.encode(text, add_special_tokens=False)
        values = []
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for chunk in hidden[0, len(pids):].split(64):
                values.append(controller.feature_values(chunk).float().cpu())
        handle = controller.layer_module.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                model(input_ids=torch.tensor([pids+tids], device=model.device), use_cache=False)
        finally:
            handle.remove()
        activation = torch.cat(values).numpy() if values else np.zeros((0, len(feature_ids)))
        record = {'kind': kind, 'identity': identity, 'prompt': prompt, 'text': text,
                  'token_ids': tids, 'tokens': [tokenizer.decode([t]) for t in tids],
                  'prompt_token_count': len(pids), 'completion_positions': list(range(len(tids))),
                  'feature_ids': feature_ids, 'activations': activation.tolist()}
        records.append(record)
        return activation, tids

    natural = []
    token_mass = {f: {} for f in feature_ids}
    for row in samples:
        activations, tids = score(row['prompt'], row['solution'], 'natural', row['trace_id'])
        natural.append(activations)
        for slot, fid in enumerate(feature_ids):
            for token, mass in zip(tids, activations[:, slot]):
                token_mass[fid][token] = token_mass[fid].get(token, 0.) + float(mass)
        scrubbed = re.sub(config['overlay']['falsification']['scrub_pattern'], '', row['solution'], flags=re.I)
        score(row['prompt'], scrubbed, 'lexically_scrubbed', row['trace_id'])
    top_tokens = {}
    background = 'The blue notebook is on the wooden shelf. A quiet room has a window and a green curtain.'
    score('Continue the following descriptive text.', background, 'non_reasoning_control', 'description')
    for fid in feature_ids:
        token_ids = sorted(token_mass[fid], key=lambda t: (-token_mass[fid][t], t))
        token_ids = [t for t in token_ids if token_mass[fid][t] > 0][:config['overlay']['falsification']['top_injected_tokens_per_feature']]
        top_tokens[fid] = [{'token_id': t, 'text': tokenizer.decode([t]), 'activation_mass': token_mass[fid][t]} for t in token_ids]
        if token_ids:
            text = background + ' The printed labels read: ' + ' '.join(tokenizer.decode([t]) for t in token_ids) + '.'
            score('Continue the following descriptive text.', text, 'token_injection', f'feature_{fid}')
    for index, pair in enumerate(config['overlay']['falsification']['arithmetic_paraphrase_pairs']):
        for side, text in enumerate(pair):
            score('Show a correct arithmetic example.', text, 'controlled_paraphrase', f'pair_{index}_side_{side}')
    path = out / 'lexical_probe_records.json'
    write_json_exclusive(path, {'condition': condition, 'features': selected, 'records': records,
        'top_tokens': top_tokens, 'formal_claim_allowed': False,
        'claim_boundary': 'Synthetic lexical counterexamples and controlled arithmetic pairs only. Not a semantic invariance certification. Natural position arrays support position-matched analysis; scrubbing shifts positions.'})
    def mean_for(kind):
        arrays = [np.asarray(r['activations']) for r in records if r['kind'] == kind and r['activations']]
        return np.concatenate(arrays).mean(axis=0) if arrays else np.zeros(len(feature_ids))
    kinds = ['natural', 'lexically_scrubbed', 'non_reasoning_control', 'token_injection', 'controlled_paraphrase']
    heat = np.column_stack([mean_for(kind) for kind in kinds])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, max(3, len(feature_ids)*.28)))
    normalizer = np.maximum(heat[:, :1], 1e-8)
    im = ax.imshow(heat/normalizer, aspect='auto', cmap='magma', vmin=0, vmax=3)
    ax.set(yticks=range(len(feature_ids)), yticklabels=[str(f) for f in feature_ids],
           xticks=range(len(kinds)), xticklabels=[k.replace('_', '\n') for k in kinds],
           ylabel='SAE feature ID', title=f'{condition}: lexical diagnostic (mean / natural mean)')
    fig.colorbar(im, ax=ax, label='Activation ratio (display capped at 3)')
    fig.tight_layout()
    fig.savefig(out / 'lexical_probe.png', dpi=180)
    plt.close(fig)
    # First question's natural shortest/longest: exactly the same question.
    display = [r for r in records if r['kind'] == 'natural'][:2]
    if display:
        fig, axes = plt.subplots(len(display), 1, figsize=(15, 5), squeeze=False)
        for ax, row in zip(axes[:, 0], display):
            activation = np.asarray(row['activations'])[:64].T
            im = ax.imshow(activation, aspect='auto', cmap='magma')
            ax.set(yticks=range(len(feature_ids)), yticklabels=[str(f) for f in feature_ids],
                   xticks=range(len(row['tokens'][:64])),
                   xticklabels=[t.replace('\n', '\\n') for t in row['tokens'][:64]],
                   ylabel='Feature ID', title=f'{row["identity"]} | full length {len(row["tokens"])} tokens')
            ax.tick_params(axis='x', rotation=90, labelsize=6)
        fig.tight_layout()
        fig.savefig(out / 'short_long_token_heatmap.png', dpi=180)
        plt.close(fig)
    marker_write(out / 'LEXICAL_PROBE_COMPLETE', {'status': 'complete', 'records_sha256': file_sha256(path),
        'semantic_mechanism_certified': False, 'feature_count': len(feature_ids)})
