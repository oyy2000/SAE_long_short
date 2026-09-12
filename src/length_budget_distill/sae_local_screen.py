"""Common-distribution reconstruction and dev-only fixed-support feature ranking."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .experiment_io import read_json, write_json_exclusive
from .sae_local_data import ROOT, paths, jsonl, evidence, verify, load_activations
from .sae_feature_analysis import paired_feature_statistics, holm_adjust


def score(config, condition):
    import torch
    from safetensors.torch import save_file
    from .sae_ablation import load_topk_sae

    root, runtime = paths(config)
    output = root / 'feature_screen' / condition
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / config['checkpoint_root'] / condition / 'sae_model.safetensors'
    train = read_json(root / 'sae_training' / condition / 'training_metrics.json')
    verify({'path': str(checkpoint), 'sha256': train['model_sha256']})
    corpus, data = load_activations(config)
    model, mean, scale = load_topk_sae(checkpoint, input_dim=3584, feature_count=28672, k=64,
                                     device=torch.device('cuda'))
    n, f = len(corpus), 28672
    clean = torch.zeros((n, max(r['solution_token_count'] for r in corpus)), dtype=torch.bool)
    for row in corpus:
        clean[row['corpus_index'], row['clean_positions']] = True
    matrices = {key: torch.zeros(n * f, device='cuda') for key in ('full', 'prefix64', 'clean64')}
    totals = {key: dict(sse=0., square=0., count=0, sum=torch.zeros(3584, dtype=torch.float64))
              for key in ('full', 'prefix64', 'clean64')}
    test_trace = torch.tensor([r['question_split'] == 'test' for r in corpus])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data['activations']), 512):
            sl = slice(start, start + 512)
            tr = data['trace_indices'][sl].long()
            pos = data['positions'][sl].long()
            x = (data['activations'][sl].float().cuda() - mean.cuda()) * scale
            with torch.autocast('cuda', dtype=torch.bfloat16):
                reconstruction, values, ids, _ = model(x)
            index = tr.cuda().unsqueeze(1) * f + ids
            masks = {'full': torch.ones_like(tr, dtype=torch.bool), 'prefix64': pos < 64,
                     'clean64': clean[tr, pos]}
            for key, mask in masks.items():
                gpu_mask = mask.cuda()
                matrices[key].scatter_add_(0, index[gpu_mask].flatten(), values[gpu_mask].float().flatten())
                selected = (mask & test_trace[tr]).cuda()
                if selected.any():
                    y = x[selected].float()
                    t = totals[key]
                    t['sse'] += float((reconstruction[selected].float() - y).square().sum())
                    t['square'] += float(y.square().sum())
                    t['sum'] += y.double().sum(0).cpu()
                    t['count'] += len(y)
            if start % 100352 == 0:
                print(f'{condition}: scored {start}/{len(data["activations"])} tokens', flush=True)
    lengths = torch.tensor([r['solution_token_count'] for r in corpus]).float()
    counts = {'full': lengths, 'prefix64': lengths.clamp_max(64),
              'clean64': torch.tensor([len(r['clean_positions']) for r in corpus]).float()}
    means = {key: (value.reshape(n, f).cpu() / counts[key].clamp_min(1).unsqueeze(1)).numpy()
             for key, value in matrices.items()}
    del matrices
    raw_file = runtime / f'trace_feature_means_{condition}.safetensors'
    save_file({key: torch.from_numpy(value) for key, value in means.items()}, str(raw_file))
    stats = {}
    for split in ('dev', 'test'):
        for metric in ('full', 'prefix64', 'clean64'):
            candidates = [i for i, r in enumerate(corpus) if r['question_split'] == split
                          and r['analysis_length_label'] in ('short', 'long')
                          and (metric != 'clean64' or len(r['clean_positions']) == 64)]
            support = {}
            for i in candidates:
                support.setdefault(corpus[i]['problem_id'], set()).add(corpus[i]['analysis_length_label'])
            ix = [i for i in candidates if len(support[corpus[i]['problem_id']]) == 2]
            stats[(split, metric)] = paired_feature_statistics(means[metric][ix], [corpus[i] for i in ix])
    primary = stats[('dev', 'clean64')]
    settings = config['feature_selection']
    selected = []
    for direction, sign in [('short', 1), ('long', -1)]:
        eligible = np.flatnonzero((np.sign(primary['paired_d']) == sign)
                                 & (primary['prevalence'] >= settings['minimum_prevalence']))
        ranked = eligible[np.argsort(-np.abs(primary['paired_d'][eligible]), kind='stable')][:16]
        for rank, fid in enumerate(ranked, 1):
            passed = bool(primary['bh_q_value'][fid] <= settings['discovery_bh_q']
                          and abs(primary['paired_d'][fid]) >= settings['minimum_abs_paired_d'])
            selected.append({'feature_id': int(fid), 'direction': direction, 'rank': rank,
                             'dev_paired_d': float(primary['paired_d'][fid]),
                             'dev_bh_q': float(primary['bh_q_value'][fid]),
                             'passes_discovery_gate': passed})
    # Confirmation is diagnostic only; it does not change which features enter generation.
    q = holm_adjust([stats[('test', 'clean64')]['p_value'][r['feature_id']] for r in selected])
    for row, adjusted in zip(selected, q):
        fid = row['feature_id']
        d = float(stats[('test', 'clean64')]['paired_d'][fid])
        row.update(test_paired_d=d, test_holm_p=float(adjusted),
                   heldout_confirmed=bool(row['passes_discovery_gate'] and adjusted < .05
                                         and d * row['dev_paired_d'] > 0 and abs(d) >= .15))
    for metric in ('full', 'prefix64', 'clean64'):
        filename = output / f'{metric}_feature_statistics.csv'
        keys = ['effect', 'paired_d', 'p_value', 'bh_q_value', 'prevalence']
        with filename.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['feature_id'] + [f'{split}_{key}' for split in ('dev','test') for key in keys])
            writer.writeheader()
            for fid in range(f):
                writer.writerow({'feature_id': fid, **{f'{split}_{key}': float(stats[(split,metric)][key][fid])
                                                       for split in ('dev','test') for key in keys}})
    reconstruction = {}
    for key, t in totals.items():
        denominator = t['square'] - float(t['sum'].square().sum()) / t['count']
        reconstruction[key] = {'token_count': t['count'], 'mse': t['sse']/(t['count']*3584),
                               'explained_variance': 1 - t['sse']/max(denominator, 1e-12)}
    manifest = {'status': 'complete', 'condition': condition, 'checkpoint': evidence(checkpoint),
                'raw_means': evidence(raw_file), 'source': evidence(Path(__file__)),
                'reconstruction_common_test': reconstruction,
                'question_counts': {f'{split}_{metric}': stats[(split,metric)]['question_count']
                                    for split in ('dev','test') for metric in ('full','prefix64','clean64')},
                'features': selected, 'generation_selection_uses_dev_only': True,
                'feature_id_scope': 'This checkpoint only; IDs are not aligned to historical dictionaries.',
                'formal_claim_allowed': False}
    write_json_exclusive(output / 'SCREEN_COMPLETE.json', manifest)

