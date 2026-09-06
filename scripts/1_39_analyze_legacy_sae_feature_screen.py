#!/usr/bin/env python3
"""Explain the completed SAE candidate screen without revising its decisions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from length_budget_distill.experiment_io import read_json, write_json_exclusive
from length_budget_distill.factorial import file_sha256
from length_budget_distill.legacy_replication import config_load, root_for, marker_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = config_load(args.config)
    root = root_for(config) / 'sae'
    summary_path = root / 'analysis/sae_summary.json'
    summary = read_json(summary_path)
    mark = marker_read(root / 'SAE_TRAINING_AND_SCORING_COMPLETE')
    if mark['summary_sha256'] != file_sha256(summary_path):
        raise ValueError('SAE audit hash mismatch.')
    output = root / 'feature_screen_diagnostics'
    output.mkdir(exist_ok=False)
    runs = {}
    evidence = []
    for condition in config['sae']['conditions']:
        for seed in config['sae']['seeds']:
            path = root / f'seed_{seed}/condition_scores/{condition}/discovered_features.json'
            runs[(condition, seed)] = read_json(path)['candidates']
            evidence.append({'path': str(path), 'sha256': file_sha256(path)})
    triples = []
    a,b,c = config['sae']['seeds']
    for condition in config['sae']['conditions']:
        maps = {}
        for x,y in [(a,b),(a,c),(b,c)]:
            maps[(x,y)] = {int(r['left_feature_id']):r for r in summary['pair_matches'][f'{condition}__{x}_{y}']
                           if r['mutual_nearest_neighbor'] and r['same_direction']}
        for fid, ab in maps[(a,b)].items():
            ac = maps[(a,c)].get(fid)
            bc = maps[(b,c)].get(ab['right_feature_id'])
            if not ac or not bc or ac['right_feature_id'] != bc['right_feature_id']:
                continue
            ids = {a:fid,b:ab['right_feature_id'],c:ac['right_feature_id']}
            features = {s:next(r for r in runs[(condition,s)] if r['feature_id']==f) for s,f in ids.items()}
            triples.append({'condition':condition,'ids':ids,'direction':ab['left_direction'],
                'minimum_decoder_cosine':min(r['decoder_cosine'] for r in [ab,ac,bc]),
                'minimum_activation_correlation':min(r['test_trace_activation_correlation'] for r in [ab,ac,bc]),
                'all_confirmed':all(r['confirmed'] for r in features.values()),
                'all_first64_directions_replicated':all(r['first_64_direction_replicated'] for r in features.values()),
                'features':features})
    write_json_exclusive(output / 'feature_screen_diagnostics.json', {
        'original_stable_feature_count': len(summary['stable_features']),
        'mutual_same_direction_triangles':triples, 'input_evidence':evidence,
        'original_summary_sha256':file_sha256(summary_path),
        'changes_original_selection':False, 'intervention_launched':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    # Reuse the project's established short/long and full/early colours. This
    # diagnostic has six seed runs, unlike the original three-dictionary plot.
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for i, condition in enumerate(config['sae']['conditions']):
        for x,y in [(a,b),(a,c),(b,c)]:
            rows = summary['pair_matches'][f'{condition}__{x}_{y}']
            axes[i].scatter([r['decoder_cosine'] for r in rows],
                            [r['test_trace_activation_correlation'] for r in rows],s=25,label=f'{x} vs {y}')
        axes[i].axvline(.5,ls='--',color='grey',lw=1)
        axes[i].axhline(.5,ls='--',color='grey',lw=1)
        axes[i].set(title=condition.replace('_',' '),xlabel='Decoder cosine',ylabel='Held-out trace activation correlation')
        axes[i].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'cross_seed_matching.png',dpi=180)
    fig.savefig(output / 'cross_seed_matching.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    for ax,condition in zip(axes,config['sae']['conditions']):
        rows=[]
        for direction in ['short','long']:
            candidates=sorted((r for r in runs[(condition,17)] if r['direction']==direction),key=lambda r:r['discovery_rank'])[:5]
            rows.extend(candidates)
        array=np.array([[r['metrics'][m]['test']['paired_d'] for m in ['token_mean_activation','first_64_token_mean_activation']] for r in rows])
        im=ax.imshow(array,cmap='RdBu',vmin=-2,vmax=2,aspect='auto')
        ax.set(yticks=range(len(rows)),yticklabels=[f'{r["direction"]} #{r["feature_id"]}' for r in rows],
               xticks=[0,1],xticklabels=['Full sequence','First 64 tokens'],title=condition.replace('_',' '))
        for (y,x),value in np.ndenumerate(array):ax.text(x,y,f'{value:+.2f}',ha='center',va='center',fontsize=8)
    fig.suptitle('Seed 17 held-out paired differences: positive = short, negative = long')
    fig.tight_layout()
    fig.savefig(output / 'short_long_feature_differences.png',dpi=180)
    fig.savefig(output / 'short_long_feature_differences.pdf')
    plt.close(fig)
    best=[]
    for condition in config['sae']['conditions']:
        rows=[r for r in triples if r['condition']==condition]
        if rows: best.append(max(rows,key=lambda r:r['minimum_decoder_cosine']))
    fig,axes=plt.subplots(1,len(best),figsize=(10,4),squeeze=False)
    for ax,triangle in zip(axes[0],best):
        for offset,(metric,label,color) in enumerate([
            ('token_mean_activation','Full sequence','#7570B3'),
            ('first_64_token_mean_activation','First 64 tokens','#1B9E77')]):
            vals=[]; lo=[]; hi=[]
            for seed in config['sae']['seeds']:
                m=triangle['features'][seed]['metrics'][metric]['test']
                vals.append(m['effect']);lo.append(m['effect']-m['ci_low']);hi.append(m['ci_high']-m['effect'])
            ax.errorbar(np.arange(3)+(offset-.5)*.15, vals, yerr=[lo,hi],fmt='o',capsize=3,color=color,label=label)
        ax.axhline(0,color='grey',lw=1)
        ax.set(xticks=range(3),xticklabels=[f'Seed {s}\n#{triangle["ids"][s]}' for s in config['sae']['seeds']],
               ylabel='Short minus long mean activation',title=triangle['condition'].replace('_',' '))
        ax.legend(frameon=False)
    fig.suptitle('Best matched long features: whole-trace effects versus early effects')
    fig.tight_layout()
    fig.savefig(output / 'matched_long_full_vs_prefix64.png',dpi=180)
    fig.savefig(output / 'matched_long_full_vs_prefix64.pdf')
    plt.close(fig)
    text = '''# SAE feature-screen diagnostics

Six SAE models and all scoring artifacts are complete. The original strict screen found zero features; it is not changed here.

The full-trace dictionaries contain a mutually matched long-associated feature (seed 17: 10498; seed 42: 18161; seed 73: 20175). Its three pairwise decoder cosines exceed 0.5 and held-out activation correlations exceed 0.5. All three whole-trace effects are negative and significant, but seed 73's first-64 effect changes sign and its interval includes zero. The inherited confirmation rule additionally requires first-64 sign replication, so this whole-trace match is excluded from the combined screen.

The closest prefix64-trained long feature triangle also fails: seed 17 versus 73 decoder cosine is about 0.495, below the registered 0.5, and seed 73 does not replicate the early sign. Thresholds have not been rounded down or relaxed.

These results distinguish reproducible whole-sequence association from stable early-prefix association. They do not establish a pre-generation intent feature or a student-utility feature. The intervals in the plot are the existing per-feature question-paired intervals, not simultaneous family-wise intervals.

![Short and long feature differences](short_long_feature_differences.png)

![Matched long features](matched_long_full_vs_prefix64.png)

![Cross-seed matching](cross_seed_matching.png)

Next decision: retain the combined screen, or explicitly register a separate exploratory full-sequence-long suppression branch. Main generation and student SFT have not started.
'''
    (output/'README.md').write_text(text,encoding='utf-8')
    print(output,flush=True)


if __name__=='__main__':main()
