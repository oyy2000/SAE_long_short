"""Audited paired teacher analysis and publication figures for the local pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .experiment_io import read_json, write_json_exclusive
from .sae_local_data import ROOT, paths, evidence, verify, jsonl
from .utility_analysis import paired_question_bootstrap


def matching_random(spec, specs, *, exact_count=False):
    candidates = [s for s in specs if s['mode']=='random' and s['dictionary']==spec['dictionary']
                  and s['rho']==spec['rho'] and s['start']==spec['start']]
    exact = [s for s in candidates if s['count']==spec['count']]
    return (exact or ([] if exact_count else candidates) or [None])[0]


def freeze_analysis(config):
    root, _ = paths(config)
    protocol = read_json(root/'protocol/generation_protocol.json')
    eligible = [s['name'] for s in protocol['specs'] if s['mode'] in ('short','joint')
                and s['start']==0 and matching_random(s,protocol['specs'],exact_count=True)]
    write_json_exclusive(root/'protocol/analysis_protocol.json', {
        'selection_cohort':'dev', 'confirmation_cohort':'test', 'eligible_target_conditions':eligible,
        'selection_uses_test_outcomes':False, 'rank_rule':['lowest_mean_all_output_tokens','highest_accuracy','weakest_rho','name'],
        'gate':config['selection_gate'],
        'primary_length':'all outputs including incorrect and capped outputs',
        'random_rule':'same SAE, count, requested norm and onset; gated long suppression is diagnostic only',
        'random_length_gate':'paired target-minus-random mean-length 95% CI wholly below zero',
        'accuracy_gate_scope':'point estimate only; does not demonstrate noninferiority',
        'interval_scope':'pointwise exploratory paired question bootstrap; full grid is not multiplicity corrected',
        'student_scope':'conditional on the preselected dev condition passing the same test gate',
        'source':evidence(Path(__file__))})


def audited_records(config, split):
    root, _ = paths(config)
    protocol = read_json(root/'protocol/generation_protocol.json')
    rows=[]
    for shard in range(config['generation']['shards']):
        marker=root/'generation'/split/f'shard_{shard}'/'GENERATION_COMPLETE.json'
        m=read_json(marker)
        verify(m['predictions']);verify(m['protocol'])
        part=jsonl(m['predictions']['path'])
        if len(part)!=m['records']: raise ValueError('Shard row count mismatch')
        rows.extend(part)
    expected={(pid,s['name']) for pid in protocol['cohorts'][split] for s in protocol['specs']}
    keys=[(r['problem_id'],r['condition']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Duplicate, missing, or extra generation cells')
    baseline={r['problem_id']:r for r in rows if r['condition']=='no_steering'}
    specs={s['name']:s for s in protocol['specs']}
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl')}
    from .verifiers import extract_final_answer,verify_answer
    for row in rows:
        spec=specs[row['condition']]
        if row['spec']!=spec: raise ValueError('Changed intervention spec')
        if row['seed']!=baseline[row['problem_id']]['seed']: raise ValueError('Unpaired RNG seeds')
        if row['output_token_count']!=len(row['token_ids']): raise ValueError('Wrong output length')
        if row['token_ids'][:spec['start']]!=baseline[row['problem_id']]['token_ids'][:spec['start']]:
            raise ValueError('Changed natural prefix')
        gold=extract_final_answer(corpus[row['problem_id']]['gold_answer'])
        prediction=extract_final_answer(row['response'])
        if row['gold_answer']!=gold or row['is_correct']!=verify_answer(prediction,gold):
            raise ValueError('Incorrect gold or verification')
        diag=row['diagnostics']
        if diag['modified_positions']>diag['eligible_forward_positions']:
            raise ValueError('Invalid intervention exposure')
        # BF16 addition slightly changes the requested float32 norm.
        if diag['max_delta_to_hidden_norm_fraction']>spec['rho']+.01:
            raise ValueError('Actual perturbation exceeded quantization tolerance')
    return rows


def analyze(config, split):
    root, _ = paths(config)
    output=root/'analysis'/split
    output.mkdir(parents=True,exist_ok=False)
    protocol=read_json(root/'protocol/generation_protocol.json')
    analysis=read_json(root/'protocol/analysis_protocol.json')
    rows=audited_records(config,split)
    by_condition={s['name']:{r['problem_id']:r for r in rows if r['condition']==s['name']} for s in protocol['specs']}
    base=by_condition['no_steering']
    n=len(base)
    gate=config['selection_gate']
    def contrast(left,right,field,correct_only=False):
        support=set(left)&set(right)
        if correct_only: support={p for p in support if left[p]['is_correct'] and right[p]['is_correct']}
        if len(support)<2: return {'estimate':None,'ci_low':None,'ci_high':None,'question_count':len(support)}
        return paired_question_bootstrap({p:float(left[p][field]) for p in support},
            {p:float(right[p][field]) for p in support},samples=gate['bootstrap_samples'],seed=gate['bootstrap_seed'])
    summaries=[]
    for spec in protocol['specs']:
        cell=by_condition[spec['name']]
        random=matching_random(spec,protocol['specs'])
        exact=matching_random(spec,protocol['specs'],exact_count=True)
        length=contrast(cell,base,'output_token_count')
        acc=contrast(cell,base,'is_correct')
        random_diff=contrast(cell,by_condition[random['name']],'output_token_count') if random else None
        mean_length=float(np.mean([r['output_token_count'] for r in cell.values()]))
        base_length=float(np.mean([r['output_token_count'] for r in base.values()]))
        diagnostic_keys=['mean_delta_to_hidden_norm_fraction','max_delta_to_hidden_norm_fraction',
                         'mean_short_activation_before','mean_short_activation_after',
                         'mean_long_activation_before','mean_long_activation_after','target_code_changed_fraction']
        summary={'condition':spec['name'],'spec':spec,'n':n,'correct':sum(r['is_correct'] for r in cell.values()),
                 'accuracy':float(np.mean([r['is_correct'] for r in cell.values()])),
                 'mean_output_tokens':mean_length,'relative_length_reduction':1-mean_length/base_length,
                 'hit_cap_fraction':float(np.mean([r['hit_max_new_tokens'] for r in cell.values()])),
                 'length_minus_base':length,'accuracy_minus_base':acc,
                 'both_correct_length_minus_base':contrast(cell,base,'output_token_count',True),
                 'random_condition':random['name'] if random else None,
                 'random_count_matched':bool(exact),'length_minus_random':random_diff,
                 'diagnostics':{k:float(np.mean([r['diagnostics'][k] for r in cell.values()])) for k in diagnostic_keys}}
        d=summary['diagnostics']
        d['short_code_change']=d['mean_short_activation_after']-d['mean_short_activation_before']
        d['long_code_change']=d['mean_long_activation_after']-d['mean_long_activation_before']
        checks={'registered_for_selection':spec['name'] in analysis['eligible_target_conditions'],
                'length_point':summary['relative_length_reduction']>=gate['minimum_relative_length_reduction'],
                'length_interval':length['ci_high'] is not None and length['ci_high']<0,
                'accuracy_point':acc['estimate']>=-gate['maximum_accuracy_drop'],
                'matched_random_interval':bool(exact and random_diff and random_diff['ci_high'] is not None and random_diff['ci_high']<0)}
        summary['gate_checks']=checks
        summary['passes_gate']=all(checks.values())
        summaries.append(summary)
    eligible=[s for s in summaries if s['passes_gate']]
    eligible.sort(key=lambda s:(s['mean_output_tokens'],-s['accuracy'],s['spec']['rho'],s['condition']))
    if split=='dev':
        selected=eligible[0]['condition'] if eligible else None
        decision={'selected_condition':selected,'passing_conditions':[s['condition'] for s in eligible],
                  'student_triggered':False,'reason':'Test confirmation pending' if selected else 'No dev condition passed the registered teacher gate'}
        write_json_exclusive(root/'DEV_SELECTION.json',decision)
    else:
        dev=read_json(root/'DEV_SELECTION.json')
        selected=dev['selected_condition']
        chosen=next((s for s in summaries if s['condition']==selected),None)
        passed=bool(chosen and chosen['passes_gate'])
        decision={'selected_condition':selected,'preselected_test_gate_passed':passed,'student_triggered':False,
                  'student_authorized_by_protocol':passed,
                  'reason':'Teacher gate passed; student follow-up required' if passed else 'Registered teacher gate failed; no new student SFT triggered'}
        write_json_exclusive(root/'TEACHER_DECISION.json',decision)
    report={'status':'complete','split':split,'row_count':len(rows),'question_count':n,
            'condition_count':len(summaries),'summaries':summaries,'decision':decision,
            'audit':'all registered cells, paired seeds, natural prefixes, golds and shard hashes passed',
            'formal_claim_allowed':False,'source':evidence(Path(__file__))}
    write_json_exclusive(output/'teacher_summary.json',report)
    with (output/'teacher_metrics.csv').open('x',newline='') as handle:
        fields=['condition','accuracy','mean_output_tokens','hit_cap_fraction','relative_length_reduction','passes_gate']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for s in summaries:writer.writerow({k:s[k] for k in fields})
    lines=[f'# {split} teacher dose-response results','',f'{n} paired questions; {len(summaries)} conditions. All effects below are exploratory pointwise intervals.','',
           '| Condition | Accuracy | Mean tokens | Length difference vs base [95% CI] | Actual norm | Gate |',
           '|---|---:|---:|---:|---:|---|']
    for s in summaries:
        d=s['length_minus_base']
        lines.append(f"| {s['condition']} | {s['accuracy']:.2%} | {s['mean_output_tokens']:.2f} | {d['estimate']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] | {s['diagnostics']['mean_delta_to_hidden_norm_fraction']:.2%} | {s['passes_gate']} |")
    lines+=['',json.dumps(decision,ensure_ascii=False,indent=2)]
    (output/'teacher_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decision),flush=True)


def plots(config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,_=paths(config)
    output=ROOT/config['figure_root']
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    conditions=config['sampling']['conditions']
    palette=['#777777','#2166AC','#B2182B']
    short_labels=['Token-uniform','Trace-balanced','Prefix64-balanced']
    sampling=read_json(root/'SAMPLING_COMPLETE.json')
    screens=[read_json(root/'feature_screen'/c/'SCREEN_COMPLETE.json') for c in conditions]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    exposure=[s['length_exposure_spearman'] for s in sampling['conditions']]
    axes[0].bar(short_labels,exposure,color=palette)
    axes[0].axhline(0,color='black',lw=.7)
    for i,v in enumerate(exposure):axes[0].text(i,v+.025,f'{v:.3f}',ha='center')
    axes[0].set(title='Length versus training exposure',ylabel='Spearman correlation',ylim=(-.12,1.08))
    supports=['full','prefix64','clean64'];x=np.arange(3)
    for i,s in enumerate(screens):
        y=[s['reconstruction_common_test'][key]['explained_variance'] for key in supports]
        axes[1].bar(x+(i-1)*.24,y,.24,label=short_labels[i],color=palette[i])
    axes[1].set(xticks=x,xticklabels=['Full','First 64','Clean positions'],ylim=(.55,.85),
                title='Same held-out positions',ylabel='Explained variance')
    axes[1].legend(frameon=False,fontsize=8)
    for i,s in enumerate(screens):
        for j,direction in enumerate(['short','long']):
            count=sum(f['heldout_confirmed'] for f in s['features'] if f['direction']==direction)
            axes[2].bar(i+(j-.5)*.32,count,.3,color=['#2166AC','#B2182B'][j],label=direction if i==0 else None)
    axes[2].set(xticks=x,xticklabels=short_labels,ylim=(0,18),title='Dev-ranked candidates: test confirmation',ylabel='Confirmed / 16 per direction')
    axes[2].legend(frameon=False)
    for a in [axes[0],axes[2]]:a.tick_params(axis='x',rotation=16)
    fig.suptitle('Recovered 881-question rank corpus: length-controlled SAE (seed 17)')
    fig.tight_layout()
    files=[]
    for suffix in ['png','pdf']:
        p=output/f'01_length_controlled_sae.{suffix}';fig.savefig(p,dpi=240,bbox_inches='tight');files.append(p)
    plt.close(fig)
    if not (root/'analysis/test/teacher_summary.json').is_file():return files
    report=read_json(root/'analysis/test/teacher_summary.json')
    summaries=report['summaries']
    groups=[('full_trace_balanced','short',4,'Full / short 4','#2166AC'),
            ('full_trace_balanced','short',16,'Full / short 16','#4393C3'),
            ('full_trace_balanced','joint',16,'Full / short-long 16','#B2182B'),
            ('full_trace_balanced','random',16,'Full / random 16','#777777'),
            ('prefix64_trace_balanced','short',16,'Prefix64 / short 16','#762A83')]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for dictionary,mode,count,label,color in groups:
        items=sorted([s for s in summaries if s['spec']['dictionary']==dictionary and s['spec']['mode']==mode
                      and s['spec']['count']==count and s['spec']['start']==0 and s['spec']['rho']>0],key=lambda s:s['spec']['rho'])
        xs=np.array([100*s['spec']['rho'] for s in items])
        ys=np.array([s['length_minus_base']['estimate'] for s in items])
        low=np.array([s['length_minus_base']['ci_low'] for s in items]);high=np.array([s['length_minus_base']['ci_high'] for s in items])
        axes[0,0].errorbar(xs,ys,yerr=[ys-low,high-ys],marker='o',label=label,color=color,capsize=3)
        axes[0,1].plot(xs,[100*s['accuracy_minus_base']['estimate'] for s in items],marker='o',label=label,color=color)
        axes[1,0].plot(xs,[100*s['diagnostics']['mean_delta_to_hidden_norm_fraction'] for s in items],marker='o',color=color)
        axes[1,1].plot(xs,[s['diagnostics']['short_code_change'] for s in items],marker='o',color=color,label=label)
    for ax in axes.flat:ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Requested relative norm (%)')
    axes[0,0].set(title='Paired output-length change',ylabel='Tokens versus no steering')
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,1].set(title='Teacher accuracy change',ylabel='Percentage points versus no steering')
    axes[1,0].plot([0,30],[0,30],'k--',lw=.8)
    axes[1,0].set(title='Actual delivered perturbation',ylabel='Mean actual norm (%)')
    axes[1,1].set(title='Target readback after intervention',ylabel='Change in summed short-feature activation')
    fig.suptitle('Held-out teacher intervention: 64 paired questions; exploratory pointwise intervals')
    fig.tight_layout()
    for suffix in ['png','pdf']:
        p=output/f'02_teacher_dose_response.{suffix}';fig.savefig(p,dpi=240,bbox_inches='tight');files.append(p)
    plt.close(fig)
    return files
