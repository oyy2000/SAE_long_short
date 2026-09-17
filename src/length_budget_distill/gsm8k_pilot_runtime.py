"""Dependency-ordered execution and student-based selection for the GSM8K pilot."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import time

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .ncsu_reproduction import save, seal, verify, isolated_gpu_preflight
from .factorial import canonical_sha256
from . import gsm8k_student_pilot as data

CODE = Path(__file__).resolve().parents[2]
ENTRY = '13_60_run_gsm8k_student_pilot.py'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def execution_config(main):
    return main.get('execution_config_path',str(Path(main['result_root'])/'protocol/frozen_config.json'))


def submit(main, key, stage, *, route='cpu', parents=(), round_number=1, group='', shard=0, method='', seed=17):
    """One recorded intent per job; uncertain submissions are never retried."""
    root=Path(main['result_root']); spec=main['runtime']['routes'][route]
    out=root/'launch'/key; out.mkdir(parents=True,exist_ok=False)
    cmd=['sbatch','--parsable','--account='+spec['account'],'--partition='+spec['partition'],'--qos='+spec['qos'],
         '--nodes=1','--ntasks=1','--cpus-per-task='+str(spec['cpus']),'--mem='+spec['memory'],'--time='+spec['time'],
         '--job-name=gsm_pilot_'+key,'--output='+str(out/'%j.log')]
    if parents: cmd+=['--dependency=afterok:'+':'.join(str(j) for j in parents)]
    if spec.get('gres'): cmd+=['--gres='+spec['gres'],'--export=ALL,LBD_PILOT_GPU_ROUTE='+route]
    cmd += [str(Path(main['code_root'])/'scripts/slurm/13_6_run_frozen_python.sh'),main['code_root'],
            execution_config(main),main['runtime']['python'],main['runtime']['overlay'],main.get('entrypoint',ENTRY),
            '--stage',stage,'--round',str(round_number),'--group',group,'--shard',str(shard),'--method',method,'--seed',str(seed)]
    intent={'key':key,'stage':stage,'route':route,'parents':list(parents),'command':cmd,'created_utc':timestamp()}
    save(out/'intent.json',intent)
    response=subprocess.run(cmd,check=True,text=True,capture_output=True)
    job=int(response.stdout.strip().split(';')[0])
    save(out/'submission.json',{**intent,'job_id':job,'stdout':response.stdout,'stderr':response.stderr,'experiment_complete':False})
    logging.info('Submitted %s job=%s parents=%s',key,job,parents)
    return job


def submit_lanes(main, specifications, *, parents=(), prefix):
    lanes=[None]*main['runtime']['maximum_gpu_lanes']; jobs=[]
    for i,spec in enumerate(specifications):
        lane=i%len(lanes); dependencies=[*parents]+([lanes[lane]] if lanes[lane] else [])
        if spec.get('route')!='cpu' and main['runtime'].get('worker_routes'):
            spec['route']=main['runtime']['worker_routes'][lane % len(main['runtime']['worker_routes'])]
        job=submit(main,prefix+'_'+spec.pop('key'),parents=dependencies,**spec)
        lanes[lane]=job;jobs.append(job)
    return jobs


def source_chain(main, group, number):
    data.candidate_config(main,group,number)
    prefix=f'r{number}_{group}'; route=main['runtime']['generation_route']
    smoke=submit(main,prefix+'_smoke','generate',route=route,group=group,round_number=number,method='smoke')
    smoke_merge=submit(main,prefix+'_smoke_merge','merge-candidates',parents=[smoke],group=group,round_number=number,method='smoke')
    jobs=submit_lanes(main,[{'key':f'{i:02d}','stage':'generate','route':route,'group':group,'round_number':number,
                            'shard':i,'method':'student_pool'} for i in range(main['shards'])],parents=[smoke_merge],prefix=prefix)
    merged=submit(main,prefix+'_merge','merge-candidates',parents=jobs,group=group,round_number=number,method='student_pool')
    submit(main,prefix+'_next','advance',parents=[merged],group='steered' if group=='raw' else 'text',round_number=number)


def advance(main, phase, number):
    root=Path(main['result_root'])
    if phase=='calibration':
        data.prepare_calibration(main)
        smoke=submit(main,'calibration_generate','calibrate',route=main['runtime']['asc_route'])
        fit_smoke=submit(main,'asc_smoke','asc-smoke',route=main['runtime']['asc_route'],parents=[smoke])
        fit=submit(main,'asc_fit','asc-fit',route=main['runtime']['asc_route'],parents=[fit_smoke])
        prep=submit(main,'directions_prepare','prepare-directions',parents=[fit])
        directions=submit(main,'directions_extract','directions',route=main['runtime']['generation_route'],parents=[prep])
        submit(main,'r1_begin','advance',group='raw',parents=[directions])
    elif phase in ('raw','steered'):
        source_chain(main,phase,number)
    elif phase=='text':
        data.prepare_compression(main,number); prefix=f'r{number}_text'
        ds=submit(main,prefix+'_dap_smoke','compress',route=main['runtime']['generation_route'],group='dap',method='smoke',round_number=number)
        ts=submit(main,prefix+'_skip_smoke','compress',route='l40s',group='tokenskip',method='smoke',round_number=number)
        sm=submit(main,prefix+'_smoke_merge','merge-text',parents=[ds,ts],method='smoke',round_number=number)
        jobs=submit_lanes(main,[{'key':f'{m}_{i:02d}','stage':'compress','group':m,'shard':i,'method':'student_pool',
                                'route':main['runtime']['generation_route'] if m=='dap' else 'l40s','round_number':number}
                               for m in ('dap','tokenskip') for i in range(main['shards'])],parents=[sm],prefix=prefix)
        merge=submit(main,prefix+'_merge','merge-text',parents=jobs,method='student_pool',round_number=number)
        submit(main,prefix+'_support','advance',group='support',parents=[merge],round_number=number)
    elif phase=='support':
        if not data.prepare_sft(main,list(range(1,number+1))):
            if number!=1: raise ValueError('Only one source extension is registered')
            source_chain(main,'raw',2);return
        cells=[('base',17),*[(m,17) for m in data.condition_ids(main)]]
        jobs=submit_lanes(main,[{'key':f'{m}_{s}','stage':'train-dev','route':main['runtime']['training_route'],
                               'method':m,'seed':s} for m,s in cells],prefix='initial')
        submit(main,'select_operating_points','select',parents=jobs)
    elif phase=='confirmation':
        choice=read_json(root/'selection/operating_points.json'); verify(root/'selection/COMPLETE.json')
        if choice['repeat_gate_passed']:
            cells=[(m,s) for m in choice['repeat_conditions'] for s in (42,73)]
            jobs=submit_lanes(main,[{'key':f'{m}_{s}','stage':'train-dev','route':main['runtime']['training_route'],
                                   'method':m,'seed':s} for m,s in cells],prefix='repeat')
            submit(main,'freeze_evaluation','freeze-evaluation',parents=jobs)
        else:
            submit(main,'freeze_evaluation','freeze-evaluation')
    else: raise ValueError('Unknown pilot advance phase')


def development_directory(main, method, seed):
    return Path(main['result_root'])/'development'/method/f'seed_{seed}'


def check_cell(main, method, seed):
    if 'registered_student_cells' in main:
        if [method,seed] not in main['registered_student_cells']:
            raise ValueError('Unregistered aligned-recipe student cell')
        return
    if method=='base' and seed==17:return
    if method not in data.condition_ids(main): raise ValueError('Unknown SFT condition')
    if seed==17:return
    verify(Path(main['result_root'])/'selection/COMPLETE.json')
    selected=read_json(Path(main['result_root'])/'selection/operating_points.json')
    if seed not in (42,73) or not selected['repeat_gate_passed'] or method not in selected['repeat_conditions']:
        raise ValueError('Unregistered conditional repeat')


def train_and_dev(main, method, seed):
    check_cell(main,method,seed)
    stages=['evaluate-dev'] if method=='base' else ['train','evaluate-dev']
    for stage in stages:
        subprocess.run([sys.executable,str(CODE/'scripts'/ENTRY),'--config',execution_config(main),
                        '--stage',stage,'--method',method,'--seed',str(seed)],check=True)


def evaluation_bundle(main, method, seed):
    from .student_evaluation import load_student_for_evaluation
    from .factorial import file_sha256
    check_cell(main,method,seed); root=Path(main['result_root']); adapter=None; bindings=[]
    for p,digest in read_json(root/'inputs/model_hashes.json')['qwen3b_student'].items():
        if file_sha256(p)!=digest: raise ValueError('Student weights or tokenizer changed')
    if method!='base':
        adapter=Path(main['checkpoint_root'])/'qwen3b_student'/method/f'seed_{seed}'
        verify(adapter/'TRAIN_COMPLETE.json'); bindings.append(adapter/'TRAIN_COMPLETE.json')
    model={**main['students']['qwen3b_student'],'model_name':main['students']['qwen3b_student']['snapshot_path'],
           'torch_dtype':'bfloat16','attn_implementation':'sdpa'}
    return load_student_for_evaluation(model,adapter_path=str(adapter) if adapter else None),bindings


def evaluate(main, method, seed, *, development, shard=0):
    import torch
    from .baseline_reproduction import record_hardware
    from .unified_student_evaluation import generate_student_batch,audit_student_prediction
    from .student_cap_validation import prefix_prediction,summarize_cap_grid
    root=Path(main['result_root']); ratios=main['tokenskip_ratios'] if method=='B6' else [None]
    if development:
        out=development_directory(main,method,seed); questions=list(read_jsonl(root/'inputs/development.jsonl')); cap=max(main['caps']); bindings=[]
    else:
        marker=root/'evaluation_protocol/COMPLETE.json'; verify(marker)
        protocol=read_json(root/'evaluation_protocol/protocol.json')
        if [method,seed] not in protocol['cells']: raise ValueError('Unregistered test model')
        if not 0<=shard<main['evaluation_shards']: raise ValueError('Invalid test shard')
        out=root/'evaluation'/method/f'seed_{seed}'/f'shard_{shard:02d}'
        questions=list(read_jsonl(root/'inputs/evaluation.jsonl'))[shard::main['evaluation_shards']]
        cap=protocol['max_new_tokens']; ratios=[protocol['tokenskip_ratio']] if method=='B6' else [None]; bindings=[marker]
    out.mkdir(parents=True,exist_ok=False); record_hardware(out)
    bundle,adapter_bindings=evaluation_bundle(main,method,seed);bindings.extend(adapter_bindings)
    torch.cuda.reset_peak_memory_stats(); start=time.monotonic(); full=[]; views=[]; timings=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for ratio in ratios:
            for offset in range(0,len(questions),main['batch_size']):
                batch=questions[offset:offset+main['batch_size']]
                rows,seconds=generate_student_batch(bundle,batch,main,ratio=ratio,max_new_tokens=cap)
                bid=f'{ratio}_{offset}'
                timings.append({'batch_id':bid,'questions':len(batch),'generation_wall_seconds':seconds})
                for row,source in zip(rows,batch):
                    audit_student_prediction(row,source,main,bundle['tokenizer'])
                    row={**row,'method':method,'seed':seed,'batch_id':bid,'amortized_generation_seconds':seconds/len(batch)}
                    full.append(row);handle.write(json.dumps(row)+'\n')
                    if development:
                        views.extend(prefix_prediction(row,source,c,main,bundle['tokenizer']) for c in main['caps'])
                handle.flush();logging.info('Student %s seed=%s dev=%s ratio=%s records=%s',method,seed,development,ratio,len(full))
    if len(full)!=len(questions)*len(ratios) or len({(r['problem_id'],r['ratio']) for r in full})!=len(full):
        raise ValueError('Incomplete student evaluation grid')
    write_jsonl(out/'batches.jsonl',timings)
    summary={'questions':len(questions),'records':len(full),'elapsed_seconds':time.monotonic()-start,
             'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'method':method,'seed':seed,'development':development}
    if development:
        write_jsonl(out/'cap_predictions.jsonl',views)
        summary.update(summarize_cap_grid(views,questions,ratios,main))
    save(out/'summary.json',summary)
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*bindings,*sorted(out.glob('*'))],formal_claim_allowed=False)


def initial_cells(main):
    return [('base',17),*[(m,17) for m in data.condition_ids(main)]]


def common_cap(main, cells):
    admissible=set(main['caps']); summaries={}; markers=[]
    for method,seed in cells:
        out=development_directory(main,method,seed); verify(out/'COMPLETE.json');markers.append(out/'COMPLETE.json')
        summary=read_json(out/'summary.json');admissible &= set(summary['admissible_caps']);summaries[(method,seed)]=summary
    if not admissible: raise ValueError('No common admissible output budget; preserve outputs and freeze no test protocol')
    return min(admissible),summaries,markers


def select_points(main, metrics, lengths):
    """Pure development-only decision, with a strict shortening constraint for SAE."""
    expected=set(data.condition_ids(main))|{'base'}
    if set(metrics)!=expected or set(lengths)!=expected-{'base'}: raise ValueError('Incomplete operating-point inputs')
    def key(m):return (-metrics[m]['accuracy'],lengths[m],float(m.split('__')[1]) if '__' in m else 0,m)
    selected={m:m for m in ('B0','B1','B2','B5','B6')}
    for family in ('B3','B4','B7'):
        candidates=[m for m in metrics if m.startswith(family+'__')]
        if family=='B7':candidates=[m for m in candidates if lengths[m]<lengths['B0']]
        selected[family]=min(candidates,key=key) if candidates else None
    champion=min(('B2','B3','B4','B5','B6'),key=lambda m:(-metrics[selected[m]]['accuracy'],m))
    sae=selected['B7']; gain=main['gate_minimum_accuracy_gain']; competitor=selected[champion]
    gate=bool(sae and metrics[sae]['accuracy']>metrics['B0']['accuracy'] and metrics[sae]['accuracy']>metrics['base']['accuracy']
              and all(metrics[sae]['accuracy']-metrics[b]['accuracy']>=gain-1e-12 for b in ('B1',competitor)))
    return {'selected_conditions':selected,'champion_baseline':champion,'champion_condition':competitor,
            'repeat_gate_passed':gate,'repeat_conditions':['B0','B1',competitor,sae] if gate else [],
            'sae_shortening_feasible':sae is not None,'selection_uses_test_outputs':False}


def select(main):
    root=Path(main['result_root']); cap,summaries,markers=common_cap(main,initial_cells(main))
    audit=read_json(root/'sft/inputs/token_audit.json');lengths={m:v['mean_teacher_tokens'] for m,v in audit.items()}
    metrics={}; chosen_ratio=None
    for (method,seed),summary in summaries.items():
        scores=summary['by_ratio_and_cap']
        ratio=min(scores,key=lambda r:(-scores[r][str(cap)]['accuracy'],scores[r][str(cap)]['mean_output_tokens'],r))
        metrics[method]={**scores[ratio][str(cap)],'ratio':None if ratio=='None' else float(ratio)}
        if method=='B6':chosen_ratio=float(ratio)
    choice=select_points(main,metrics,lengths)
    choice.update(development_cap=cap,tokenskip_ratio=chosen_ratio,metrics=metrics,teacher_lengths=lengths)
    out=root/'selection';save(out/'operating_points.json',choice)
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',root/'sft/protocol/FROZEN.json',*markers,out/'operating_points.json'])
    advance(main,'confirmation',1)


def freeze_evaluation(main):
    root=Path(main['result_root']);verify(root/'selection/COMPLETE.json')
    selected=read_json(root/'selection/operating_points.json');cells=initial_cells(main)
    if selected['repeat_gate_passed']:cells += [(m,s) for m in selected['repeat_conditions'] for s in (42,73)]
    cap,_,markers=common_cap(main,cells)
    # A larger shared cap after repeats does not reselect doses or ratios.
    out=root/'evaluation_protocol';save(out/'protocol.json',{'cells':cells,'max_new_tokens':cap,
         'tokenskip_ratio':selected['tokenskip_ratio'],'locked_source_indices':list(range(50,1319)),
         'operating_points_remain_frozen':True,'test_questions':1269,'formal_claim_allowed':False})
    seal(out/'COMPLETE.json',[root/'selection/COMPLETE.json',*markers,out/'protocol.json'])
    jobs=submit_lanes(main,[{'key':f'{m}_{s}_{i:02d}','stage':'evaluate-test','route':main['runtime']['evaluation_route'],
                           'method':m,'seed':s,'shard':i} for m,s in cells for i in range(main['evaluation_shards'])],prefix='test')
    submit(main,'analyze','analyze',parents=jobs)


def analyze(main):
    import numpy as np
    from scipy.stats import binomtest
    from .factorial_analysis import holm_adjust
    from .utility_analysis import paired_question_bootstrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def interval(left,right):
        result=paired_question_bootstrap(left,right,samples=main['bootstrap_samples'],seed=main['bootstrap_seed'])
        # Bootstrap tail fractions are not exact hypothesis-test p values.
        result.pop('bootstrap_p_value',None)
        return result
    root=Path(main['result_root']);verify(root/'evaluation_protocol/COMPLETE.json')
    protocol=read_json(root/'evaluation_protocol/protocol.json');choice=read_json(root/'selection/operating_points.json')
    questions=list(read_jsonl(root/'inputs/evaluation.jsonl'));ids={q['problem_id'] for q in questions}
    metrics=[];predictions={};markers=[root/'evaluation_protocol/COMPLETE.json'];all_rows=[]
    for method,seed in protocol['cells']:
        rows=[]
        for i in range(main['evaluation_shards']):
            part=root/'evaluation'/method/f'seed_{seed}'/f'shard_{i:02d}';verify(part/'COMPLETE.json');markers.append(part/'COMPLETE.json')
            values=list(read_jsonl(part/'predictions.jsonl'));expected={q['problem_id'] for q in questions[i::main['evaluation_shards']]}
            if len(values)!=len(expected) or {r['problem_id'] for r in values}!=expected: raise ValueError('Missing or duplicate test predictions')
            if any(r['method']!=method or r['seed']!=seed or r['max_new_tokens']!=protocol['max_new_tokens'] for r in values):
                raise ValueError('Changed test method/seed/budget')
            rows.extend(values)
        predictions[(method,seed)]={r['problem_id']:int(r['grade']['is_correct']) for r in rows};all_rows.extend(rows)
        metrics.append({'condition':method,'seed':seed,'questions':len(rows),'accuracy':statistics.mean(r['grade']['is_correct'] for r in rows),
                        'mean_output_tokens':statistics.mean(r['output_tokens'] for r in rows),'cap_hit_rate':statistics.mean(r['hit_max_new_tokens'] for r in rows)})
    comparisons=[];sae=choice['selected_conditions']['B7']
    if sae:
        left=predictions[(sae,17)]
        for family in ('B0','B1','B2','B3','B4','B5','B6'):
            method=choice['selected_conditions'][family];right=predictions[(method,17)]
            wins=sum(left[k]>right[k] for k in ids);losses=sum(left[k]<right[k] for k in ids)
            comparisons.append({'baseline':family,'condition':method,'seed':17,
                **interval(left,right),
                'sae_only_correct':wins,'baseline_only_correct':losses,
                'p_value':float(binomtest(wins,wins+losses,p=0.5).pvalue) if wins+losses else 1.0})
        for row,p in zip(comparisons,holm_adjust([r['p_value'] for r in comparisons])):row['holm_p_value']=p
    out=root/'analysis';write_jsonl(out/'metrics.jsonl',metrics);write_jsonl(out/'comparisons.jsonl',comparisons)
    write_jsonl(out/'all_predictions.jsonl',all_rows)
    repeat_differences=[]
    if choice['repeat_gate_passed']:
        for seed in main['student_seeds']:
            for baseline in ['base','B0','B1',choice['champion_condition']]:
                right=predictions[(baseline,17 if baseline=='base' else seed)]
                left=predictions[(sae,seed)]
                repeat_differences.append({'seed':seed,'baseline':baseline,
                    **interval(left,right)})
    save(out/'repeat_differences.json',{'rows':repeat_differences,'uncertainty':'Question-paired intervals conditional on each trained adapter; seeds are reported separately.'})
    seed17={r['condition']:r for r in metrics if r['seed']==17}
    labels=['base',*[choice['selected_conditions'][m] for m in ('B0','B1','B2','B3','B4','B5','B6','B7') if choice['selected_conditions'][m]]]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(labels,[100*seed17[m]['accuracy'] for m in labels],color=['#c44e52' if m==sae else '#4c72b0' for m in labels])
    ax.set_ylabel('GSM8K student accuracy (%)');ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(out/'student_accuracy.png',dpi=180);fig.savefig(out/'student_accuracy.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for m in labels[1:]:
        x=choice['teacher_lengths'][m];y=100*seed17[m]['accuracy'];ax.scatter(x,y,color='#c44e52' if m==sae else '#4c72b0');ax.annotate(m,(x,y),fontsize=8)
    ax.set_xlabel('Mean teacher supervision tokens');ax.set_ylabel('GSM8K student accuracy (%)');fig.tight_layout();fig.savefig(out/'teacher_length_student_accuracy.png',dpi=180);fig.savefig(out/'teacher_length_student_accuracy.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,family in zip(axes,('B3','B4','B7')):
        grid=main['strengths'][family];ax.plot(grid,[100*choice['metrics'][f'{family}__{s:g}']['accuracy'] for s in grid],marker='o');ax.set_title(family);ax.set_xlabel('Steering strength')
    axes[0].set_ylabel('GSM8K development accuracy (%)');fig.tight_layout();fig.savefig(out/'strength_student_accuracy.png',dpi=180);fig.savefig(out/'strength_student_accuracy.pdf');plt.close(fig)
    summary={'conditions':len(protocol['cells']),'trained_adapters':len(protocol['cells'])-1,'test_questions':1269,
        'selected_conditions':choice['selected_conditions'],'repeat_gate_passed':choice['repeat_gate_passed'],
        'sae_shortening_feasible':choice['sae_shortening_feasible'],'claim_boundary':main['claim_boundary'],
        'formal_claim_allowed':False,'no_cross_dataset_claim':True}
    save(out/'summary.json',summary)
    report=['# GSM8K SAE student pilot','',main['claim_boundary'],'',f"Audited {len(protocol['cells'])} model cells on 1,269 locked questions.",
            f"SAE shortening feasible: {choice['sae_shortening_feasible']}. Repeat gate: {choice['repeat_gate_passed']}.",'',
            '| Condition | Seed | Accuracy | Mean output tokens |','| --- | ---: | ---: | ---: |']
    report += [f"| {r['condition']} | {r['seed']} | {r['accuracy']:.4f} | {r['mean_output_tokens']:.2f} |" for r in metrics]
    (out/'report.md').write_text('\n'.join(report)+'\n')
    seal(out/'COMPLETE.json',[*markers,*sorted(out.glob('*'))],formal_claim_allowed=False)
    seal(root/'EXPERIMENT_COMPLETE.json',[root/'protocol/FROZEN.json',root/'selection/COMPLETE.json',out/'COMPLETE.json'],formal_claim_allowed=False)


def dispatch(args):
    if args.stage=='test':
        import unittest
        suite=unittest.TestSuite()
        for pattern in ('test_native_runtime.py','test_answer_marker_diagnostic.py','test_pilot_recovery.py','test_pilot_storage.py','test_gsm8k_student_pilot.py','test_gsm8k_grading_v3.py','test_completion_supervision.py',
                        'test_unified_math_candidates.py','test_unified_student_evaluation.py','test_compression_baselines.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(CODE/'tests'),pattern=pattern))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): raise RuntimeError('Pilot regression tests failed')
        seal(Path(args.config).parent/'TEST_COMPLETE.json',[Path(args.config),*sorted((CODE/'tests').glob('test_gsm8k_student_pilot.py'))],
             tests_run=result.testsRun,job_id=os.environ.get('SLURM_JOB_ID'))
        return
    if args.stage=='prepare':
        data.prepare(args.config);return
    main=read_json(args.config);root=Path(main['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if main.get('execution_marker'):
        verify(main['execution_marker'])
        if main.get('execution_tests_marker'):verify(main['execution_tests_marker'])
    if CODE!=Path(main['code_root']): raise ValueError('Execute the immutable pilot source snapshot')
    gpu_stages={'calibrate','asc-smoke','asc-fit','directions','generate','compress','train-dev','train','evaluate-dev','evaluate-test'}
    if args.stage in gpu_stages:
        out=root/'preflight'/str(os.environ['SLURM_JOB_ID'])/(args.stage+'_'+str(os.getpid()));out.mkdir(parents=True,exist_ok=False)
        if main['runtime'].get('storage_policy'):
            if not os.environ.get('LBD_PILOT_STORAGE_EVIDENCE'):raise RuntimeError('Missing workload storage admission')
        elif shutil.disk_usage(os.environ['TMPDIR']).free<main['runtime']['minimum_free_scratch_bytes']:
            raise RuntimeError('Insufficient job-local scratch')
        if args.stage in ('calibrate','asc-smoke','asc-fit'):route=main['runtime']['asc_route']
        elif args.stage in ('train','train-dev'):route=main['runtime']['training_route']
        elif args.stage in ('evaluate-dev','evaluate-test'):route=main['runtime']['evaluation_route']
        elif args.stage=='compress' and args.group=='tokenskip':route='l40s'
        else:route=main['runtime']['generation_route']
        route=os.environ.get('LBD_PILOT_GPU_ROUTE',route)
        if route not in ('l40s','h200','h100'):raise ValueError('Unregistered GPU route')
        isolated_gpu_preflight(args.config,out,expected_name={'l40s':'L40S','h200':'H200','h100':'H100'}[route])
    if args.stage=='recover':
        from .pilot_recovery import run
        run(args.config,args.shard)
    elif args.stage=='advance': advance(main,args.group,args.round)
    elif args.stage in ('calibrate','asc-smoke','asc-fit'):
        from .baseline_reproduction import generate_calibration_pairs,train_asc
        cfg=data.load_child(main,'calibration')
        if args.stage=='calibrate':generate_calibration_pairs(cfg)
        else:train_asc(cfg,smoke=args.stage=='asc-smoke')
    elif args.stage=='prepare-directions':data.prepare_directions(main)
    elif args.stage=='directions':
        from .math_steering_directions import extract
        extract(main.get('directions_execution_config',root/'directions/protocol/frozen_config.json'))
    elif args.stage in ('generate','merge-candidates'):
        from .unified_math_candidates import generate,merge
        cfg=data.load_child(main,f'round_{args.round}/{args.group}')
        if args.stage=='generate':generate(cfg,args.method,args.shard)
        else:
            merge(cfg,args.method)
            if args.method=='smoke':
                smoke=read_json(Path(cfg['result_root'])/'generation/smoke/shard_00/summary.json')
                count=sum(1 for _ in read_jsonl(Path(cfg['result_root'])/'inputs/student_pool.jsonl'))
                expected=smoke['generation_batch_seconds']*max(1,((count+cfg['shards']-1)//cfg['shards'])/smoke['questions'])*1.5+300
                route=main['runtime']['routes'][main['runtime']['generation_route']]
                hours,minutes,seconds=map(int,route['time'].split(':'))
                if expected>hours*3600+minutes*60+seconds:
                    raise RuntimeError(f'Smoke throughput does not fit registered shard walltime: projected {expected:.0f}s')
    elif args.stage in ('compress','merge-text'):
        from .unified_text_compression import run_dap,run_tokenskip,merge
        cfg=data.load_child(main,f'round_{args.round}/text')
        if args.stage=='merge-text':merge(cfg,args.method)
        elif args.group=='dap':run_dap(cfg,args.method,args.shard)
        elif args.group=='tokenskip':run_tokenskip(cfg,args.method,args.shard)
        else:raise ValueError('Unknown compression method')
    elif args.stage=='train-dev':train_and_dev(main,args.method,args.seed)
    elif args.stage=='train':
        from .unified_student_distillation import train
        check_cell(main,args.method,args.seed)
        cfg=data.load_child(main,'sft')
        adapter=Path(cfg['checkpoint_root'])/'qwen3b_student'/args.method/f'seed_{args.seed}'
        training=Path(cfg['result_root'])/'training/qwen3b_student'/args.method/f'seed_{args.seed}'
        if (adapter/'TRAIN_COMPLETE.json').exists() and (training/'COMPLETE.json').exists():
            verify(adapter/'TRAIN_COMPLETE.json');verify(training/'COMPLETE.json')
        else:train(root/'sft/protocol/frozen_config.json','qwen3b_student',args.method,args.seed)
    elif args.stage in ('evaluate-dev','evaluate-test'):evaluate(main,args.method,args.seed,development=args.stage=='evaluate-dev',shard=args.shard)
    elif args.stage=='select':select(main)
    elif args.stage=='freeze-evaluation':freeze_evaluation(main)
    elif args.stage=='analyze':analyze(main)
    else:raise ValueError('Unknown pilot stage')
