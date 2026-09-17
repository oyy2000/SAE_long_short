"""Matched baseline-by-objective KD expansion, reusing the audited pilot trainer."""
from pathlib import Path
import json
import math
import os
import subprocess
import sys

from .experiment_io import read_json
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify, isolated_gpu_preflight
from .completion_supervision import validate_encoded_record
from . import token_kd_pilot as pilot

CODE = Path(__file__).resolve().parents[2]
ENTRY = '13_72_run_token_kd_baselines.py'


def paired_ids(rows, expected=None):
    ids = [r['problem_id'] for r in rows]
    if len(set(ids)) != len(ids) or (expected is not None and ids != expected):
        raise ValueError('Duplicate, missing, or differently ordered paired examples')
    return ids


def publish_prepared_rows(path, rows, *, reuse=False):
    if reuse:
        if list(read_jsonl(path)) != rows:
            raise ValueError('Prepared B1 data differs from the matched source')
    else:
        write_jsonl(path,rows)


def prepare(cfg):
    # Shared B1 preparation establishes vocabulary identity and model/environment hashes.
    pilot.prepare(cfg)
    root = Path(cfg['result_root']); source = Path(cfg['source_pilot_root'])
    initial_marker = root/'inputs/COMPLETE.json'
    initial_marker.rename(root/'inputs/B1_PREPARED.json')
    ids = paired_ids(list(read_jsonl(root/'sft/text/B1.jsonl')))
    if len(ids) != cfg['train_questions']:
        raise ValueError('Incomplete registered training cohort')
    audits = {}; bindings = [source/'selection/COMPLETE.json']
    verify(bindings[0])
    selected = read_json(source/'selection/operating_points.json')
    if selected['selected_conditions'] != cfg['baseline_matrix'] or selected['tokenskip_ratio'] != cfg['tokenskip_ratio']:
        raise ValueError('Previously frozen baseline operating points changed')
    for method, condition in cfg['baseline_matrix'].items():
        paths = [source/'sft/text'/f'{condition}.jsonl', source/'sft/encoded/qwen3b_student'/f'{condition}.jsonl']
        text, encoded = [list(read_jsonl(p)) for p in paths]
        paired_ids(encoded, paired_ids(text))
        text_by_id = {r['problem_id']:r for r in text}; encoded_by_id = {r['problem_id']:r for r in encoded}
        if set(text_by_id) != set(ids):
            raise ValueError('Baseline does not have identical common support: '+method)
        text = [text_by_id[i] for i in ids]; encoded = [encoded_by_id[i] for i in ids]
        for row in encoded:validate_encoded_record(row,max_length=cfg['training']['max_length'])
        publish_prepared_rows(root/'sft/text'/f'{method}.jsonl',text,reuse=method=='B1')
        publish_prepared_rows(root/'sft/encoded/qwen3b_student'/f'{method}.jsonl',encoded,reuse=method=='B1')
        audits[method] = {'condition':condition,'questions':len(encoded),
            'maximum_input_tokens':max(len(r['input_ids']) for r in encoded),
            'target_tokens_per_epoch':sum(sum(t != -100 for t in r['labels'][1:]) for r in encoded),
            'source_hashes':{str(p):file_sha256(p) for p in paths},
            'training_order_sha256':canonical_sha256(ids),
            'sae_dictionary':'historical_answer_associated' if method == 'B7' else None}
        bindings.extend(paths)
    evaluation = list(read_jsonl(source/'inputs/evaluation.jsonl'))
    eval_ids = paired_ids(evaluation)
    dev_ids = paired_ids(list(read_jsonl(root/'inputs/development.jsonl')))
    if len(evaluation) != cfg['evaluation_questions'] or set(eval_ids)&(set(ids)|set(dev_ids)):
        raise ValueError('Incorrect or overlapping registered evaluation cohort')
    write_jsonl(root/'inputs/evaluation.jsonl',evaluation)
    save(root/'inputs/baseline_audit.json',{'methods':audits,'previously_observed_evaluation':True,
        'equal_examples':True,'equal_question_order':True,'equal_target_tokens_between_objectives':True,
        'equal_target_tokens_across_methods':False,'baseline_trace_generation_reused':True,
        'teacher_logits':'same unmodified frozen teacher on each method-specific completion prefix'})
    seal(root/'inputs/COMPLETE.json',[root/'protocol/FROZEN.json',*bindings,
        source/'inputs/evaluation.jsonl',*sorted((root/'inputs').glob('*.json*')),
        *sorted((root/'sft/text').glob('*')),*sorted((root/'sft/encoded/qwen3b_student').glob('*'))],formal_claim_allowed=False)


def submit(cfg,key,stage,*,parents=(),gpu=False,method='B1',arm='kd',seed=17,cohort='development'):
    root=Path(cfg['result_root']);out=root/'launch'/key;out.mkdir(parents=True,exist_ok=False)
    route_name=cfg['runtime'].get('training_route','h200') if gpu else 'cpu'
    route=cfg['runtime']['routes'][route_name]
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],
        '--qos='+route['qos'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),
        '--mem='+route['memory'],'--time='+route['time'],'--job-name=kd_full_'+key,'--output='+str(out/'%j.log')]
    if gpu:cmd.append('--gres='+route['gres'])
    if parents:cmd.extend(['--dependency=afterok:'+':'.join(map(str,parents)),'--kill-on-invalid-dep=yes'])
    cmd.extend([str(CODE/'scripts/slurm/13_6_run_frozen_python.sh'),str(CODE),str(root/'protocol/frozen_config.json'),
        cfg['runtime']['python'],cfg['runtime']['overlay'],ENTRY,'--stage',stage,'--method',method,
        '--arm',arm,'--seed',str(seed),'--cohort',cohort])
    save(out/'intent.json',{'command':cmd,'parents':list(parents)})
    reply=subprocess.run(cmd,text=True,capture_output=True,check=True)
    job=int(reply.stdout.strip().split(';')[0]);save(out/'submission.json',{'job_id':job,'command':cmd})
    # Bind the actual Slurm-spooled script, not just the working-tree launcher.
    spool=out/'submitted_batch.sh'
    subprocess.run(['scontrol','write','batch_script',str(job),str(spool)],check=True,capture_output=True,text=True)
    launcher=CODE/'scripts/slurm/13_6_run_frozen_python.sh'
    if spool.read_bytes()!=launcher.read_bytes():raise ValueError('Slurm batch differs from frozen launcher')
    seal(out/'SUBMISSION_VERIFIED.json',[out/'intent.json',out/'submission.json',spool,launcher])
    return job


def launch_matrix(cfg):
    root=Path(cfg['result_root']);method=cfg['smoke_method']
    verify(root/'training'/method/'smoke/COMPLETE.json')
    m=read_json(root/'training'/method/'smoke/metrics.json')
    estimated=m['log_history'][-1]['train_runtime']/cfg['smoke_steps']*cfg['train_questions']*cfg['training']['num_train_epochs']
    if estimated>cfg['maximum_projected_training_seconds']:
        raise RuntimeError('Longest-input smoke exceeds training time budget')
    if m['peak_gpu_reserved_mib']+cfg['memory_safety_margin_mib']>cfg['runtime']['minimum_free_mib']:
        raise RuntimeError('Measured memory peak exceeds registered admission')
    lanes=[None]*cfg['runtime']['maximum_gpu_lanes'];jobs=[]
    for method in cfg['baseline_matrix']:
        for seed in cfg['pilot_seeds']:
            for arm in ('sft','kd'):
                lane=len(jobs)%len(lanes);parents=[] if lanes[lane] is None else [lanes[lane]]
                job=submit(cfg,f'{method}_{arm}_{seed}','cell',parents=parents,gpu=True,method=method,arm=arm,seed=seed)
                lanes[lane]=job;jobs.append(job)
    base=submit(cfg,'base','base',parents=[lanes[0]],gpu=True)
    analysis=submit(cfg,'analysis','analyze',parents=[*lanes,base])
    save(root/'protocol/matrix_jobs.json',{'workers':jobs,'base':base,'analysis':analysis,
        'max_gpu_lanes':len(lanes),'training_cells':len(jobs),'smoke_peak_reserved_mib':m['peak_gpu_reserved_mib'],
        'projected_training_seconds':estimated,'user_authorized_expansion_despite_inconclusive_pilot':True,
        'math_training_submitted':False})


def analyze(cfg):
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    root=Path(cfg['result_root']);out=root/'analysis';out.mkdir(exist_ok=False)
    markers=[];summaries=[];comparisons=[];methods=list(cfg['baseline_matrix'])
    # Pair questions across every seed/method; CIs are conditional on the three trained seeds.
    for cohort in cfg['evaluation_cohorts']:
        ids=paired_ids(list(read_jsonl(root/'inputs'/f'{cohort}.jsonl')))
        base_path=root/cohort/'base';verify(base_path/'COMPLETE.json');markers.append(base_path/'COMPLETE.json')
        base_rows=list(read_jsonl(base_path/'predictions.jsonl'));paired_ids(base_rows,ids)
        summaries.append({'cohort':cohort,'method':'base','arm':'base','seed':None,**read_json(base_path/'summary.json')})
        values={}
        for method in methods:
            for arm in ('sft','kd'):
                values[method,arm]=[]
                for seed in cfg['pilot_seeds']:
                    path=root/cohort/method/arm/f'seed_{seed}'
                    verify(path/'COMPLETE.json');markers.append(path/'COMPLETE.json')
                    rows=list(read_jsonl(path/'predictions.jsonl'));paired_ids(rows,ids)
                    values[method,arm].append([int(r['grade']['is_correct']) for r in rows])
                    summaries.append({**read_json(path/'summary.json'),'cohort':cohort,'method':method})
            for seed in cfg['pilot_seeds']:
                audits=[read_json(root/'training'/method/arm/f'seed_{seed}'/'metrics.json') for arm in ('sft','kd')]
                for key in ('problem_exposures','actual_supervision_tokens_after_causal_shift','optimizer_steps'):
                    if audits[0][key]!=audits[1][key]:raise ValueError('Unmatched objective training exposure')
        contrasts=[('kd_vs_sft',method,(method,'kd'),(method,'sft')) for method in methods]
        contrasts += [('kd_vs_B1_kd',method,(method,'kd'),('B1','kd')) for method in methods if method!='B1']
        rng=np.random.default_rng(cfg['bootstrap_seed']);cohort_comparisons=[]
        for family,method,a,b in contrasts:
            delta=np.asarray(values[a],float)-np.asarray(values[b],float)
            per_question=delta.mean(axis=0);samples=[]
            for _ in range(cfg['bootstrap_samples']):
                samples.append(per_question[rng.integers(0,len(ids),len(ids))].mean())
            se=per_question.std(ddof=1)/math.sqrt(len(ids))
            p=float(2*norm.sf(abs(per_question.mean()/se))) if se else (1.0 if per_question.mean()==0 else 0.0)
            cohort_comparisons.append({'cohort':cohort,'family':family,'method':method,
                'accuracy_delta':float(delta.mean()),'per_seed_delta':delta.mean(axis=1).tolist(),
                'paired_question_bootstrap_95_ci':np.quantile(samples,[.025,.975]).tolist(),
                'paired_question_wald_p':p})
        for family in ('kd_vs_sft','kd_vs_B1_kd'):
            selected=sorted([r for r in cohort_comparisons if r['family']==family],key=lambda r:r['paired_question_wald_p'])
            adjusted=0.
            for rank,row in enumerate(selected):
                adjusted=max(adjusted,min(1.,(len(selected)-rank)*row['paired_question_wald_p']))
                row['holm_adjusted_p']=adjusted
        comparisons.extend(cohort_comparisons)
    save(out/'metrics.json',{'rows':summaries})
    save(out/'paired_comparisons.json',{'rows':comparisons,'interval_scope':'Conditional on three trained seeds; previously observed cohorts; no independent confirmation'})
    costs=[]
    for method in methods:
        for seed in cfg['pilot_seeds']:
            for arm in ('sft','kd'):
                m=read_json(root/'training'/method/arm/f'seed_{seed}'/'metrics.json')
                costs.append({'method':method,'arm':arm,'seed':seed,**{k:m[k] for k in ('elapsed_seconds','peak_gpu_reserved_mib','optimizer_steps','actual_supervision_tokens_after_causal_shift')}})
    save(out/'training_costs.json',{'rows':costs})
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(len(methods))
    for arm,shift,color in [('sft',-.18,'#4c72b0'),('kd',.18,'#c44e52')]:
        primary_cohort=cfg.get('primary_plot_cohort','evaluation')
        groups=[[r for r in summaries if r['cohort']==primary_cohort and r['method']==method and r['arm']==arm] for method in methods]
        for ax,key,scale in [(axes[0],'accuracy',100),(axes[1],'mean_output_tokens',1)]:
            ax.bar(x+shift,[scale*np.mean([r[key] for r in group]) for group in groups],.36,label=arm.upper(),color=color)
    for ax in axes:ax.set_xticks(x,methods);ax.legend()
    axes[0].set_ylabel('Accuracy (%)');axes[1].set_ylabel('Mean output tokens')
    fig.tight_layout();fig.savefig(out/'baseline_comparison.png',dpi=180);fig.savefig(out/'baseline_comparison.pdf');plt.close(fig)
    (out/'report.md').write_text('# Matched KD/SFT baseline expansion\n\n48 training cells; paired seeds 17/42/73. See metrics, paired comparisons, and costs. B7 uses the historical Answer-associated dictionary. Both GSM8K evaluation cohorts were previously observed.\n')
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',root/'inputs/COMPLETE.json',*markers,*sorted(out.glob('*'))],formal_claim_allowed=False)


def dispatch(args):
    cfg=read_json(args.config);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen baseline source')
    from .pilot_storage import configure
    stage={'cell':'train','smoke':'train','train':'train','evaluate':'evaluate-dev','base':'evaluate-test'}.get(args.stage,args.stage)
    configure(cfg,stage,args.method)
    if args.stage=='test':
        import unittest
        suite=unittest.TestSuite()
        for pattern in ('test_token_kd.py','test_completion_supervision.py','test_pilot_storage.py','test_token_kd_baselines.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(CODE/'tests'),pattern=pattern))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():raise RuntimeError('KD baseline tests failed')
        seal(root/'protocol/TEST_COMPLETE.json',[root/'protocol/FROZEN.json'],tests_run=result.testsRun)
    elif args.stage=='prepare':verify(root/'protocol/TEST_COMPLETE.json');prepare(cfg)
    elif args.stage=='launch-matrix':launch_matrix(cfg)
    elif args.stage=='analyze':analyze(cfg)
    elif args.stage in ('cell','base'):
        stages=[('train',None)] if args.stage=='cell' else []
        stages += [('evaluate',cohort) for cohort in cfg['evaluation_cohorts']]
        for stage,cohort in stages:
            cmd=[sys.executable,str(CODE/'scripts'/ENTRY),'--config',args.config,'--stage',stage,
                 '--method',args.method,'--arm',args.arm,'--seed',str(args.seed)]
            if cohort:cmd.extend(['--cohort',cohort])
            if args.stage=='base':cmd.append('--base')
            subprocess.run(cmd,check=True)
    else:
        pilot.verify_inputs(cfg)
        pre=root/'preflight'/str(os.environ['SLURM_JOB_ID'])/(args.stage+'_'+str(os.getpid()))
        isolated_gpu_preflight(args.config,pre,expected_name=cfg['runtime'].get('expected_gpu_name','H200'))
        if args.stage=='evaluate':pilot.evaluate(cfg,args.arm,args.seed,method=args.method,cohort=args.cohort,base=args.base)
        elif args.stage in ('train','smoke'):pilot.train(cfg,args.arm,args.seed,method=args.method,smoke=args.stage=='smoke')
        else:raise ValueError(args.stage)
