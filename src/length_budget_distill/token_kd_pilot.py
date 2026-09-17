"""Frozen, development-only small KD pilot with matched SFT controls."""
from pathlib import Path
import copy
import importlib.metadata
import json
import logging
import math
import os
import subprocess
import sys
import time

from .experiment_io import read_json, publish_files_hash_verified
from .records import read_jsonl, write_jsonl
from .factorial import canonical_sha256, file_sha256
from .ncsu_reproduction import save, seal, verify, isolated_gpu_preflight
from .completion_supervision import TrainingExposureAudit, validate_encoded_record

CODE=Path(__file__).resolve().parents[2]
ENTRY='13_63_run_token_kd_pilot.py'


def prepare(cfg):
    from transformers import AutoTokenizer
    root=Path(cfg['result_root']);source=Path(cfg['source_pilot_root'])
    verify(source/'sft/protocol/FROZEN.json');verify(source/'sft/protocol/SOURCES.json')
    verify(source/'inputs/COMPLETE.json') if (source/'inputs/COMPLETE.json').exists() else verify(source/'protocol/FROZEN.json')
    text=list(read_jsonl(source/'sft/text/B1.jsonl'))
    encoded=list(read_jsonl(source/'sft/encoded/qwen3b_student/B1.jsonl'))
    if [r['problem_id'] for r in text] != [r['problem_id'] for r in encoded]:raise ValueError('Source text/encoded mismatch')
    order=sorted(range(len(text)),key=lambda i:canonical_sha256([cfg['subset_seed'],text[i]['problem_id']]))[:cfg['train_questions']]
    text=[text[i] for i in order];encoded=[encoded[i] for i in order]
    dev=list(read_jsonl(source/'inputs/development.jsonl'))
    if len(dev)!=cfg['development_questions'] or any(r['question_role']!='development' for r in dev):raise ValueError('Changed development cohort')
    if set(r['problem_id'] for r in dev)&set(r['problem_id'] for r in text):raise ValueError('Train/development overlap')
    for row in encoded:validate_encoded_record(row,max_length=cfg['training']['max_length'])
    teacher=AutoTokenizer.from_pretrained(cfg['teacher']['snapshot_path'],local_files_only=True)
    student=AutoTokenizer.from_pretrained(cfg['students']['qwen3b_student']['snapshot_path'],local_files_only=True)
    vocab=student.get_vocab()
    if teacher.get_vocab()!=vocab or set(vocab.values())!=set(range(len(vocab))):raise ValueError('Teacher/student token-ID alignment is not exact and contiguous')
    if teacher.backend_tokenizer.to_str()!=student.backend_tokenizer.to_str():raise ValueError('Tokenizer normalizer/merges differ')
    write_jsonl(root/'sft/text/B1.jsonl',text);write_jsonl(root/'sft/encoded/qwen3b_student/B1.jsonl',encoded)
    write_jsonl(root/'inputs/development.jsonl',dev)
    save(root/'inputs/vocabulary.json',{'valid_vocabulary_size':len(vocab),'mapping_sha256':canonical_sha256(vocab),
        'projection':'Renormalize teacher and student over identical valid token IDs; exclude unnamed padded LM-head rows',
        'teacher_output_size':read_json(Path(cfg['teacher']['snapshot_path'])/'config.json')['vocab_size'],
        'student_output_size':read_json(Path(cfg['students']['qwen3b_student']['snapshot_path'])/'config.json')['vocab_size']})
    save(root/'inputs/data_audit.json',{'train_questions':len(text),'development_questions':len(dev),'train_unique':len(set(r['problem_id'] for r in text)),
         'train_ids':[r['problem_id'] for r in text],'maximum_training_tokens':max(len(r['input_ids']) for r in encoded),
         'previously_observed_development':True,'test_predictions_read':False,'source':'B1 shortest-correct, unchanged teacher completions'})
    save(root/'inputs/runtime_versions.json',read_json(source/'inputs/runtime_versions.json'))
    save(root/'inputs/model_hashes.json',read_json(source/'inputs/model_hashes.json'))
    seal(root/'inputs/COMPLETE.json',[root/'protocol/FROZEN.json',source/'sft/protocol/FROZEN.json',source/'inputs/development.jsonl',
         *sorted((root/'inputs').glob('*.json*')),*sorted((root/'sft/text').glob('*')),*sorted((root/'sft/encoded/qwen3b_student').glob('*'))],formal_claim_allowed=False)


def verify_inputs(cfg):
    root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json');verify(root/'inputs/COMPLETE.json')
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen KD source')
    for name,version in read_json(root/'inputs/runtime_versions.json').items():
        if importlib.metadata.version(name)!=version:raise ValueError('Pinned environment changed: '+name)


def train(cfg, arm, seed, *, smoke=False, method='B1'):
    import torch
    from transformers import set_seed
    from safetensors.torch import load_file
    from .training import run_trl_sft
    from .token_kd import install_kd_loss
    from .pilot_storage import validate_scratch
    from .baseline_reproduction import record_hardware
    verify_inputs(cfg)
    if arm not in ('sft','kd') or seed not in cfg['pilot_seeds']:raise ValueError('Unregistered pilot cell')
    root=Path(cfg['result_root']);key='smoke' if smoke else f'{arm}/seed_{seed}'
    if cfg.get('baseline_matrix'):key=f'{method}/{key}'
    out=root/'training'/key;out.mkdir(parents=True,exist_ok=False);record_hardware(out)
    rows=list(read_jsonl(root/'sft/encoded/qwen3b_student'/f'{method}.jsonl'))
    if smoke:
        rows=sorted(rows,key=lambda r:len(r['input_ids']),reverse=True)[:cfg['smoke_questions']]
    local=validate_scratch(cfg['runtime']['storage_policy'],os.environ)/'token_kd'/key
    path=root/'sft/encoded/qwen3b_student'/f'{method}.jsonl'
    if smoke:path=out/'smoke_encoded.jsonl';write_jsonl(path,rows)
    training={**cfg['training'],'seed':seed,'data_seed':seed,'output_dir':str(local),'save_strategy':'no',
              'model_init_kwargs':{'local_files_only':True,'attn_implementation':'sdpa'}}
    if smoke:training.update(max_steps=cfg['smoke_steps'],gradient_accumulation_steps=1)
    spec=cfg['students']['qwen3b_student']
    run={'student':{'model_name':spec['snapshot_path'],'torch_dtype':'bfloat16','use_lora':True,'lora':cfg['lora'],
                    'tokenizer_kwargs':{'local_files_only':True,'padding_side':'right'}},
         'training':training,'data':{'train_path':str(path),'text_format':'pretokenized_completion'}}
    kd_cfg=copy.deepcopy(cfg)
    if arm=='sft':kd_cfg['kd']['alpha']=0.
    for model in ('teacher','qwen3b_student'):
        if model=='teacher' and arm=='sft':continue
        for name,want in read_json(root/'inputs/model_hashes.json')[model].items():
            if file_sha256(name)!=want:raise ValueError('Model input hash changed: '+name)
    save(out/'run_config.json',{'run':run,'kd':kd_cfg['kd'],'arm':arm,'seed':seed,'smoke':smoke,'method':method})
    exposure=TrainingExposureAudit(rows,max_length=training['max_length']);meters={}
    def before_train(trainer):
        meters['kd']=install_kd_loss(trainer,kd_cfg,read_json(root/'inputs/vocabulary.json')['valid_vocabulary_size'])
        exposure.install(trainer)
    set_seed(seed);os.environ['LBD_RUNTIME_OUTPUT_DIR']=str(local)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    trainer=run_trl_sft(run,before_train=before_train)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    expected=cfg['smoke_steps'] if smoke else math.ceil(len(rows)/training['gradient_accumulation_steps'])*training['num_train_epochs']
    if trainer.state.global_step!=expected:raise ValueError('Incomplete optimizer steps')
    audit=exposure.summary()
    if not smoke and any(n!=training['num_train_epochs'] for n in audit['problem_exposures'].values()):raise ValueError('Unequal question exposure')
    weights=load_file(str(local/'adapter_model.safetensors'))
    if not all(torch.isfinite(v).all() for v in weights.values()) or not any(v.count_nonzero() for k,v in weights.items() if 'lora_B' in k):raise ValueError('Invalid adapter')
    metrics={**audit,'arm':arm,'seed':seed,'smoke':smoke,'optimizer_steps':trainer.state.global_step,
         'elapsed_seconds':seconds,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
         'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'kd':meters['kd'],'log_history':trainer.state.log_history,
         'temporary_output_dir':str(local),'training_complete':True,'evaluation_complete':False}
    save(out/'metrics.json',metrics)
    adapter=Path(cfg['checkpoint_root'])/key
    publish_files_hash_verified(local,adapter,('adapter_model.safetensors','adapter_config.json'))
    seal(adapter/'TRAIN_COMPLETE.json',[root/'inputs/COMPLETE.json',root/'protocol/SOURCES.json',out/'run_config.json',out/'metrics.json',out/'hardware.json',*sorted(adapter.glob('*'))])
    seal(out/'COMPLETE.json',[adapter/'TRAIN_COMPLETE.json',out/'metrics.json'],smoke=smoke,formal_claim_allowed=False)


def evaluate(cfg,arm,seed,*,method='B1',cohort='development',base=False):
    import torch
    from .student_evaluation import load_student_for_evaluation
    from .unified_student_evaluation import generate_student_batch,audit_student_prediction
    from .baseline_reproduction import record_hardware
    verify_inputs(cfg);root=Path(cfg['result_root']);key=f'{arm}/seed_{seed}'
    if cfg.get('baseline_matrix'):key=f'{method}/{key}'
    if base:key='base'
    adapter=Path(cfg['checkpoint_root'])/key
    if not base:verify(adapter/'TRAIN_COMPLETE.json')
    out=root/cohort/key;out.mkdir(parents=True,exist_ok=False);record_hardware(out)
    spec=cfg['students']['qwen3b_student']
    bundle=load_student_for_evaluation({**spec,'model_name':spec['snapshot_path'],'torch_dtype':'bfloat16','attn_implementation':'sdpa'},adapter_path=None if base else str(adapter))
    questions=list(read_jsonl(root/'inputs'/f'{cohort}.jsonl'));rows=[];start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (out/'predictions.jsonl').open('x') as f:
        for begin in range(0,len(questions),cfg['batch_size']):
            batch=questions[begin:begin+cfg['batch_size']]
            ratio=cfg.get('tokenskip_ratio',1.0) if method=='B6' and not base else None
            values,seconds=generate_student_batch(bundle,batch,cfg,ratio=ratio,max_new_tokens=cfg['evaluation_cap'])
            for row,source in zip(values,batch):
                audit_student_prediction(row,source,cfg,bundle['tokenizer'])
                row.update(arm='base' if base else arm,seed=None if base else seed,method='base' if base else method,cohort=cohort);rows.append(row);f.write(json.dumps(row)+'\n')
            f.flush();logging.info('KD pilot %s seed %s development %s/%s',arm,seed,len(rows),len(questions))
    if len(rows)!=len(questions) or len({r['problem_id'] for r in rows})!=len(rows):raise ValueError('Incomplete development predictions')
    summary={'arm':'base' if base else arm,'seed':None if base else seed,'method':'base' if base else method,'cohort':cohort,'questions':len(rows),'accuracy':sum(r['grade']['is_correct'] for r in rows)/len(rows),
        'mean_output_tokens':sum(r['output_tokens'] for r in rows)/len(rows),
        'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rows)/len(rows),'elapsed_seconds':time.monotonic()-start,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
    save(out/'summary.json',summary)
    parents=[] if base else [adapter/'TRAIN_COMPLETE.json']
    seal(out/'COMPLETE.json',[root/'inputs/COMPLETE.json',*parents,*sorted(out.glob('*'))],formal_claim_allowed=False)


def analyze(cfg):
    import numpy as np
    import matplotlib.pyplot as plt
    root=Path(cfg['result_root']);out=root/'analysis';out.mkdir(exist_ok=False)
    metrics=[];differences=[];markers=[];audits=[]
    ids=[r['problem_id'] for r in read_jsonl(root/'inputs/development.jsonl')]
    for seed in cfg['pilot_seeds']:
        predictions={}
        for arm in ('sft','kd'):
            p=root/'development'/arm/f'seed_{seed}';verify(p/'COMPLETE.json');markers.append(p/'COMPLETE.json')
            rows=list(read_jsonl(p/'predictions.jsonl'));assert [r['problem_id'] for r in rows]==ids
            predictions[arm]=np.array([r['grade']['is_correct'] for r in rows],dtype=float)
            metrics.append(read_json(p/'summary.json'))
            a=read_json(root/'training'/arm/f'seed_{seed}/metrics.json');audits.append(a)
        if audits[-1]['problem_exposures']!=audits[-2]['problem_exposures'] or audits[-1]['actual_supervision_tokens_after_causal_shift']!=audits[-2]['actual_supervision_tokens_after_causal_shift']:
            raise ValueError('Matched SFT/KD training exposure differs')
        differences.append(predictions['kd']-predictions['sft'])
    delta=np.stack(differences);rng=np.random.default_rng(cfg['bootstrap_seed']);samples=[]
    # Resample questions with paired predictions across the two fixed trained seeds.
    for _ in range(cfg['bootstrap_samples']):samples.append(delta[:,rng.integers(0,len(ids),len(ids))].mean())
    low,high=map(float,np.quantile(samples,[.025,.975]));byseed=delta.mean(axis=1)
    lengths_ok=all(next(m for m in metrics if m['seed']==s and m['arm']=='kd')['mean_output_tokens'] <= cfg['gate']['maximum_length_ratio']*next(m for m in metrics if m['seed']==s and m['arm']=='sft')['mean_output_tokens'] for s in cfg['pilot_seeds'])
    caps_ok=all(m['cap_hit_rate']<=cfg['gate']['maximum_cap_hit_rate'] for m in metrics)
    passed=bool((byseed>0).all() and delta.mean()>=cfg['gate']['minimum_mean_accuracy_gain'] and low>0 and lengths_ok and caps_ok)
    decision={'scale_up_eligible':passed,'mean_accuracy_delta':float(delta.mean()),'per_seed_deltas':dict(zip(map(str,cfg['pilot_seeds']),map(float,byseed))),
        'paired_question_bootstrap_95_ci':[low,high],'length_gate_passed':lengths_ok,'cap_gate_passed':caps_ok,
        'gate':cfg['gate'],'claims':'Exploratory, previously observed GSM8K development; interval conditional on two trained seeds; no locked-test or SAE efficacy claim.',
        'next_stage':'Freeze a 1024-question matched B1 replication before any SAE/method expansion' if passed else 'Do not expand; preserve negative or inconclusive pilot without post-hoc tuning'}
    save(out/'metrics.json',{'rows':metrics});save(out/'decision.json',decision)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    x=np.arange(len(cfg['pilot_seeds']))
    for arm,shift,color in [('sft',-.18,'#4c72b0'),('kd',.18,'#c44e52')]:
        selected=[next(m for m in metrics if m['arm']==arm and m['seed']==s) for s in cfg['pilot_seeds']]
        axes[0].bar(x+shift,[100*m['accuracy'] for m in selected],.36,label=arm.upper(),color=color)
        axes[1].bar(x+shift,[m['mean_output_tokens'] for m in selected],.36,label=arm.upper(),color=color)
    for ax in axes:ax.set_xticks(x,[str(s) for s in cfg['pilot_seeds']]);ax.set_xlabel('Student seed');ax.legend()
    axes[0].set_ylabel('Development accuracy (%)');axes[1].set_ylabel('Mean output tokens')
    fig.tight_layout();fig.savefig(out/'sft_vs_kd.png',dpi=180);fig.savefig(out/'sft_vs_kd.pdf');plt.close(fig)
    (out/'report.md').write_text('# Small token-KD pilot\n\n'+json.dumps(decision,indent=2)+'\n')
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',*markers,*sorted(out.glob('*'))],formal_claim_allowed=False)


def submit(cfg,key,stage,*,parents=(),gpu=False,arm='kd',seed=17):
    root=Path(cfg['result_root']);out=root/'launch'/key;out.mkdir(parents=True,exist_ok=False)
    route=cfg['runtime']['routes']['h200' if gpu else 'cpu']
    cmd=['sbatch','--parsable','--account='+route['account'],'--partition='+route['partition'],'--qos='+route['qos'],
         '--nodes=1','--ntasks=1','--cpus-per-task='+str(route['cpus']),'--mem='+route['memory'],'--time='+route['time'],
         '--job-name=kd_pilot_'+key,'--output='+str(out/'%j.log')]
    if gpu:cmd+=['--gres='+route['gres']]
    if parents:cmd+=['--dependency=afterok:'+':'.join(map(str,parents))]
    cmd += [str(Path(cfg['code_root'])/'scripts/slurm/13_6_run_frozen_python.sh'),cfg['code_root'],str(root/'protocol/frozen_config.json'),
        cfg['runtime']['python'],cfg['runtime']['overlay'],ENTRY,'--stage',stage,'--arm',arm,'--seed',str(seed)]
    save(out/'intent.json',{'command':cmd,'stage':stage,'arm':arm,'seed':seed,'parents':list(parents)})
    reply=subprocess.run(cmd,check=True,text=True,capture_output=True);job=int(reply.stdout.strip().split(';')[0])
    save(out/'submission.json',{'command':cmd,'job_id':job,'stage':stage,'arm':arm,'seed':seed,'parents':list(parents)})
    return job


def dispatch(args):
    cfg=read_json(args.config);root=Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json');verify(root/'protocol/SOURCES.json')
    from .pilot_storage import configure
    stage={'train':'train','cell':'train','smoke':'train','evaluate':'evaluate-dev'}.get(args.stage,args.stage)
    # Smoke uses the same input-size estimate after preparation.
    configure(cfg,stage,'B1')
    if args.stage=='test':
        import unittest
        suite=unittest.TestSuite()
        for pattern in ('test_token_kd.py','test_completion_supervision.py','test_pilot_storage.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(CODE/'tests'),pattern=pattern))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():raise RuntimeError('KD tests failed')
        seal(root/'protocol/TEST_COMPLETE.json',[root/'protocol/FROZEN.json'],tests_run=result.testsRun)
    elif args.stage=='prepare':verify(root/'protocol/TEST_COMPLETE.json');prepare(cfg)
    elif args.stage in ('train','evaluate','smoke','cell'):
        verify_inputs(cfg)
        pre=root/'preflight'/str(os.environ['SLURM_JOB_ID'])/(args.stage+'_'+str(os.getpid()))
        isolated_gpu_preflight(args.config,pre,expected_name='H200')
        if args.stage=='cell':
            for stage in ('train','evaluate'):
                subprocess.run([sys.executable,str(CODE/'scripts'/ENTRY),'--config',args.config,'--stage',stage,'--arm',args.arm,'--seed',str(args.seed)],check=True)
        elif args.stage=='evaluate':evaluate(cfg,args.arm,args.seed)
        else:train(cfg,args.arm,args.seed,smoke=args.stage=='smoke')
    elif args.stage=='launch-pilot':
        verify(root/'training/smoke/COMPLETE.json');m=read_json(root/'training/smoke/metrics.json')
        estimated=m['log_history'][-1]['train_runtime']/cfg['smoke_steps']*math.ceil(cfg['train_questions']/cfg['training']['gradient_accumulation_steps'])*cfg['training']['num_train_epochs']*cfg['training']['gradient_accumulation_steps']
        if estimated > cfg['maximum_projected_training_seconds']:raise RuntimeError('Smoke does not fit pilot time budget')
        if m['peak_gpu_reserved_mib']+cfg['memory_safety_margin_mib']>cfg['runtime']['minimum_free_mib']:raise RuntimeError('Measured peak exceeds registered memory admission')
        jobs=[]
        for seed in cfg['pilot_seeds']:
            for arm in ('sft','kd'):jobs.append(submit(cfg,f'{arm}_{seed}','cell',gpu=True,arm=arm,seed=seed))
        analysis=submit(cfg,'analysis','analyze',parents=jobs)
        save(root/'protocol/pilot_jobs.json',{'workers':jobs,'analysis':analysis,'smoke_peak_reserved_mib':m['peak_gpu_reserved_mib'],
            'l40s_considered':True,'route_reason':'H200 has available quota and tested runtime; no L40S throughput measurement yet','scale_up_submitted':False})
    elif args.stage=='analyze':analyze(cfg)
    else:raise ValueError(args.stage)
