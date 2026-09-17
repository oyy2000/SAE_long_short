"""Measure held-out response energies and math-context drift of fitted ASC.

No vectors are retrained and no new answers are generated. All source responses
and adverse conditions are retained; this is an objective diagnostic, not proof
of a successful reproduction of the published compression effect.
"""
from pathlib import Path
import importlib.metadata
import json
import logging
import os
import re
import subprocess

from .experiment_io import read_json
from .records import read_jsonl,write_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save,seal,verify,admission,teacher_bundle
from .baseline_method_analysis import audit_cohort
from .gsm8k_grading_v3 import grade_gsm8k_response

CODE=Path(__file__).resolve().parents[2]


def run(config_path,teacher):
    import torch
    from safetensors.torch import load_file
    from .asc_ces import ResidualAddition,response_inputs,response_energy,contrastive_energy,forward_kl
    cfg=read_json(config_path);parent=Path(cfg['parents'][teacher]);out=Path(cfg['result_root'])/teacher
    if CODE!=Path(cfg['code_root']):raise ValueError('Use frozen diagnostic snapshot')
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    bindings=[]
    for relative in ('protocol/FROZEN.json','protocol/SOURCES.json','asc/COMPLETE.json','asc_generation/COMPLETE.json'):
        path=parent/relative;verify(path);bindings.append(path)
    parent_cfg=read_json(parent/'protocol/frozen_config.json')
    rows=list(read_jsonl(parent/'inputs/sources.jsonl'))
    base_rows=[r for r in read_jsonl(parent/'asc_generation/predictions.jsonl') if r['scale']==0.]
    base=audit_cohort(base_rows,[r['problem_id'] for r in rows])
    questions=sorted(rows,key=lambda r:canonical_sha256([cfg['cohort_seed'],r['problem_id']]))[:cfg['questions']]
    if len(questions)!=cfg['questions']:raise ValueError('Insufficient held-out questions')
    # The selected dev cohort must not be among the 100 optimization pairs.
    pairs=parent/('asc_calibration/pairs.jsonl' if 'calibration_generation' in parent_cfg else 'inputs/pairs.jsonl')
    training_ids={r['problem_id'] for r in read_jsonl(pairs)}
    if training_ids & {r['problem_id'] for r in questions}:raise ValueError('Energy check leaks fit questions')
    bindings.append(pairs);out.mkdir(parents=True,exist_ok=False)
    admission(cfg)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv'],capture_output=True,text=True,check=True)
    save(out/'hardware.json',{'job_id':os.environ['SLURM_JOB_ID'],'csv':hardware.stdout,
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))})
    model,tok=teacher_bundle(parent_cfg)
    model.requires_grad_(False);torch.cuda.reset_peak_memory_stats()
    vector=load_file(str(parent/'asc/vector.safetensors'))['vector'].to(model.device)
    layer=model.model.layers[parent_cfg['asc']['layer_index']]
    measurements=[];selected=[]
    with torch.no_grad(),(out/'measurements.jsonl').open('x') as handle:
        for row in questions:
            pid=row['problem_id'];long=base[pid]['solution']
            short=re.sub(r'<<[^>]*>>','',row['raw_answer']).replace('####','Answer:')
            input_short,mask_short=response_inputs(tok,row['prompt'],short,device=model.device,max_length=cfg['max_sequence_length'])
            input_long,mask_long=response_inputs(tok,row['prompt'],long,device=model.device,max_length=cfg['max_sequence_length'])
            selected.append({'problem_id':pid,'prompt':row['prompt'],'concise':short,'verbose':long,'answer':row['answer'],
                'verbose_grade':grade_gsm8k_response(long,row['answer']),
                'source_uncapped':not base[pid]['hit_max_new_tokens'],
                'concise_tokens_with_eos':int(mask_short.sum()),'verbose_tokens_with_eos':int(mask_long.sum())})
            captured=[]
            def capture(module,inputs,output):
                hidden=output[0] if isinstance(output,tuple) else output
                captured.append(hidden[mask_long].detach().clone())
            hook=layer.register_forward_hook(capture)
            try:base_logits=model(**input_long,use_cache=False).logits
            finally:hook.remove()
            if len(captured)!=1:raise ValueError('Unexpected layer invocation count')
            states=captured[0]
            long0=float(response_energy(base_logits,input_long['input_ids'],mask_long))
            short_logits=model(**input_short,use_cache=False).logits
            short0=float(response_energy(short_logits,input_short['input_ids'],mask_short));del short_logits
            for scale in cfg['scales']:
                controller=ResidualAddition(layer,scale*vector)
                with controller.applied(mask_short):
                    logits=model(**input_short,use_cache=False).logits
                short_nll=float(response_energy(logits,input_short['input_ids'],mask_short));del logits
                with controller.applied(mask_long):
                    logits=model(**input_long,use_cache=False).logits
                long_nll=float(response_energy(logits,input_long['input_ids'],mask_long))
                kl=float(forward_kl(base_logits,logits,mask_long));del logits
                actual=(states+(scale*vector).to(states)).float()-states.float()
                relative=actual.norm(dim=-1)/states.float().norm(dim=-1).clamp_min(1e-12)
                result={'problem_id':pid,'teacher':teacher,'scale':scale,
                    'base_concise_nll':short0,'base_verbose_nll':long0,
                    'concise_nll':short_nll,'verbose_nll':long_nll,
                    'concise_nll_change':short_nll-short0,'verbose_nll_change':long_nll-long0,
                    'ces':float(contrastive_energy(torch.tensor(short_nll),torch.tensor(long_nll))),
                    'math_response_kl_base_to_steered':kl,
                    'mean_actual_delta_over_hidden_norm':float(relative.mean()),
                    'max_actual_delta_over_hidden_norm':float(relative.max()),
                    'base_response_correct':selected[-1]['verbose_grade']['is_correct']}
                if not all(torch.isfinite(torch.tensor(result[k])) for k in ('concise_nll','verbose_nll','ces','math_response_kl_base_to_steered')):
                    raise ValueError('Nonfinite objective diagnostic')
                measurements.append(result);handle.write(json.dumps(result)+'\n');handle.flush()
            del base_logits,states,captured
            logging.info('ASC diagnostic %s %s complete',teacher,pid)
    if len(measurements)!=cfg['questions']*len(cfg['scales']):raise ValueError('Incomplete diagnostic matrix')
    for scale in cfg['scales']:audit_cohort([r for r in measurements if r['scale']==scale],[r['problem_id'] for r in questions])
    means={str(scale):{field:sum(r[field] for r in measurements if r['scale']==scale)/len(questions)
        for field in ('concise_nll_change','verbose_nll_change','ces','math_response_kl_base_to_steered','mean_actual_delta_over_hidden_norm')}
        for scale in cfg['scales']}
    fit=read_json(parent/'asc/summary.json')
    summary={'status':'complete','teacher':teacher,'questions':len(questions),'by_scale':means,
        'registered_fit_wikitext_holdout_kl':fit['heldout_generic_kl_mean'],
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'question_ids':[r['problem_id'] for r in questions],
        'claim_boundary':cfg['claim_boundary'],'formal_claim_allowed':False,
        'math_kl_context':'Teacher-forced response positions of the complete unmodified sampled response; differs from the generic WikiText context distribution.'}
    save(out/'summary.json',summary);write_jsonl(out/'selected_pairs.jsonl',selected)
    save(out/'versions.json',{n:importlib.metadata.version(n) for n in ('torch','transformers','math-verify')})
    report=['# ASC held-out objective diagnostic','',f'Model: {teacher}; {len(questions)} fixed development questions disjoint from the 100 fitting pairs.','',
        '| Scale | Concise NLL change | Verbose NLL change | CES | Math response KL | Delta / hidden norm |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for scale,m in means.items():report.append(f"| {scale} | {m['concise_nll_change']:.4f} | {m['verbose_nll_change']:.4f} | {m['ces']:.4f} | {m['math_response_kl_base_to_steered']:.4f} | {m['mean_actual_delta_over_hidden_norm']:.4f} |")
    report+=['',f"Original held-out WikiText KL: {fit['heldout_generic_kl_mean']:.6f}.",
        'A positive NLL change means the trajectory becomes less likely. Low CES can arise from raising verbose NLL even when concise NLL also rises.',
        'The paired inputs, all scales and correctness flags are retained. No correctness filtering or generation is performed in this stage. Prose-number extraction remains subject to the documented v3 limitation.',
        '',cfg['claim_boundary'],'']
    (out/'report.md').write_text('\n'.join(report))
    seal(out/'COMPLETE.json',bindings+[Path(config_path),Path(cfg['launch_root'])/'FROZEN.json']+
         [p for p in out.rglob('*') if p.is_file()],stage='asc_heldout_objective_diagnostic',formal_claim_allowed=False)
