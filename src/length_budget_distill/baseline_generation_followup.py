"""Prespecified lower-dose ASC generation and separate R1 prompt diagnostics."""
from pathlib import Path
import json
import logging
import os
import subprocess

from .experiment_io import read_json
from .records import read_jsonl
from .factorial import canonical_sha256
from .ncsu_reproduction import save,seal,verify,admission,teacher_bundle
from .baseline_method_analysis import audit_cohort
from .baseline_reproduction import generate_text,summarize_generations
from .gsm8k_grading_v3 import grade_gsm8k_response

CODE=Path(__file__).resolve().parents[2]


def run(config_path,teacher):
    import torch
    from safetensors.torch import load_file
    from .asc_ces import ResidualAddition
    cfg=read_json(config_path);parent=Path(cfg['parents'][teacher]);out=Path(cfg['result_root'])/teacher
    if CODE!=Path(cfg['code_root']):raise ValueError('Use the frozen follow-up snapshot')
    verify(Path(cfg['launch_root'])/'FROZEN.json')
    bindings=[]
    for relative in ('protocol/FROZEN.json','protocol/SOURCES.json','asc/COMPLETE.json','asc_generation/COMPLETE.json'):
        marker=parent/relative;verify(marker);bindings.append(marker)
    diagnostic=Path(cfg['objective_diagnostic_root'])/teacher/'COMPLETE.json'
    verify(diagnostic);bindings.append(diagnostic)
    pcfg=read_json(parent/'protocol/frozen_config.json')
    sources=list(read_jsonl(parent/'inputs/sources.jsonl'))
    if len(sources)!=cfg['expected_questions']:raise ValueError('Unexpected development cohort')
    prompt_ids={r['problem_id'] for r in sorted(sources,key=lambda r:canonical_sha256([cfg['prompt_seed'],r['problem_id']]))[:cfg['prompt_questions']]}
    out.mkdir(parents=True,exist_ok=False);admission(cfg)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv'],capture_output=True,text=True,check=True)
    save(out/'hardware.json',{'job_id':os.environ['SLURM_JOB_ID'],'csv':hardware.stdout,
        'torch_device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))})
    model,tok=teacher_bundle(pcfg);torch.cuda.reset_peak_memory_stats()
    vector=load_file(str(parent/'asc/vector.safetensors'))['vector'].to(model.device)
    settings=pcfg['validation'];rows=[]
    with (out/'predictions.jsonl').open('x') as handle:
        for source in sources:
            # Match the existing three-scale sweep's per-question seed exactly.
            seed=int(canonical_sha256([pcfg['seed'],source['problem_id']])[:8],16)
            cells=[{'name':f'project_scale_{s:g}','scale':s,'prompt':source['prompt'],'rendering':'native_chat','kind':'lower_scale'} for s in cfg['new_scales']]
            if teacher=='r1' and source['problem_id'] in prompt_ids:
                cells += [{'name':name,'scale':0.,'prompt':source['question']+spec['suffix'],
                           'rendering':spec['rendering'],'kind':'prompt_probe'} for name,spec in cfg['prompt_probes'].items()]
            for cell in cells:
                controller=ResidualAddition(model.model.layers[pcfg['asc']['layer_index']],cell['scale']*vector) if cell['scale'] else None
                result=generate_text(model,tok,[{'role':'user','content':cell['prompt']}],
                                     {**settings,'prompt_rendering':cell['rendering']},seed,controller)
                grade=grade_gsm8k_response(result['solution'],source['answer'])
                row={'problem_id':source['problem_id'],'question':source['question'],'gold_answer':source['answer'],
                     'teacher':teacher,'condition':cell,**result,**grade,'grading_status':'provisional_pending_uniform_prose_answer_audit'}
                rows.append(row);handle.write(json.dumps(row)+'\n');handle.flush()
                logging.info('ASC follow-up %s %s %s tokens=%d correct=%s',teacher,source['problem_id'],cell['name'],row['generated_tokens'],row['is_correct'])
    expected_cells={f'project_scale_{s:g}':{r['problem_id'] for r in sources} for s in cfg['new_scales']}
    if teacher=='r1':expected_cells.update({name:prompt_ids for name in cfg['prompt_probes']})
    if len(rows)!=sum(map(len,expected_cells.values())):raise ValueError('Unexpected row count')
    summaries={}
    for name,ids in expected_cells.items():
        subset=[r for r in rows if r['condition']['name']==name];audit_cohort(subset,ids)
        measured=summarize_generations(subset)
        measured['accuracy_before_prose_review']=measured.pop('accuracy');summaries[name]=measured
    save(out/'summary.json',{'status':'generation_complete','teacher':teacher,'by_condition':summaries,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'accuracy_audit_complete':False,'formal_claim_allowed':False,'claim_boundary':cfg['claim_boundary']})
    seal(out/'COMPLETE.json',bindings+[Path(config_path),Path(cfg['launch_root'])/'FROZEN.json']+
         [p for p in out.rglob('*') if p.is_file()],stage='asc_lower_dose_and_prompt_generation',formal_claim_allowed=False)
