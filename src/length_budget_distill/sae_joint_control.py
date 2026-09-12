"""Paired structural random control for the fixed dev-selected joint intervention."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .experiment_io import read_json,write_json_exclusive
from .sae_local_data import ROOT,paths,jsonl,evidence,verify
from .sae_norm_intervention import NormMatchedController,generate_condition
from .utility_analysis import paired_question_bootstrap


class JointRandomController(NormMatchedController):
    def __init__(self, *, positive_ids, negative_ids, **kwargs):
        super().__init__(**kwargs)
        self.positive_ids=positive_ids
        self.negative_ids=negative_ids

    def begin(self,spec,batch_size):
        super().begin(spec,batch_size)
        if spec.get('structural_random'):
            positive=self.decoder[self.positive_ids].float().sum(0)
            negative=self.decoder[self.negative_ids].float().sum(0)
            delta=positive/positive.norm()-negative/negative.norm()
            self.direction=delta/delta.norm()


def freeze(config,addon,addon_path):
    root,_=paths(config)
    output=root/'matched_joint_control'
    output.mkdir(exist_ok=False)
    protocol=read_json(root/'protocol/generation_protocol.json')
    selection=read_json(root/'DEV_SELECTION.json')
    if addon['selected_condition']!=selection['selected_condition']:raise ValueError('Selected condition changed')
    selected=next(s for s in protocol['specs'] if s['name']==selection['selected_condition'])
    if selected['mode']!='joint':raise ValueError('This addendum is for a selected joint intervention')
    screen=read_json(protocol['sources'][selected['dictionary']]['path'])
    target_ids=[f['feature_id'] for f in screen['features']]
    pool=sorted(set(range(28672))-set(target_ids))
    controls=np.random.default_rng(addon['random_seed']).choice(pool,2*selected['count'],replace=False).tolist()
    tasks=[]
    for split in ['dev','test']:
        ids=protocol['cohorts'][split]
        for original_shard in range(config['generation']['shards']):
            shard_ids=[pid for i,pid in enumerate(ids) if i%config['generation']['shards']==original_shard]
            for start in range(0,len(shard_ids),config['generation']['batch_size']):
                tasks.append({'split':split,'problem_ids':shard_ids[start:start+config['generation']['batch_size']]})
    specs=[dict(selected,name='no_steering_replay',rho=0.),dict(selected,name='selected_target_replay'),
           dict(selected,name='matched_joint_random',structural_random=True,total_feature_count=2*selected['count'])]
    write_json_exclusive(output/'protocol.json',{'addon':addon,'config':evidence(addon_path),
        'parent_generation_protocol':evidence(root/'protocol/generation_protocol.json'),
        'selection':evidence(root/'DEV_SELECTION.json'),'checkpoint':screen['checkpoint'],
        'features':screen['features'],'positive_random_ids':controls[:selected['count']],
        'negative_random_ids':controls[selected['count']:],'specs':specs,'tasks':tasks,
        'source':evidence(Path(__file__)),'formal_claim_allowed':False})


def generate(config,shard):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    root,_=paths(config)
    output=root/'matched_joint_control'
    protocol=read_json(output/'protocol.json')
    verify(protocol['source']);verify(protocol['checkpoint']);verify(protocol['selection'])
    directory=output/f'shard_{shard}'
    directory.mkdir(exist_ok=False)
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl') if r['analysis_length_label']=='short'}
    tasks=[t for i,t in enumerate(protocol['tasks']) if i%protocol['addon']['shards']==shard]
    teacher=config['teacher']['snapshot_path']
    tokenizer=AutoTokenizer.from_pretrained(teacher,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(teacher,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    short=[f['feature_id'] for f in protocol['features'] if f['direction']=='short']
    long=[f['feature_id'] for f in protocol['features'] if f['direction']=='long']
    controller=JointRandomController(positive_ids=protocol['positive_random_ids'],negative_ids=protocol['negative_random_ids'],
        short_ids=short,long_ids=long,random_ids=protocol['positive_random_ids'],torch_module=torch,
        checkpoint_path=protocol['checkpoint']['path'],layer_module=model.model.layers[17],k=64,
        maximum_delta_fraction=.30,device=model.device,dtype=torch.bfloat16)
    filename=directory/'predictions.jsonl';count=0
    with filename.open('x') as handle:
        for task in tasks:
            batch=[corpus[p] for p in task['problem_ids']]
            for spec in protocol['specs']:
                rows=generate_condition(model,tokenizer,controller,spec,batch,config['generation'])
                for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
                handle.flush()
            print(json.dumps({'shard':shard,'records':count,'split':task['split']}),flush=True)
    write_json_exclusive(directory/'COMPLETE.json',{'status':'complete','records':count,
        'predictions':evidence(filename),'protocol':evidence(output/'protocol.json')})


def analyze(config):
    from .verifiers import extract_final_answer,verify_answer
    root,_=paths(config)
    output=root/'matched_joint_control';protocol=read_json(output/'protocol.json')
    rows=[]
    for shard in range(protocol['addon']['shards']):
        marker=read_json(output/f'shard_{shard}'/'COMPLETE.json')
        verify(marker['predictions']);verify(marker['protocol'])
        rows.extend(jsonl(marker['predictions']['path']))
    expected={(t['split'],pid,s['name']) for t in protocol['tasks'] for pid in t['problem_ids'] for s in protocol['specs']}
    keys=[(r['question_split'],r['problem_id'],r['condition']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Missing or duplicate paired replay records')
    corpus={r['problem_id']:r for r in jsonl(root/'corpus.jsonl')}
    for row in rows:
        gold=extract_final_answer(corpus[row['problem_id']]['gold_answer'])
        if gold!=row['gold_answer'] or verify_answer(extract_final_answer(row['response']),gold)!=row['is_correct']:
            raise ValueError('Replay gold or correctness mismatch')
    gate=config['selection_gate'];summaries={}
    for split in ['dev','test']:
        arms={name:{r['problem_id']:r for r in rows if r['question_split']==split and r['condition']==name}
              for name in protocol['addon']['conditions']}
        base=arms['no_steering_replay'];target=arms['selected_target_replay'];random=arms['matched_joint_random']
        if any(base[p]['seed']!=target[p]['seed'] or base[p]['seed']!=random[p]['seed'] for p in base):
            raise ValueError('Replay RNG is unpaired')
        def contrast(left,right,field):
            return paired_question_bootstrap({p:float(r[field]) for p,r in left.items()},
                {p:float(r[field]) for p,r in right.items()},samples=gate['bootstrap_samples'],seed=gate['bootstrap_seed'])
        metrics={name:{'n':len(cell),'accuracy':float(np.mean([r['is_correct'] for r in cell.values()])),
                       'mean_tokens':float(np.mean([r['output_token_count'] for r in cell.values()])),
                       'hit_cap_fraction':float(np.mean([r['hit_max_new_tokens'] for r in cell.values()]))}
                 for name,cell in arms.items()}
        length=contrast(target,base,'output_token_count');accuracy=contrast(target,base,'is_correct')
        random_length=contrast(target,random,'output_token_count')
        reduction=1-metrics['selected_target_replay']['mean_tokens']/metrics['no_steering_replay']['mean_tokens']
        passed=(reduction>=gate['minimum_relative_length_reduction'] and length['ci_high']<0
                and accuracy['estimate']>=-gate['maximum_accuracy_drop'] and random_length['ci_high']<0)
        summaries[split]={'metrics':metrics,'length_minus_base':length,'accuracy_minus_base':accuracy,
                          'length_minus_structural_random':random_length,'relative_length_reduction':reduction,'passes_gate':passed}
    write_json_exclusive(output/'analysis.json',{'status':'complete','row_count':len(rows),'summaries':summaries,
                         'control_matches':'16 positive plus 16 negative directions, separately normalized groups, same residual norm',
                         'formal_claim_allowed':False})
    write_json_exclusive(root/'MATCHED_CONTROL_DECISION.json',{
        'selected_condition':protocol['addon']['selected_condition'],'dev_passed':summaries['dev']['passes_gate'],
        'test_passed':summaries['test']['passes_gate'],'matched_control_gate_passed':all(s['passes_gate'] for s in summaries.values()),
        'original_test_gate_still_required':True,'analysis':evidence(output/'analysis.json')})
    print(json.dumps(summaries,indent=2),flush=True)
