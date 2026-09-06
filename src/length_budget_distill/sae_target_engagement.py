"""Read back targeted and off-target TopK changes before running generation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256, read_key_value_marker
from .sae_clean_features import load_clean_protocol


def freeze_engagement(config_path,project,version=1):
    config,rows=load_clean_protocol(config_path);root=project/config['outputs']['result_root']
    gate=root/'feature_gate';mp=gate/'feature_gate_manifest.json'
    if read_key_value_marker(gate/'FEATURE_GATE_COMPLETE')['manifest_sha256']!=file_sha256(mp):raise ValueError('Feature gate changed')
    selection=read_json(gate/'selected_features.json')
    if not selection['gate_passed']:raise ValueError('No clean feature passed')
    primary=selection['selected'][0]['feature_id'];size=config['parent_sae']['feature_count']
    counts=np.zeros(size);mass=np.zeros(size);ntokens=0
    evidence=[]
    for shard in range(config['scoring']['shards']):
        m=read_json(root/f'score_shards/shard_{shard:02d}/manifest.json')
        if file_sha256(m['data_path'])!=m['data_sha256']:raise ValueError('Codes changed')
        with np.load(m['data_path']) as data:
            keep=np.array([rows[int(i)]['question_split']=='dev' for i in data['row_indices']])
            values=data['values'][keep];indices=data['indices'][keep]
            if version==2:
                values=values[:,config['scoring']['readback_positions']];indices=indices[:,config['scoring']['readback_positions']]
            counts+=np.bincount(indices.ravel(),weights=(values>0).ravel(),minlength=size)
            mass+=np.bincount(indices.ravel(),weights=values.ravel(),minlength=size)
            ntokens+=values.shape[0]*values.shape[1]
        evidence.append({'path':m['data_path'],'sha256':m['data_sha256']})
    frequency=counts/ntokens;amplitude=mass/np.maximum(counts,1)
    excluded={r['feature_id'] for r in selection['candidates']}
    excluded.update(config['old_selected_features']['short_feature_ids']);excluded.update(config['old_selected_features']['long_feature_ids'])
    distance=np.abs(np.log(np.maximum(frequency,1e-12)/max(frequency[primary],1e-12)))+np.abs(np.log(np.maximum(amplitude,1e-12)/max(amplitude[primary],1e-12)))
    pool=[int(i) for i in np.argsort(distance) if int(i) not in excluded and frequency[i]>0 and (version==1 or frequency[i]>=config['engagement']['minimum_engaged_state_fraction'])][:32]
    controls=[]
    for seed in config['engagement']['control_seeds']:
        available=[i for i in pool if i not in controls]
        controls.append(int(np.random.default_rng(seed).choice(available)))
    features=[primary,*controls]
    protocol={'status':'frozen','config_hash':canonical_sha256(config),'feature_gate_manifest_sha256':file_sha256(mp),'primary_feature_id':primary,'feature_ids':features,'matching_method':'Seeded choice without replacement from 32 closest dev frequency/conditional-amplitude features; all discovery candidates and old target sets excluded','matching':{str(fid):{'frequency':float(frequency[fid]),'conditional_amplitude':float(amplitude[fid]),'log_distance':float(distance[fid])} for fid in features},'maximum_control_fold_difference':2.0,'batch_size':128,'inputs':evidence,'source_sha256':file_sha256(Path(__file__)),'formal_claim_allowed':False}
    protocol['version']=version
    protocol['matching_support']='all_clean_64_tokens' if version==1 else 'dev_readback_positions_only_with_minimum_dev_coverage'
    if version==2:
        old=root/'engagement_protocol/protocol.json'
        protocol['amendment_parent_sha256']=file_sha256(old)
        protocol['amendment_reason']='Control 882 failed readback coverage because matching used a different position distribution. Match on dev readback positions, preserve all thresholds, do not use confirmation outcomes to rank replacements.'
    out=root/('engagement_protocol' if version==1 else f'engagement_protocol_v{version}');out.mkdir(exist_ok=False);p=out/'protocol.json';write_json_exclusive(p,protocol)
    (out/'ENGAGEMENT_PROTOCOL_FROZEN').write_text(f'status=frozen\nprotocol_sha256={file_sha256(p)}\n')
    print(json.dumps(protocol),flush=True)


def run_engagement(config_path,project,feature_slot,version=1):
    import torch
    from safetensors.torch import load_file
    config,rows=load_clean_protocol(config_path);root=project/config['outputs']['result_root'];protocol_path=root/('engagement_protocol' if version==1 else f'engagement_protocol_v{version}')/'protocol.json';protocol=read_json(protocol_path)
    if read_key_value_marker(protocol_path.parent/'ENGAGEMENT_PROTOCOL_FROZEN')['protocol_sha256']!=file_sha256(protocol_path):raise ValueError('Engagement protocol changed')
    feature=protocol['feature_ids'][feature_slot]
    checkpoint=Path(config['parent_evidence']['checkpoint_path'])
    if file_sha256(checkpoint)!=config['parent_evidence']['checkpoint_sha256']:raise ValueError('SAE changed')
    tensors=load_file(str(checkpoint),device='cpu')
    torch.backends.cuda.matmul.allow_tf32=False
    encoder=tensors['encoder_weight'].float().cuda();bias=tensors['encoder_bias'].float().cuda();decoder_bias=tensors['decoder_bias'].float().cuda();mean=tensors['activation_mean'].float().cuda();scale=float(tensors['activation_scale'].item());direction=tensors['decoder_weight'][feature].float().cuda()
    states=[];state_ids=[]
    for item in protocol['inputs']:
        if file_sha256(item['path'])!=item['sha256']:raise ValueError('State input changed')
        with np.load(item['path']) as data:
            for idx,h in zip(data['row_indices'],data['hidden_sample']):
                if rows[int(idx)]['question_split']=='test':
                    states.extend(h);state_ids.extend([f'{rows[int(idx)]["trace_id"]}:slot_{s}' for s in config['scoring']['readback_positions']])
    states=np.array(states,dtype=np.float32);k=config['parent_sae']['k'];settings=config['engagement'];result=[];per_state=[]
    def encode(h):
        pre=torch.nn.functional.linear((h-mean)*scale-decoder_bias,encoder,bias)
        v,i=torch.topk(pre,k,dim=-1,sorted=False);v=v.relu()
        dense=torch.zeros_like(pre).scatter(1,i,v)
        return dense
    for strength in settings['signed_strengths']:
        before_sum=after_sum=target_sq=other_sq=fraction_sum=0.;active_count=clip_count=correct_direction=0
        with torch.inference_mode():
            for start in range(0,len(states),protocol['batch_size']):
                h=torch.from_numpy(states[start:start+protocol['batch_size']]).cuda();before=encode(h);z=before[:,feature]
                delta=float(strength)*z[:,None]*direction[None,:]/scale
                requested=delta.norm(dim=-1);maximum=settings['maximum_delta_fraction']*h.norm(dim=-1);multiplier=torch.minimum(torch.ones_like(requested),maximum/requested.clamp_min(1e-12))
                delta*=multiplier[:,None]
                changed_hidden=(h+delta).to(torch.bfloat16).float()
                delta=changed_hidden-h
                after=encode(changed_hidden);change=after-before;target_change=change[:,feature].clone();change[:,feature]=0
                off=change.norm(dim=-1);fraction=delta.norm(dim=-1)/h.norm(dim=-1).clamp_min(1e-12);active=z>0
                before_sum+=float(z.sum());after_sum+=float(after[:,feature].sum());target_sq+=float(target_change.square().sum());other_sq+=float(off.square().sum());fraction_sum+=float(fraction.sum());active_count+=int(active.sum());clip_count+=int((multiplier<1).sum());correct_direction+=int(((target_change*strength>0)&active).sum())
                for j in range(len(h)):
                    per_state.append({'state_id':state_ids[start+j],'feature_id':feature,'strength':strength,'activation_before':float(z[j]),'activation_after':float(after[j,feature]),'delta_fraction':float(fraction[j]),'off_target_l2':float(off[j]),'clipped':bool(multiplier[j]<1)})
        relative=(after_sum-before_sum)/max(before_sum,1e-12);coverage=active_count/len(states);off_ratio=float(np.sqrt(other_sq)/max(np.sqrt(target_sq),1e-12))
        reasons=[]
        if coverage<settings['minimum_engaged_state_fraction']:reasons.append('insufficient_active_state_coverage')
        if relative*np.sign(strength)<settings['minimum_target_change_fraction']:reasons.append('insufficient_target_activation_change')
        if off_ratio>settings['maximum_nontarget_change_ratio']:reasons.append('excessive_off_target_change')
        result.append({'feature_id':feature,'role':'primary' if feature_slot==0 else 'matched_random','strength':strength,'state_count':len(states),'active_states':active_count,'active_state_fraction':coverage,'mean_target_before':before_sum/len(states),'mean_target_after':after_sum/len(states),'relative_target_change':relative,'correct_direction_among_active':correct_direction/max(active_count,1),'off_target_to_target_l2_ratio':off_ratio,'mean_delta_fraction':fraction_sum/len(states),'clip_fraction':clip_count/len(states),'passed':not reasons,'rejection_reasons':reasons})
    out=root/('engagement_shards' if version==1 else f'engagement_shards_v{version}')/f'feature_{feature_slot:02d}';out.mkdir(parents=True,exist_ok=False);records=out/'state_readbacks.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in per_state))
    m={'status':'complete','config_hash':canonical_sha256(config),'protocol_sha256':file_sha256(protocol_path),'feature_slot':feature_slot,'feature_id':feature,'results':result,'state_readbacks_sha256':file_sha256(records),'state_count':len(states),'source_sha256':file_sha256(Path(__file__))}
    p=out/'manifest.json';write_json_exclusive(p,m);(out/'ENGAGEMENT_SHARD_COMPLETE').write_text(f'status=complete\nmanifest_sha256={file_sha256(p)}\n')
    print(json.dumps(m),flush=True)
