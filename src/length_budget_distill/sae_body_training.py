"""Build an answer-free, trace-balanced SAE sample from audited causal activations."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re

import numpy as np

from .factorial import canonical_sha256, file_sha256, read_key_value_marker
from .ncsu_reproduction import save, seal, verify
from .records import read_jsonl, write_jsonl
from .sae_sampling import maximally_equal_trace_quotas, deterministic_positions_for_quotas, activation_normalization


def body_positions(tokenizer, solution, settings):
    """Only text before the first explicit answer heading can enter the SAE."""
    pattern = re.compile(settings['answer_heading_pattern'])
    matches = list(pattern.finditer(solution))
    if len(matches) != 1:
        return None, 'missing_heading' if not matches else 'multiple_headings'
    encoded = tokenizer(solution, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    boundary = matches[0].start('marker') if 'marker' in pattern.groupindex else matches[0].start()
    # A token crossing the marker boundary is excluded in its entirety.
    before = [i for i, (left, right) in enumerate(offsets) if right <= boundary and right > left]
    if len(before) <= settings['exclude_last_body_tokens']:
        return None, 'short_body'
    stop = before[-settings['exclude_last_body_tokens']]
    excluded = set(settings['exclude_words'])
    eligible = []
    for i in before:
        if i >= stop or i >= settings['maximum_native_position']:
            continue
        piece = tokenizer.decode([ids[i]], skip_special_tokens=False)
        words = set(re.findall(r'[a-z]+', piece.lower()))
        if words & excluded or '\\' in piece or not any(c.isalnum() for c in piece):
            continue
        eligible.append(i)
    if len(eligible) < settings['minimum_eligible_positions']:
        return None, 'insufficient_clean_positions'
    return {'token_ids': ids, 'eligible_positions': eligible,
            'body_cutoff_token': before[-1] + 1, 'marker_byte_offset': boundary}, None


def paired_trace_rows(rows):
    """Keep short/long pairs even when the source question has other rollouts."""
    labels=defaultdict(set)
    for row in rows:labels[(row['question_split'],row['problem_id'])].add(row['analysis_length_label'])
    paired={key for key,value in labels.items() if {'short','long'}.issubset(value)}
    return [r for r in rows if (r['question_split'],r['problem_id']) in paired and
            r['analysis_length_label'] in ('short','long')]


def prepare(cfg):
    from transformers import AutoTokenizer
    source = Path(cfg['source_root']); root = Path(cfg['result_root'])
    verify(root/'protocol/FROZEN.json')
    parent = source/'sae'; p = parent/'protocol/frozen_protocol.json'
    source_cfg = json.loads(p.read_text()); corpus = parent/'corpus/mixed_trajectories.jsonl'
    marker = read_key_value_marker(parent/'corpus/CORPUS_COMPLETE')
    if marker.get('corpus_sha256') != file_sha256(corpus):
        raise ValueError('Parent SAE corpus marker does not verify')
    if source_cfg['teacher']['tokenizer_json_sha256'] != file_sha256(Path(source_cfg['teacher']['snapshot_path'])/'tokenizer.json'):
        raise ValueError('Frozen teacher tokenizer changed')
    tok = AutoTokenizer.from_pretrained(source_cfg['teacher']['snapshot_path'],local_files_only=True)
    rows = list(read_jsonl(corpus)); kept=[]; excluded=Counter(); split_counts=Counter()
    if [r['corpus_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Parent corpus index changed')
    for row in rows:
        value, why = body_positions(tok,row['solution'],cfg['clean_tokens'])
        if why:
            excluded[why]+=1; continue
        if len(value['token_ids']) != row['solution_token_count']:
            raise ValueError('Parent corpus tokenization changed: '+row['trace_id'])
        retained = {k:row[k] for k in ('corpus_index','trace_id','problem_id','question_split','analysis_length_label')}
        retained.update(value);kept.append(retained);split_counts[row['question_split']]+=1
    # A question must retain both length labels before it enters the comparison.
    balanced=paired_trace_rows(kept)
    excluded['unpaired_after_cleaning']=len(kept)-len(balanced)
    if any(sum(r['question_split']==s for r in balanced)<cfg['minimum_traces_per_split'][s] for s in ('train','dev','test')):
        raise ValueError('Insufficient paired trace support after answer removal')
    out=root/'inputs';write_jsonl(out/'eligible_traces.jsonl',balanced)
    counts={s:sum(r['question_split']==s for r in balanced) for s in ('train','dev','test')}
    save(out/'support_audit.json',{'parent_traces':len(rows),'eligible_before_pairing':len(kept),
         'eligible_after_pairing':len(balanced),'split_traces':counts,'excluded':dict(excluded),
         'heading_required':True,'removed_answer_and_all_following_tokens':True,
         'old_phase5_confirmations_are_not_independent':True})
    seal(out/'COMPLETE.json',[root/'protocol/FROZEN.json',corpus,p,out/'eligible_traces.jsonl',out/'support_audit.json'],formal_claim_allowed=False)


def sample(cfg):
    import torch
    from safetensors.torch import load_file, save_file
    from .sae_sampling import token_priorities
    source=Path(cfg['source_root'])/'sae';root=Path(cfg['result_root'])
    verify(root/'inputs/COMPLETE.json')
    rows=list(read_jsonl(root/'inputs/eligible_traces.jsonl'))
    parent_cfg=json.loads((source/'protocol/frozen_protocol.json').read_text())
    hidden=parent_cfg['teacher']['hidden_size'];layer_index=cfg['layer_index']
    selected={};slot_counts={};quotas_by_split={};counts={}
    for split_code,split in enumerate(('train','dev','test')):
        available={r['corpus_index']:len(r['eligible_positions']) for r in rows if r['question_split']==split}
        target=cfg['sample_tokens'][split]
        quotas=maximally_equal_trace_quotas(available,target_tokens=target,seed=cfg['sample_seed'])
        # Priorities use the native position; no reindexing after filtering.
        for row in rows:
            if row['question_split']!=split:continue
            trace=row['corpus_index'];positions=np.asarray(row['eligible_positions'],dtype=np.int64)
            token_ids=np.asarray(row['token_ids'],dtype=np.int64)[positions]
            priority=token_priorities(np.full(len(positions),trace),positions,token_ids,
                seed=cfg['sample_seed'],layer_index=layer_index,split_code=split_code)
            chosen=np.argsort(priority,kind='stable')[:quotas[trace]]
            for i in chosen:
                position=int(positions[i]);selected[(trace,position)]=(split,int(token_ids[i]),int(priority[i]))
        quotas_by_split[split]={'traces':len(available),'target_tokens':target,'minimum_quota':min(quotas.values()),'maximum_quota':max(quotas.values())}
        slot_counts[split]=target;counts[split]=0
    tensors={split:{'activations':torch.empty((n,hidden),dtype=torch.bfloat16),
                    'trace_indices':torch.empty(n,dtype=torch.int32),'positions':torch.empty(n,dtype=torch.int32),
                    'token_ids':torch.empty(n,dtype=torch.int32),'priorities':torch.empty(n,dtype=torch.int64)}
             for split,n in slot_counts.items()}
    shard_bindings=[];chunk_count=0
    for shard in range(parent_cfg['activation_extraction']['trajectory_shards']):
        shard_dir=source/'activations'/f'shard_{shard:02d}_of_{parent_cfg["activation_extraction"]["trajectory_shards"]:02d}'
        manifest_path=shard_dir/'activation_manifest.json'
        marker=read_key_value_marker(shard_dir/'ACTIVATIONS_COMPLETE')
        if marker.get('manifest_sha256')!=file_sha256(manifest_path):raise ValueError('Parent activation manifest changed')
        manifest=json.loads(manifest_path.read_text());shard_bindings.append(manifest_path)
        layer=next(x for x in manifest['layers'] if x['layer_index']==layer_index)
        for chunk in layer['chunks']:
            p=Path(chunk['path'])
            if file_sha256(p)!=chunk['sha256']:raise ValueError('Parent activation chunk changed')
            values=load_file(str(p),device='cpu');traces=values['trace_indices'].tolist();positions=values['positions'].tolist()
            picks=[];where=[]
            for i,(trace,pos) in enumerate(zip(traces,positions)):
                key=(trace,pos)
                item=selected.pop(key,None)
                if item is None:continue
                split,token_id,priority=item
                if int(values['token_ids'][i])!=token_id:raise ValueError('Token ID changed at selected position')
                picks.append(i);where.append((split,counts[split],trace,pos,token_id,priority));counts[split]+=1
            if picks:
                for index,(split,slot,trace,pos,token_id,priority) in zip(picks,where):
                    dst=tensors[split];dst['activations'][slot]=values['activations'][index]
                    dst['trace_indices'][slot]=trace;dst['positions'][slot]=pos;dst['token_ids'][slot]=token_id
                    dst['priorities'][slot]=np.uint64(priority).view(np.int64)
            chunk_count+=1
    if selected or counts!=slot_counts:raise ValueError('Missing/duplicate selected activation positions')
    out=root/'token_samples';out.mkdir(parents=True,exist_ok=False)
    mean,scale,norm=activation_normalization(tensors['train']['activations'])
    normalizer=out/f'layer_{layer_index:02d}_normalizer.safetensors'
    save_file({'mean':mean.contiguous(),'scale':scale.reshape(1)},str(normalizer))
    samples=[]
    for split in ('train','dev','test'):
        path=out/f'layer_{layer_index:02d}_{split}.safetensors';save_file(tensors[split],str(path))
        samples.append({'split':split,'path':str(path),'sha256':file_sha256(path),'sampled_tokens':counts[split],
                        'available_tokens':sum(len(r['eligible_positions']) for r in rows if r['question_split']==split)})
    training=json.loads((source/'protocol/frozen_protocol.json').read_text())
    training['experiment_name']=cfg['experiment_id'];training['sae']['seed']=cfg['sae_seed']
    training['activation_extraction']['layer_indices_zero_based']=[layer_index]
    training['sae']['k_values']=[cfg['k']]
    save(root/'protocol/training_config.json',training)
    manifest={'status':'complete','experiment_name':cfg['experiment_id'],'config_hash':canonical_sha256(training),
              'config_path':str(root/'protocol/training_config.json'),'config_sha256':file_sha256(root/'protocol/training_config.json'),
              'parent_corpus_path':str(source/'corpus/mixed_trajectories.jsonl'),
              'parent_corpus_sha256':file_sha256(source/'corpus/mixed_trajectories.jsonl'),
              'layers':[{'layer_index':layer_index,'samples':samples,'normalizer_path':str(normalizer),
                         'normalizer_sha256':file_sha256(normalizer),'normalization':norm}],
              'quotas':quotas_by_split,'activation_manifests':[{'path':str(p),'sha256':file_sha256(p)} for p in shard_bindings],
              'parent_activation_chunks_verified':chunk_count,'answer_and_suffix_excluded':True,
              'formal_claim_allowed':False}
    save(out/'sample_manifest.json',manifest)
    (out/'TOKEN_SAMPLES_COMPLETE').write_text(f'status=complete\nconfig_hash={manifest["config_hash"]}\nmanifest_sha256={file_sha256(out/"sample_manifest.json")}\n')
    seal(out/'COMPLETE.json',[root/'inputs/COMPLETE.json',out/'sample_manifest.json',normalizer,
         *[Path(x['path']) for x in samples]],formal_claim_allowed=False)
