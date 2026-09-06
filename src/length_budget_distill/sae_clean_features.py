"""Fixed-support SAE feature screening without answer or length denominators."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .experiment_io import read_json, write_json_exclusive
from .factorial import canonical_sha256, file_sha256, read_key_value_marker
from .sae_ablation import load_topk_sae
from .sae_feature_analysis import paired_feature_statistics, holm_adjust


def clean_positions(tokenizer, solution, settings):
    encoded = tokenizer(solution, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    boundary = re.search(r'(?im)^\s*(?:final\s+)?answer\s*:', solution)
    answer_start = boundary.start() if boundary else len(solution)
    limit = min(len(ids) - settings['exclude_last_tokens'], settings['maximum_native_position'])
    excluded = set(settings['exclude_words'])
    positions = []
    for position in range(max(0, limit)):
        start, end = encoded['offset_mapping'][position]
        piece = tokenizer.decode([ids[position]])
        words = set(re.findall(r'[a-z]+', piece.lower()))
        if end > answer_start or start >= answer_start:
            continue
        if words & excluded or '\\' in piece or not any(c.isalnum() for c in piece):
            continue
        positions.append(position)
    selected = positions[:settings['count']]
    return ids, selected


def freeze_clean_protocol(config_path, project):
    from transformers import AutoTokenizer, AutoConfig
    config = read_json(config_path)
    parent_path = project / config['parent_protocol']
    parent = read_json(parent_path)
    if read_key_value_marker(parent_path.parent/'PROTOCOL_FROZEN')['config_hash'] != canonical_sha256(parent):
        raise ValueError('Parent protocol hash mismatch')
    for name in ('checkpoint', 'corpus'):
        if file_sha256(parent['parent_evidence'][name+'_path']) != parent['parent_evidence'][name+'_sha256']:
            raise ValueError('Parent input hash mismatch: '+name)
    config['teacher'] = parent['teacher']
    config['parent_sae'] = parent['parent_sae']
    config['parent_evidence'] = parent['parent_evidence']
    config['old_selected_features'] = parent['selected_features']
    config['parent_protocol_sha256'] = file_sha256(parent_path)
    tokenizer = AutoTokenizer.from_pretrained(parent['teacher']['snapshot_path'], local_files_only=True)
    config['teacher']['hidden_size']=int(AutoConfig.from_pretrained(parent['teacher']['snapshot_path'],local_files_only=True).hidden_size)
    corpus = [json.loads(line) for line in Path(parent['parent_evidence']['corpus_path']).open()]
    eligible = []
    for row in corpus:
        if row['question_split'] not in ('dev','test') or not row['is_correct'] or row['analysis_length_label'] not in ('short','long'):
            continue
        ids, positions = clean_positions(tokenizer, row['solution'], config['clean_tokens'])
        if len(ids) != row['solution_token_count']:
            raise ValueError('Corpus/tokenizer length mismatch')
        if len(positions) != config['clean_tokens']['count']:
            continue
        pieces = [tokenizer.decode([ids[i]]) for i in positions]
        eligible.append({k:row[k] for k in ('corpus_index','trace_id','problem_id','question_split','analysis_length_label','solution_token_count')} | {
            'positions':positions,'token_ids':[ids[i] for i in positions],
            'nuisance':[float(np.mean([any(c.isdigit() for c in p) for p in pieces])),float(np.mean(positions))/config['clean_tokens']['maximum_native_position'],(positions[-1]-positions[0])/config['clean_tokens']['maximum_native_position']]})
    support=defaultdict(set)
    for r in eligible:support[(r['question_split'],r['problem_id'])].add(r['analysis_length_label'])
    eligible=[r for r in eligible if len(support[(r['question_split'],r['problem_id'])])==2]
    config['activation_shards']=[]
    for shard in range(config['scoring']['shards']):
        root=project/config['activation_root']/f'shard_{shard:02d}_of_{config["scoring"]["shards"]:02d}'
        manifest_path=root/'activation_manifest.json';m=read_json(manifest_path)
        marker=read_key_value_marker(root/'ACTIVATIONS_COMPLETE')
        if marker['manifest_sha256']!=file_sha256(manifest_path):raise ValueError('Activation manifest hash mismatch')
        layer=next(r for r in m['layers'] if r['layer_index']==config['parent_sae']['layer_index'])
        config['activation_shards'].append({'manifest_path':str(manifest_path),'manifest_sha256':file_sha256(manifest_path),'chunks':layer['chunks']})
    root=project/config['outputs']['result_root'];protocol=root/'protocol';protocol.mkdir(parents=True,exist_ok=False)
    rows_path=protocol/'eligible_traces.jsonl'
    rows_path.write_text(''.join(json.dumps(r)+'\n' for r in eligible))
    config['eligible_traces_path']=str(rows_path);config['eligible_traces_sha256']=file_sha256(rows_path)
    config['cohort_audit']={split:{'traces':sum(r['question_split']==split for r in eligible),'questions':len({r['problem_id'] for r in eligible if r['question_split']==split})} for split in ('dev','test')}
    config['eligible_trace_count']=len(eligible)
    frozen=protocol/'frozen_protocol.json';write_json_exclusive(frozen,config)
    (protocol/'PROTOCOL_FROZEN').write_text(f'status=frozen\nconfig_hash={canonical_sha256(config)}\nconfig_sha256={file_sha256(frozen)}\n')
    print(json.dumps({'status':'frozen','cohorts':config['cohort_audit'],'path':str(frozen)}),flush=True)


def load_clean_protocol(path):
    config=read_json(path);marker=read_key_value_marker(path.parent/'PROTOCOL_FROZEN')
    if marker['config_hash']!=canonical_sha256(config) or marker['config_sha256']!=file_sha256(path):raise ValueError('Frozen config changed')
    if file_sha256(config['eligible_traces_path'])!=config['eligible_traces_sha256']:raise ValueError('Eligible rows changed')
    return config,[json.loads(l) for l in Path(config['eligible_traces_path']).open()]


def score_clean_shard(path,project,shard):
    import torch
    from safetensors.torch import load_file
    config,rows=load_clean_protocol(path)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    source=config['activation_shards'][shard]
    if file_sha256(source['manifest_path'])!=source['manifest_sha256']:raise ValueError('Activation metadata changed')
    checkpoint=Path(config['parent_evidence']['checkpoint_path'])
    if file_sha256(checkpoint)!=config['parent_evidence']['checkpoint_sha256']:raise ValueError('SAE checkpoint changed')
    width=config['clean_tokens']['count'];features=config['parent_sae']['feature_count'];k=config['parent_sae']['k']
    dim=config['teacher']['hidden_size']
    model,mean,scale=load_topk_sae(checkpoint,input_dim=dim,feature_count=features,k=k,device=torch.device('cuda'))
    wanted={int(r['corpus_index']):{position:slot for slot,position in enumerate(r['positions'])} for r in rows}
    by_corpus={r['corpus_index']:i for i,r in enumerate(rows)}
    values=np.zeros((len(rows),width,k),dtype=np.float32);indices=np.zeros((len(rows),width,k),dtype=np.int32);seen=np.zeros((len(rows),width),dtype=np.uint8)
    replay_slots=config['scoring']['readback_positions']
    hidden_sample=np.zeros((len(rows),len(replay_slots),dim),dtype=np.float32)
    batch=config['scoring']['batch_size']
    for number,chunk in enumerate(source['chunks']):
        if file_sha256(chunk['path'])!=chunk['sha256']:raise ValueError('Activation chunk changed')
        tensors=load_file(chunk['path'],device='cpu');trace=tensors['trace_indices'].tolist();position=tensors['positions'].tolist()
        take=[i for i,(t,p) in enumerate(zip(trace,position)) if t in wanted and p in wanted[t]]
        if take:
            h=tensors['activations'][take];rr=np.array([by_corpus[trace[i]] for i in take]);ss=np.array([wanted[trace[i]][position[i]] for i in take])
            if np.any(seen[rr,ss]):raise ValueError('Duplicate activation positions')
            expected_tokens=[rows[r]['token_ids'][s] for r,s in zip(rr,ss)]
            if tensors['token_ids'][take].tolist()!=expected_tokens:raise ValueError('Activation token alignment mismatch')
            for start in range(0,len(take),batch):
                stop=min(start+batch,len(take));normalized=(h[start:stop].to('cuda').float()-mean)*scale
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):v,i,_=model.encode(normalized)
                values[rr[start:stop],ss[start:stop]]=v.float().cpu().numpy();indices[rr[start:stop],ss[start:stop]]=i.cpu().numpy()
            seen[rr,ss]=1
            for n,(r,s) in enumerate(zip(rr,ss)):
                if s in replay_slots:hidden_sample[r,replay_slots.index(s)]=h[n].float().numpy()
        del tensors
        if (number+1)%6==0:print(json.dumps({'event':'score_progress','shard':shard,'chunks':number+1,'total_chunks':len(source['chunks']),'tokens':int(seen.sum())}),flush=True)
    present=np.flatnonzero(seen.sum(axis=1)>0)
    if np.any(seen[present].sum(axis=1)!=width):raise ValueError('Partial trace in activation shard')
    output=project/config['outputs']['result_root']/f'score_shards/shard_{shard:02d}'
    output.mkdir(parents=True,exist_ok=False);data_path=output/'clean_codes.npz'
    np.savez(data_path,row_indices=present,values=values[present],indices=indices[present],hidden_sample=hidden_sample[present])
    manifest={'status':'complete','shard':shard,'config_hash':canonical_sha256(config),'data_path':str(data_path),'data_sha256':file_sha256(data_path),'trace_count':len(present),'token_count':int(seen.sum()),'source_sha256':file_sha256(Path(__file__))}
    write_json_exclusive(output/'manifest.json',manifest)
    (output/'SCORE_SHARD_COMPLETE').write_text(f'status=complete\nmanifest_sha256={file_sha256(output/"manifest.json")}\n')
    print(json.dumps(manifest),flush=True)
